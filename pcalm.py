"""Finite-inference PC-ALM for the existing feedforward circuit CNN.

Algorithm 1 and feedforward graph extension of arXiv:2605.31022.
The default primal step adds objective-decrease backtracking; disabling it
uses the paper's fixed-step update order.
The supervised output uses this project's cross entropy, not the paper's MSE.
Autograd differentiates individual blocks with independent state leaves;
it never differentiates through the inference trajectory or across blocks.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from nematode_circuit_cnn_integrated import S_NODES, M_NODES


class CircuitGraph:
    """The exact existing model, split at circuit nodes and hidden blocks."""

    def __init__(self, model):
        self.model = model
        self.order = model._toposort()
        self.names = ["stem", *self.order, "post1", "post2", "fc1", "fc2"]

    def prediction(self, name, states, x):
        m = self.model
        if name == "stem":
            return m.pool1(m.stem(x))
        if name in self.order:
            inputs = states["stem"] if name in S_NODES else sum(
                states[p] for p in m.parents[name]
            )
            return m.node_ops[name](inputs)
        if name == "post1":
            return m.post[1](m.post[0](torch.cat([states[n] for n in M_NODES], 1)))
        if name == "post2":
            return m.post[4](m.post[3](m.post[2](states["post1"])))
        if name == "fc1":
            return m.fc[2](m.fc[1](m.fc[0](states["post2"])))
        if name == "fc2":
            return m.fc[4](m.fc[3](states["fc1"]))
        raise KeyError(name)

    def output(self, states):
        return self.model.fc[5](states["fc2"])

    @torch.no_grad()
    def initialize(self, x):
        states = {}
        for name in self.names:
            states[name] = self.prediction(name, states, x)
        return states, self.output(states)


@contextmanager
def fixed_batch_functions(model, rng_cpu, rng_cuda, device):
    """Replay one dropout mask; update BN running statistics only at init.

BN continues to use batch statistics (as in BP). fork_rng prevents inference
iterations from consuming the random stream used by later batches.
"""
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, device)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, buffer in model.named_buffers():
                    buffer.copy_(buffers[name])


def pcalm_gradients(model, x, y, *, steps=64, state_lr=0.03,
                    dual_lr=1.0, rho=1.0, label_smoothing=0.1,
                    backtracking=True, diagnostics=None):
    """Populate parameter .grad; caller applies the same optimizer as BP.

    L = CE(sum) + sum_nodes [lambda*r + rho/2*r^2], r=h-f(parents).
    States update simultaneously; duals use residuals AFTER the primal step.
    The last primal step has no dual update, following Algorithm 1.
    State updates use summed per-sample energies, weight updates batch means.
    """
    if steps < 1 or state_lr <= 0 or dual_lr < 0 or rho <= 0:
        raise ValueError("steps>=1, state_lr>0, dual_lr>=0, rho>0 required")
    graph = CircuitGraph(model)
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
    states, initial_logits = graph.initialize(x)
    initial_loss = F.cross_entropy(initial_logits, y, label_smoothing=label_smoothing)
    duals = {n: torch.zeros_like(h) for n, h in states.items()}
    params = list(model.parameters())
    original_flags = [p.requires_grad for p in params]
    eta = state_lr
    reductions = 0

    def residuals(h):
        return {n: h[n] - graph.prediction(n, h, x) for n in graph.names}

    def energy(h):
        logits = graph.output(h)
        # Objective comparisons need more precision than the small FP32 state
        # updates; double reduction keeps backtracking from accepting increases
        # hidden by roundoff in the much larger supervised loss.
        result = F.cross_entropy(logits.double(), y, reduction="sum", label_smoothing=label_smoothing)
        rs = residuals(h)
        for n, r in rs.items():
            result = result + (duals[n] * r + 0.5 * rho * r.square()).sum(dtype=torch.float64)
        return result, rs

    try:
        for p in params:
            p.requires_grad_(False)
        for t in range(steps):
            states = {n: h.detach().requires_grad_(True) for n, h in states.items()}
            with fixed_batch_functions(model, rng_cpu, rng_cuda, x.device):
                total, _ = energy(states)
                grads = torch.autograd.grad(total, tuple(states.values()))
            if backtracking:
                current = total.detach().item()
                grad_sq = sum(g.detach().square().sum(dtype=torch.float64) for g in grads).item()
                # Minimize AL at FIXED duals; then perform dual ascent. This is
                # a safeguarded primal-step variant of the fixed-step algorithm.
                for attempt in range(20):
                    candidate = {n: (h - eta * g).detach()
                                 for (n, h), g in zip(states.items(), grads)}
                    with torch.no_grad(), fixed_batch_functions(model, rng_cpu, rng_cuda, x.device):
                        trial, rs = energy(candidate)
                    trial_value = trial.item()
                    tolerance = 1e-8 * max(1., abs(current))
                    if trial_value <= current - 1e-4 * eta * grad_sq + tolerance:
                        break
                    eta *= .5
                    reductions += 1
                else:
                    raise FloatingPointError("PC-ALM primal step could not decrease its objective")
                states = candidate
            else:
                states = {n: (h - eta * g).detach()
                          for (n, h), g in zip(states.items(), grads)}
                with torch.no_grad(), fixed_batch_functions(model, rng_cpu, rng_cuda, x.device):
                    rs = residuals(states)
            if t < steps - 1:
                duals = {n: duals[n] + dual_lr * rs[n] for n in graph.names}
        for p, flag in zip(params, original_flags):
            p.requires_grad_(flag)
        # Each block receives independent, detached parent states here.
        with fixed_batch_functions(model, rng_cpu, rng_cuda, x.device):
            total, rs = energy(states)
            (total / len(y)).backward()
        residual_rms = torch.sqrt(sum(r.detach().square().sum() for r in rs.values()) /
                                  sum(r.numel() for r in rs.values()))
        checks = [torch.isfinite(p.grad).all() for p in params if p.grad is not None]
        if checks and not torch.stack(checks).all().item():
            raise FloatingPointError("PC-ALM diverged; reduce --state-lr / --dual-lr")
        if diagnostics is not None:
            diagnostics.update(state_lr_min=eta, backtracks=reductions)
        return initial_loss.item(), residual_rms.item()
    finally:
        for p, flag in zip(params, original_flags):
            p.requires_grad_(flag)
