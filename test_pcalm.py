"""Numerical checks for the graph adapter and PC-ALM update convention."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from nematode_circuit_cnn_integrated import NematodeCircuitCNN
from pcalm import CircuitGraph, pcalm_gradients


class LinearGraph:
    def __init__(self, model):
        self.model = model
        self.names = ["hidden"]

    def prediction(self, name, states, x):
        return self.model[0](x)

    def output(self, states):
        return self.model[1](states["hidden"])

    def initialize(self, x):
        with torch.no_grad():
            h = {"hidden": self.prediction("hidden", {}, x)}
            return h, self.output(h)


class PCALMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_graph_is_original_forward_including_dropout_and_bn(self):
        model = NematodeCircuitCNN(c=2)
        model.eval()
        x = torch.randn(3, 3, 32, 32)
        model(x)  # materialize lazy parameters
        model.train()
        reference = copy.deepcopy(model)
        rng = torch.get_rng_state()
        with torch.no_grad():
            expected = reference(x)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        _, actual = CircuitGraph(model).initialize(x)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for a, b in zip(model.buffers(), reference.buffers()):
            torch.testing.assert_close(a, b)

    def test_inference_replays_dropout_and_updates_bn_only_once(self):
        model = NematodeCircuitCNN(c=2)
        model.eval()
        x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 1])
        model(x)
        model.train()
        reference = copy.deepcopy(model)
        rng = torch.get_rng_state()
        with torch.no_grad():
            reference(x)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        pcalm_gradients(model, x, y, steps=3)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for a, b in zip(model.buffers(), reference.buffers()):
            torch.testing.assert_close(a, b)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    @patch("pcalm.CircuitGraph", LinearGraph)
    def test_linear_convergence_recovers_bp_gradient_with_cross_entropy(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)).double()
        x = torch.tensor([[.2, -.4], [.5, .1]], dtype=torch.float64)
        y = torch.tensor([0, 1])
        F.cross_entropy(model(x), y, label_smoothing=0.).backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        pcalm_gradients(model, x, y, steps=500, state_lr=.1, dual_lr=.1, label_smoothing=0.)
        for p, grad in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad, grad, atol=1e-8, rtol=1e-6)

    @patch("pcalm.CircuitGraph", LinearGraph)
    def test_backtracking_reduces_an_excessive_primal_step(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).double()
        x = torch.tensor([[.2, -.4], [.5, .1]], dtype=torch.float64)
        y = torch.tensor([0, 1])
        diagnostic = {}
        pcalm_gradients(model, x, y, steps=3, state_lr=100., diagnostics=diagnostic)
        self.assertGreater(diagnostic["backtracks"], 0)
        self.assertLess(diagnostic["state_lr_min"], 100.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    @patch("pcalm.CircuitGraph", LinearGraph)
    def test_batch_duplication_preserves_state_dynamics_and_weight_gradients(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).double()
        other = copy.deepcopy(model)
        x = torch.tensor([[.2, -.4]], dtype=torch.float64)
        y = torch.tensor([0])
        a = pcalm_gradients(model, x, y, steps=5)
        b = pcalm_gradients(other, x.repeat(4, 1), y.repeat(4), steps=5)
        self.assertAlmostEqual(a[1], b[1], places=12)
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad)


if __name__ == "__main__":
    unittest.main()
