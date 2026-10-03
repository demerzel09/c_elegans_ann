"""Run paired BP / PC-ALM experiments on NematodeCircuitCNN."""
import argparse
import copy
import csv
import json
import random
import time
import os
import sys
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, FakeData

from nematode_circuit_cnn_integrated import NematodeCircuitCNN, build_transforms, evaluate
from pcalm import pcalm_gradients


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


TRAINING_KEYS = ("epochs", "c", "bs", "lr", "optimizer", "weight_decay",
                 "label_smoothing", "autoaugment", "seed", "steps", "state_lr",
                 "dual_lr", "rho", "train_samples", "test_samples", "synthetic",
                 "workers", "data_dir", "validation_samples", "backtracking", "log_gradients")

GRADIENT_PARAMETERS = ("stem.conv.weight", "node_ops.s0.conv.weight",
                       "node_ops.i0.conv.weight", "node_ops.m0.conv.weight",
                       "post.0.conv.weight", "post.2.conv.weight", "fc.1.weight",
                       "fc.3.weight", "fc.5.weight")


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reference_bp_gradients(model, x, y, label_smoothing):
    """Audit only: independent BP reference, without changing training RNG/BN."""
    devices = [x.device.index or 0] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        reference = copy.deepcopy(model)
        F.cross_entropy(reference(x), y, label_smoothing=label_smoothing).backward()
        return {name: p.grad.detach().clone() for name, p in reference.named_parameters()
                if name in GRADIENT_PARAMETERS and p.grad is not None}


def gradient_rows(model, reference, method, epoch):
    rows = []
    for name, p in model.named_parameters():
        if name not in reference or p.grad is None:
            continue
        grad, bp = p.grad.detach().flatten(), reference[name].flatten()
        norm, bp_norm = grad.norm().item(), bp.norm().item()
        rows.append(dict(method=method, epoch=epoch, parameter=name,
                         grad_norm=norm, bp_grad_norm=bp_norm,
                         norm_ratio=norm / bp_norm if bp_norm else None,
                         cosine_to_bp=F.cosine_similarity(grad, bp, dim=0).item(),
                         weight_norm=p.detach().norm().item()))
    return rows


def load_resume(args):
    if not args.resume:
        if args.extra_epochs or args.restart_lr is not None:
            raise ValueError("--extra-epochs / --restart-lr require --resume")
        return None
    if len(args.methods) != 1:
        raise ValueError("Resume one method at a time: --methods bp or pcalm")
    # Legacy files serialized torch.__version__ as this string subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != args.methods[0]:
        raise ValueError("Checkpoint method differs from --methods")
    saved = checkpoint["config"]
    for name in TRAINING_KEYS:
        if name in saved:
            setattr(args, name, saved[name])
    args.backtracking = saved.get("backtracking", False)
    if args.output is None:
        args.output = str(Path(args.resume).parent)
    epoch = checkpoint.get("epoch", saved["epochs"])
    if args.extra_epochs:
        args.epochs = epoch + args.extra_epochs
        args.lr = args.restart_lr if args.restart_lr is not None else saved["lr"]
        for name, value in args._inference_overrides.items():
            setattr(args, name, value)
    elif args.restart_lr is not None:
        raise ValueError("--restart-lr requires --extra-epochs")
    if args.epochs <= epoch:
        raise ValueError(f"Checkpoint already completed epoch {epoch}; use --extra-epochs 50 to continue")
    checkpoint["epoch"] = epoch
    return checkpoint


def make_scheduler(opt, plan):
    if plan["kind"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=plan["epochs"])
    epochs = plan["epochs"]
    warmup = max(1, min(10, epochs // 40))
    return torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))
    ], milestones=[warmup])


def rng_state(train_loader, test_loader):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_generator": train_loader.generator.get_state(),
            "test_generator": test_loader.generator.get_state()}


def restore_rng(state, train_loader, test_loader, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; RNG state cannot be restored exactly")
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["train_generator"])
    test_loader.generator.set_state(state["test_generator"])


def save_checkpoint(checkpoint, path):
    """Replace atomically so an interrupted write preserves the prior epoch."""
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_legacy_history(path, method):
    """Old final checkpoints stored metrics beside the model, not inside it."""
    csv_path = Path(path).parent / "metrics.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == method]
    for row in rows:
        row["epoch"] = int(row["epoch"])
        for key in ("train_loss", "test_accuracy", "train_seconds", "peak_cuda_mb", "lr", "residual_rms",
                    "state_lr_min", "backtracks"):
            row[key] = float(row[key]) if row.get(key) else None
    return rows


def run(args):
    if args.extra_epochs < 0 or (args.restart_lr is not None and args.restart_lr <= 0):
        raise ValueError("extra-epochs must be nonnegative; restart-lr must be positive")
    checkpoint = load_resume(args)
    if args.epochs < 1 or args.bs < 1 or args.c < 1 or args.workers < 0:
        raise ValueError("epochs, bs, c must be positive; workers must be nonnegative")
    if args.train_samples < 0 or args.test_samples < 0 or args.validation_samples < 0:
        raise ValueError("sample counts must be nonnegative")
    if args.steps < 1 or args.state_lr <= 0 or args.dual_lr < 0 or args.rho <= 0:
        raise ValueError("steps>=1, state-lr>0, dual-lr>=0, rho>0 required")
    if args.collapse_patience < 0:
        raise ValueError("collapse-patience must be nonnegative")
    if args.run_epochs is not None and args.run_epochs < 1:
        raise ValueError("run-epochs must be positive")
    if args.output is None:
        args.output = "results/" + (args.methods[0] if len(args.methods) == 1 else "bp_pcalm")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # Required by deterministic CUDA matrix multiplication.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    edges = checkpoint["config"].get("edges") if checkpoint else None
    if edges is None and args.edges_json:
        edges = json.loads(Path(args.edges_json).read_text(encoding="utf-8"))
    train_tf, test_tf = build_transforms(args.autoaugment)
    if args.synthetic:
        from torchvision.transforms import ToTensor
        train_ds = FakeData(args.train_samples or 32, (3, 32, 32), 10, ToTensor())
        test_ds = FakeData(args.test_samples or 16, (3, 32, 32), 10, ToTensor(), random_offset=100000)
    else:
        train_ds = CIFAR10(args.data_dir, train=True, download=True, transform=train_tf)
        if args.validation_samples:
            if args.validation_samples >= len(train_ds):
                raise ValueError("validation-samples must be smaller than the training dataset")
            ids = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed))
            validation_ds = CIFAR10(args.data_dir, train=True, download=False, transform=test_tf)
            test_ds = Subset(validation_ds, ids[:args.validation_samples].tolist())
            remaining = ids[args.validation_samples:]
            if args.train_samples:
                remaining = remaining[:args.train_samples]
            train_ds = Subset(train_ds, remaining.tolist())
        else:
            test_ds = CIFAR10(args.data_dir, train=False, download=True, transform=test_tf)
        if args.train_samples and not args.validation_samples:
            ids = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed))
            train_ds = Subset(train_ds, ids[:args.train_samples].tolist())
        if args.test_samples and not args.validation_samples:
            ids = torch.randperm(len(test_ds), generator=torch.Generator().manual_seed(args.seed + 1))
            test_ds = Subset(test_ds, ids[:args.test_samples].tolist())
    seed_all(args.seed)
    base = NematodeCircuitCNN(c=args.c, edges=edges).to(device)
    # Materialize LazyLinear without updating BN statistics or consuming dropout.
    base.eval()
    with torch.no_grad():
        base(torch.zeros(1, 3, 32, 32, device=device))
    initial = copy.deepcopy(base.state_dict())
    config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    config.update(device=str(device), torch_version=str(torch.__version__),
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  train_size=len(train_ds), test_size=len(test_ds),
                  parameters=sum(p.numel() for p in base.parameters()),
                  edges=base.edges, synthetic_results_are_accuracy_smoke_tests=args.synthetic)
    config["initial_model_sha256"] = model_digest(base)
    config["evaluation_split"] = "train_validation" if args.validation_samples else "test"
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, summary = [], {}
    grad_rows = checkpoint.get("gradient_history", []) if checkpoint else []
    for method in args.methods:
        model = copy.deepcopy(base)
        model.load_state_dict(checkpoint["model"] if checkpoint else initial)
        seed_all(args.seed + 2)
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                                  num_workers=args.workers, pin_memory=device.type == "cuda",
                                  generator=torch.Generator().manual_seed(args.seed + 3))
        test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                                 num_workers=args.workers,
                                 generator=torch.Generator().manual_seed(args.seed + 4))
        if args.optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                  weight_decay=args.weight_decay, nesterov=True)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        full_resume = checkpoint is not None and "optimizer" in checkpoint
        plan = (checkpoint["scheduler_plan"] if full_resume and not args.extra_epochs else
                {"kind": "cosine", "epochs": args.extra_epochs} if args.extra_epochs else
                {"kind": "warmup_cosine", "epochs": args.epochs})
        sched = make_scheduler(opt, plan)
        best, total_train_time = 0., 0.
        collapse_streak = checkpoint.get("collapse_streak", 0) if checkpoint else 0
        start_epoch = 0
        if checkpoint:
            start_epoch = checkpoint["epoch"]
            history = checkpoint.get("history")
            if history is None:
                history = read_legacy_history(args.resume, method)
            rows.extend(history)
            best = checkpoint.get("best_accuracy", max((r["test_accuracy"] for r in history), default=0.))
            total_train_time = checkpoint.get("total_train_seconds", sum(r["train_seconds"] for r in history))
            if full_resume:
                opt.load_state_dict(checkpoint["optimizer"])
                if args.extra_epochs:
                    for group in opt.param_groups:
                        group["lr"] = group["initial_lr"] = args.lr
                    sched = make_scheduler(opt, plan)
                else:
                    sched.load_state_dict(checkpoint["scheduler"])
                restore_rng(checkpoint["rng"], train_loader, test_loader, device)
            else:
                print("Legacy model: weights restored; optimizer, scheduler and RNG start fresh.", flush=True)
            print(f"{method}: loaded {args.resume}, continue epoch {start_epoch + 1} to {args.epochs}", flush=True)
        print(f"{method}: device={device}, save={out / (method + '.pt')}", flush=True)
        end_epoch = min(args.epochs, start_epoch + args.run_epochs) if args.run_epochs else args.epochs
        for ep in range(start_epoch + 1, end_epoch + 1):
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            start = time.perf_counter()
            loss_sum = rms_sum = seen = 0
            eta_min, backtrack_count = args.state_lr, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                reference = (reference_bp_gradients(model, x, y, args.label_smoothing)
                             if args.log_gradients and seen == 0 else None)
                if method == "bp":
                    loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
                    loss.backward()
                    loss_value, rms = loss.item(), 0.
                else:
                    diagnostics = {}
                    loss_value, rms = pcalm_gradients(model, x, y, steps=args.steps,
                        state_lr=args.state_lr, dual_lr=args.dual_lr, rho=args.rho,
                        label_smoothing=args.label_smoothing, backtracking=args.backtracking,
                        diagnostics=diagnostics)
                    eta_min = min(eta_min, diagnostics["state_lr_min"])
                    backtrack_count += diagnostics["backtracks"]
                if reference is not None:
                    grad_rows.extend(gradient_rows(model, reference, method, ep))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                loss_sum += loss_value * len(y)
                rms_sum += rms * len(y)
                seen += len(y)
            sync(device)
            seconds = time.perf_counter() - start
            total_train_time += seconds
            peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            acc = evaluate(model, test_loader, device)
            improved = acc > best or (not checkpoint and ep == 1)
            best = max(best, acc)
            row = dict(method=method, epoch=ep, train_loss=loss_sum/seen,
                       test_accuracy=acc, train_seconds=seconds, peak_cuda_mb=peak,
                       lr=opt.param_groups[0]["lr"], residual_rms=rms_sum/seen if method == "pcalm" else None,
                       state_lr_min=eta_min if method == "pcalm" else None,
                       backtracks=backtrack_count if method == "pcalm" else None)
            rows.append(row)
            if (method == "pcalm" and not args.synthetic and best >= .2 and
                    acc <= .12 and row["train_loss"] >= 2.25):
                collapse_streak += 1
            else:
                collapse_streak = 0
            with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            if grad_rows:
                with (out / "gradients.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(grad_rows[0]))
                    writer.writeheader()
                    writer.writerows(grad_rows)
            print(f"{method:5s} [{ep}/{args.epochs}] loss={row['train_loss']:.4f} "
                  f"acc={acc:.4f} train={seconds:.2f}s residual={row['residual_rms']}", flush=True)
            sched.step()
            summary[method] = dict(final_accuracy=acc, best_accuracy=best,
                                   total_train_seconds=total_train_time, epoch=ep)
            state = dict(format_version=3, model=model.state_dict(), config=config, method=method,
                         epoch=ep, optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                         scheduler_plan=plan, rng=rng_state(train_loader, test_loader),
                         best_accuracy=best, total_train_seconds=total_train_time,
                         history=[r for r in rows if r["method"] == method],
                         collapse_streak=collapse_streak,
                         gradient_history=[r for r in grad_rows if r["method"] == method])
            save_checkpoint(state, out / f"{method}.pt")
            if improved:
                save_checkpoint(state, out / f"{method}_best.pt")
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if args.collapse_patience and collapse_streak >= args.collapse_patience:
                raise RuntimeError("PC-ALM accuracy collapsed near chance for consecutive epochs. "
                                   "Latest and best checkpoints are retained; review LR and inference settings.")
        del model, opt, sched
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=["bp", "pcalm"], required=True,
                   help="Specify bp or pcalm explicitly")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--run-epochs", type=int,
                   help="Run at most this many epochs this invocation; keep the full --epochs schedule")
    p.add_argument("--log-gradients", action="store_true",
                   help="Audit first-batch parameter gradients against BP once per epoch (not used for learning)")
    p.add_argument("--c", type=int, default=48)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--autoaugment", action="store_true")
    p.add_argument("--edges-json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--state-lr", type=float, default=0.03)
    p.add_argument("--dual-lr", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=1.)
    p.add_argument("--train-samples", type=int, default=0, help="0: full CIFAR-10")
    p.add_argument("--test-samples", type=int, default=0, help="0: full test set")
    p.add_argument("--validation-samples", type=int, default=0,
                   help="Hold out this many training images instead of using the test set for tuning")
    p.add_argument("--backtracking", action=argparse.BooleanOptionalAction, default=True,
                   help="Safeguard primal steps; --no-backtracking uses the original fixed steps")
    p.add_argument("--collapse-patience", type=int, default=3,
                   help="Stop after this many chance-level epochs following learning; 0 disables")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", help="Default: results/bp or results/pcalm for individual runs")
    p.add_argument("--synthetic", action="store_true", help="FakeData smoke test only")
    p.add_argument("--resume", help="Checkpoint path; restores architecture and training settings")
    p.add_argument("--extra-epochs", type=int, default=0,
                   help="Train this many additional epochs from saved epoch with a new cosine schedule")
    p.add_argument("--restart-lr", type=float,
                   help="Initial LR for --extra-epochs (default: saved configuration LR)")
    supplied = sys.argv[1:] if argv is None else argv
    args = p.parse_args(supplied)
    flags = {arg.split("=", 1)[0] for arg in supplied if arg.startswith("--")}
    args._inference_overrides = {name: getattr(args, name) for flag, name in (
        ("--steps", "steps"), ("--state-lr", "state_lr"), ("--dual-lr", "dual_lr"),
        ("--rho", "rho"), ("--backtracking", "backtracking"),
        ("--no-backtracking", "backtracking")) if flag in flags}
    return args


if __name__ == "__main__":
    run(parse_args())
