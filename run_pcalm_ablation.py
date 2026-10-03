"""One controlled PC-ALM ablation per invocation; never silently run all cases."""
import argparse
import json
from pathlib import Path

from compare_bp_pcalm import parse_args as training_args, run


BASELINE = dict(lr=.1, steps=16, state_lr=.01, dual_lr=.1, backtracking=False,
                weight_decay=5e-4)
CHANGES = {
    "baseline": {},
    "lr_only": {"lr": .01},
    "steps_only": {"steps": 64},
    "state_lr_only": {"state_lr": .03},
    "dual_lr_only": {"dual_lr": 1.},
    "backtracking_only": {"backtracking": True},
    "inference_bundle": dict(steps=64, state_lr=.03, dual_lr=1.),
    "lr_inference": dict(lr=.01, steps=64, state_lr=.03, dual_lr=1.),
    "combined": dict(lr=.01, steps=64, state_lr=.03, dual_lr=1., backtracking=True),
    "weight_decay_only": {"weight_decay": 0.},
}


def case_settings(case):
    return {**BASELINE, **CHANGES[case]}


def build_training_args(args):
    setting = case_settings(args.case)
    output = Path(args.output_root) / f"{args.case}_seed{args.seed}"
    argv = ["--methods", "pcalm", "--epochs", str(args.training_epochs),
            "--run-epochs", str(args.run_epochs), "--c", str(args.c), "--bs", str(args.bs),
            "--seed", str(args.seed), "--lr", str(setting["lr"]),
            "--weight-decay", str(setting["weight_decay"]),
            "--steps", str(setting["steps"]), "--state-lr", str(setting["state_lr"]),
            "--dual-lr", str(setting["dual_lr"]), "--rho", "1",
            "--backtracking" if setting["backtracking"] else "--no-backtracking",
            "--train-samples", str(args.train_samples),
            "--validation-samples", str(args.validation_samples),
            "--collapse-patience", "0", "--log-gradients",
            "--device", args.device, "--data-dir", args.data_dir, "--output", str(output)]
    if args.edges_json:
        argv += ["--edges-json", args.edges_json]
    if args.smoke:
        argv += ["--synthetic", "--c", "2", "--bs", "2", "--train-samples", "4",
                 "--test-samples", "4", "--validation-samples", "0"]
    if args.resume:
        argv += ["--resume", str(output / "pcalm.pt")]
    configured = training_args(argv)
    configured.experiment_case = args.case
    return configured


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", choices=list(CHANGES))
    selection.add_argument("--list", action="store_true")
    p.add_argument("--training-epochs", type=int, default=200)
    p.add_argument("--run-epochs", type=int, default=12)
    p.add_argument("--c", type=int, default=48)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-samples", type=int, default=0)
    p.add_argument("--validation-samples", type=int, default=5000)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--device", default="auto")
    p.add_argument("--edges-json")
    p.add_argument("--output-root", default="results/pcalm_ablation")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny FakeData plumbing check, not a scientific result")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.list:
        print(json.dumps({case: case_settings(case) for case in CHANGES}, indent=2))
        return
    configured = build_training_args(args)
    print(f"Case: {args.case}; settings: {case_settings(args.case)}", flush=True)
    print(f"Planned schedule: {configured.epochs}; invocation limit: {configured.run_epochs}", flush=True)
    run(configured)


if __name__ == "__main__":
    main()
