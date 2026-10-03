"""Run explicitly selected ablations sequentially and save live progress/results.

The scientific gate requires baseline collapse before advancing past baseline
and lr_only. Each case has its own process, log and epoch checkpoints.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from run_pcalm_ablation import CHANGES, build_training_args, parse_args as case_args


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def inspect_case(folder):
    result = {}
    rows = read_rows(folder / "metrics.csv")
    try:
        if rows:
            accuracies = [float(r["test_accuracy"]) for r in rows]
            result.update(completed_epoch=int(rows[-1]["epoch"]),
                          final_accuracy=accuracies[-1], best_accuracy=max(accuracies),
                          train_seconds=sum(float(r["train_seconds"]) for r in rows))
            peak = 0.
            collapse = None
            for row, acc in zip(rows, accuracies):
                peak = max(peak, acc)
                if collapse is None and peak >= .2 and acc <= .12 and float(row["train_loss"]) >= 2.25:
                    collapse = int(row["epoch"])
            result["collapse_epoch"] = collapse
        gradients = read_rows(folder / "gradients.csv")
        stem = [r for r in gradients if r.get("parameter") == "stem.conv.weight"]
        if stem:
            result["stem_norm_ratio"] = float(stem[-1]["norm_ratio"]) if stem[-1]["norm_ratio"] else None
            result["stem_cosine_to_bp"] = float(stem[-1]["cosine_to_bp"])
        cfg = folder / "config.json"
        if cfg.exists():
            result["initial_model_sha256"] = json.loads(cfg.read_text(encoding="utf-8"))["initial_model_sha256"]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        # A CSV may be briefly incomplete during the epoch's write.
        pass
    return result


def write_reports(root, status):
    for record in status["cases"]:
        record.update(inspect_case(Path(record["output"])))
    atomic_json(root / "suite_status.json", status)
    fields = ["case", "state", "completed_epoch", "final_accuracy", "best_accuracy",
              "collapse_epoch", "stem_norm_ratio", "stem_cosine_to_bp", "train_seconds",
              "initial_model_sha256", "output", "log", "exit_code"]
    temp = root / "suite_summary.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(status["cases"])
    os.replace(temp, root / "suite_summary.csv")
    atomic_json(root / "suite_findings.json", findings(status))


def findings(status):
    records = {r["case"]: r for r in status["cases"]}
    baseline = records.get("baseline", {})
    baseline_finished = baseline.get("state") in ("completed", "failed")
    reproduced = baseline.get("collapse_epoch") is not None if baseline_finished else None
    result = dict(provisional=True, synthetic=status["synthetic"], seed=status["seed"],
                  target_epoch=status["target_epoch"], baseline_reproduced=reproduced,
                  single_factor_evidence=[], contrasts={},
                  limits=["One seed; repeat effects with paired seeds before attributing a cause.",
                          "A short horizon does not establish final model accuracy.",
                          "If baseline collapse is not reproduced, this cannot explain the original failure."])
    if status["synthetic"]:
        result["baseline_reproduced"] = None
        result["limits"].insert(0, "FakeData plumbing check, not a scientific result.")
        return result
    for case in ("lr_only", "steps_only", "state_lr_only", "dual_lr_only", "backtracking_only"):
        record = records.get(case, {})
        if record.get("state") != "completed" or baseline.get("state") != "completed":
            continue
        result["single_factor_evidence"].append(dict(
            case=case, initial_weights_match=record.get("initial_model_sha256") == baseline.get("initial_model_sha256"),
            baseline_final_accuracy=baseline.get("final_accuracy"),
            variant_final_accuracy=record.get("final_accuracy"),
            variant_collapse_epoch=record.get("collapse_epoch"),
            learned_and_avoided_collapse=(record.get("collapse_epoch") is None and
                                        record.get("final_accuracy", 0.) >= .2)))
    pairs = {
        "lr_effect_original_inference": ("lr_only", "baseline"),
        "lr_effect_new_inference": ("lr_inference", "inference_bundle"),
        "backtracking_effect_original_inference": ("backtracking_only", "baseline"),
        "backtracking_effect_new_inference": ("combined", "lr_inference"),
    }
    for name, (variant, control) in pairs.items():
        a, b = records.get(variant, {}), records.get(control, {})
        if a.get("state") == "completed" and b.get("state") == "completed":
            result["contrasts"][name] = a["final_accuracy"] - b["final_accuracy"]
    contrasts = result["contrasts"]
    if "lr_effect_original_inference" in contrasts and "lr_effect_new_inference" in contrasts:
        contrasts["lr_inference_interaction"] = contrasts["lr_effect_new_inference"] - contrasts["lr_effect_original_inference"]
    return result


def checkpoint_epoch(path, configured):
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved = checkpoint["config"]
    for key in ("epochs", "c", "bs", "seed", "lr", "steps", "state_lr", "dual_lr",
                "rho", "backtracking", "train_samples", "validation_samples", "synthetic"):
        if saved.get(key) != getattr(configured, key):
            raise ValueError(f"Existing checkpoint setting differs: {key}; choose another output root")
    return checkpoint["epoch"]


def run_suite(args):
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("run_pcalm_ablation.py")
    records = []
    for case in args.cases:
        folder = root / f"{case}_seed{args.seed}"
        records.append(dict(case=case, state="pending", output=str(folder),
                            log=str(root / f"{case}_seed{args.seed}.log"),
                            completed_epoch=0, exit_code=None))
    status = dict(state="running", pid=os.getpid(), seed=args.seed,
                  training_epochs=args.training_epochs, target_epoch=args.run_epochs,
                  synthetic=args.smoke, cases=records, message="Starting controlled experiments")
    write_reports(root, status)
    expected_hash = None
    try:
        for record in records:
            case = record["case"]
            if (args.require_baseline_collapse and not args.smoke and
                    case not in ("baseline", "lr_only")):
                baseline = next((r for r in records if r["case"] == "baseline"), None)
                if baseline is None or baseline.get("collapse_epoch") is None:
                    status.update(state="needs_reproduction",
                                  message="Baseline did not reproduce collapse; remaining cases are pending")
                    break
            argv = ["--case", case, "--training-epochs", str(args.training_epochs),
                    "--run-epochs", str(args.run_epochs), "--seed", str(args.seed),
                    "--c", str(args.c), "--bs", str(args.bs),
                    "--train-samples", str(args.train_samples),
                    "--validation-samples", str(args.validation_samples),
                    "--device", args.device, "--output-root", str(root)]
            if args.smoke:
                argv += ["--smoke"]
            configured = build_training_args(case_args(argv))
            try:
                saved_epoch = checkpoint_epoch(Path(record["output"]) / "pcalm.pt", configured)
            except ValueError as error:
                record.update(state="configuration_mismatch", error=str(error))
                status.update(state="error", message=str(error))
                break
            remaining = args.run_epochs - saved_epoch
            if remaining <= 0:
                record.update(state="completed", exit_code=0)
                write_reports(root, status)
                case_hash = record.get("initial_model_sha256")
                if expected_hash is not None and case_hash != expected_hash:
                    raise RuntimeError("Initial model hashes differ between cases")
                expected_hash = case_hash
                continue
            argv[argv.index("--run-epochs") + 1] = str(remaining)
            if saved_epoch:
                argv += ["--resume"]
            command = [sys.executable, "-u", str(script), *argv]
            record.update(state="running", command=command)
            status["message"] = f"Running {case} to epoch {args.run_epochs}"
            print(status["message"], flush=True)
            write_reports(root, status)
            environment = os.environ.copy()
            environment.setdefault("OMP_NUM_THREADS", "2")
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            with Path(record["log"]).open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=script.parent, env=environment,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                record["pid"] = process.pid
                while process.poll() is None:
                    write_reports(root, status)
                    case_hash = record.get("initial_model_sha256")
                    if case_hash:
                        if expected_hash is not None and case_hash != expected_hash:
                            process.terminate()
                            process.wait()
                            raise RuntimeError("Initial model hashes differ between cases")
                        expected_hash = case_hash
                    time.sleep(10)
                record.update(exit_code=process.returncode,
                              state="completed" if process.returncode == 0 else "failed")
            write_reports(root, status)
            print(f"{case}: {record['state']}; epoch={record['completed_epoch']}; "
                  f"accuracy={record.get('final_accuracy')}", flush=True)
        else:
            status.update(state="completed", message="All selected cases finished; inspect failed cases if present")
        write_reports(root, status)
    except BaseException as error:
        status.update(state="error", message=str(error))
        write_reports(root, status)
        raise
    return status


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", choices=list(CHANGES), required=True)
    p.add_argument("--training-epochs", type=int, default=200)
    p.add_argument("--run-epochs", type=int, default=12)
    p.add_argument("--c", type=int, default=48)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-samples", type=int, default=0)
    p.add_argument("--validation-samples", type=int, default=5000)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-root", default="results/pcalm_ablation")
    p.add_argument("--require-baseline-collapse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if len(args.cases) != len(set(args.cases)):
        p.error("cases must be unique")
    if not 1 <= args.run_epochs <= args.training_epochs:
        p.error("require 1 <= run-epochs <= training-epochs")
    if args.require_baseline_collapse and not args.smoke and args.cases[0] != "baseline":
        p.error("baseline must be first when requiring reproduction")
    return args


if __name__ == "__main__":
    run_suite(parse_args())
