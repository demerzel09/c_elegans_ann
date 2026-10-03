"""VS Code entry point for choosing new training or checkpoint resume."""
import argparse
from pathlib import Path


def training_arguments(method, action, output=None):
    from compare_bp_pcalm import parse_args

    output = output or ("results/pcalm_lr_only_full" if method == "pcalm" else "results/bp_full")
    argv = ["--methods", method]
    if action == "resume":
        argv += ["--resume", f"{output}/{method}.pt"]
    else:
        argv += ["--epochs", "200", "--c", "48", "--bs", "32",
                 "--lr", "0.01", "--output", output]
        if method == "pcalm":
            argv += ["--steps", "16", "--state-lr", "0.01",
                     "--dual-lr", "0.1", "--no-backtracking"]
    return parse_args(argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bp", "pcalm"), required=True)
    parser.add_argument("--action", choices=("new", "resume"), required=True)
    parser.add_argument("--output", help="Directory for new training and its resume checkpoint")
    selection = parser.parse_args()
    output = selection.output or ("results/pcalm_lr_only_full" if selection.method == "pcalm" else "results/bp_full")
    checkpoint = Path(output) / f"{selection.method}.pt"
    if selection.action == "resume" and not checkpoint.is_file():
        parser.error(f"再開するモデルがありません: {checkpoint}。起動時に「新規」を選んでください。")
    print(f"Starting {selection.method}: action={selection.action}; loading PyTorch and data...", flush=True)
    from compare_bp_pcalm import run

    run(training_arguments(selection.method, selection.action, output))
