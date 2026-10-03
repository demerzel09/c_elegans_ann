"""Resume must reproduce uninterrupted epoch updates, including PC-ALM."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import compare_bp_pcalm as training


def args_for(method, output, *extra):
    argv = ["compare_bp_pcalm.py", "--methods", method, "--synthetic", "--device", "cpu",
            "--epochs", "3", "--c", "2", "--bs", "2", "--train-samples", "4",
            "--test-samples", "4", "--steps", "3", "--output", str(output), *extra]
    with patch("sys.argv", argv):
        return training.parse_args()


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def quiet_run(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return training.run(args)

    def test_interrupted_resume_matches_uninterrupted_bp_and_pcalm(self):
        for method, optimizer in [("bp", "sgd"), ("pcalm", "adamw")]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.quiet_run(args_for(method, root / "full", "--optimizer", optimizer))
                save = training.save_checkpoint

                def interrupt_after_first_epoch(state, path):
                    save(state, path)
                    if state["epoch"] == 1 and path.name == f"{method}.pt":
                        raise RuntimeError("simulated interruption")

                with patch.object(training, "save_checkpoint", side_effect=interrupt_after_first_epoch):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        self.quiet_run(args_for(method, root / "split", "--optimizer", optimizer))
                checkpoint_path = root / "split" / f"{method}.pt"
                # Different CLI defaults must be replaced by saved settings.
                with patch("sys.argv", ["compare_bp_pcalm.py", "--methods", method,
                                       "--resume", str(checkpoint_path), "--device", "cpu"]):
                    self.quiet_run(training.parse_args())
                full = torch.load(root / "full" / f"{method}.pt", weights_only=True)
                resumed = torch.load(checkpoint_path, weights_only=True)
                self.assertEqual(resumed["epoch"], 3)
                self.assertEqual(resumed["config"]["optimizer"], optimizer)
                self.assertEqual(full["scheduler"], resumed["scheduler"])
                for key in full["model"]:
                    torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
                for a, b in zip(full["history"], resumed["history"]):
                    for key in ("epoch", "lr", "train_loss", "test_accuracy", "residual_rms"):
                        self.assertEqual(a[key], b[key])
                for key in ("torch", "train_generator", "test_generator"):
                    self.assertTrue(torch.equal(full["rng"][key], resumed["rng"][key]))

    def test_old_weight_only_model_can_start_additional_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.quiet_run(args_for("bp", root / "old"))
            full = torch.load(root / "old" / "bp.pt", weights_only=True)
            legacy = {k: full[k] for k in ("model", "config", "method")}
            legacy["config"]["torch_version"] = torch.__version__
            source = root / "legacy.pt"
            torch.save(legacy, source)
            self.quiet_run(args_for("bp", root / "extended", "--resume", str(source),
                                   "--extra-epochs", "2", "--restart-lr", "0.01"))
            result = torch.load(root / "extended" / "bp.pt", weights_only=True)
            self.assertEqual(result["epoch"], 5)
            self.assertEqual(result["scheduler_plan"], {"kind": "cosine", "epochs": 2})
            self.assertEqual(result["history"][0]["epoch"], 4)
            self.assertEqual(result["history"][0]["lr"], .01)
            self.assertTrue(any(not torch.equal(v, result["model"][k])
                                for k, v in full["model"].items()))

    def test_epoch_cap_and_gradient_audit_preserve_training_and_resume(self):
        for method in ("bp", "pcalm"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.quiet_run(args_for(method, root / "reference"))
                self.quiet_run(args_for(method, root / "audited", "--run-epochs", "1", "--log-gradients"))
                source = root / "audited" / f"{method}.pt"
                partial = torch.load(source, weights_only=True)
                self.assertEqual(partial["epoch"], 1)
                self.assertEqual(partial["config"]["epochs"], 3)
                with patch("sys.argv", ["compare_bp_pcalm.py", "--methods", method,
                                       "--resume", str(source), "--device", "cpu"]):
                    self.quiet_run(training.parse_args())
                reference = torch.load(root / "reference" / f"{method}.pt", weights_only=True)
                result = torch.load(source, weights_only=True)
                self.assertEqual(result["epoch"], 3)
                self.assertEqual(reference["config"]["initial_model_sha256"],
                                 result["config"]["initial_model_sha256"])
                self.assertEqual(reference["scheduler"], result["scheduler"])
                for key in reference["model"]:
                    torch.testing.assert_close(reference["model"][key], result["model"][key], rtol=0, atol=0)
                self.assertEqual(len(result["gradient_history"]), 27)

    def test_extra_epochs_from_full_checkpoint_retain_optimizer_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.quiet_run(args_for("pcalm", root / "first", "--optimizer", "adamw"))
            source = root / "first" / "pcalm.pt"
            prior = torch.load(source, weights_only=True)
            self.quiet_run(args_for("pcalm", root / "extra", "--resume", str(source),
                                   "--extra-epochs", "1", "--restart-lr", "0.001",
                                   "--steps", "5", "--state-lr", "0.02", "--dual-lr", "0.5"))
            result = torch.load(root / "extra" / "pcalm.pt", weights_only=True)
            self.assertEqual(result["epoch"], 4)
            self.assertEqual(result["config"]["steps"], 5)
            self.assertEqual(result["config"]["state_lr"], .02)
            self.assertEqual(result["config"]["dual_lr"], .5)
            key = next(iter(prior["optimizer"]["state"]))
            self.assertEqual(result["optimizer"]["state"][key]["step"].item(),
                             prior["optimizer"]["state"][key]["step"].item() + 2)

    def test_resume_rejects_wrong_method_and_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.quiet_run(args_for("bp", root / "source"))
            source = root / "source" / "bp.pt"
            with self.assertRaisesRegex(ValueError, "method differs"):
                self.quiet_run(args_for("pcalm", root / "wrong", "--resume", str(source)))
            with self.assertRaisesRegex(ValueError, "already completed"):
                self.quiet_run(args_for("bp", root / "completed", "--resume", str(source)))
            self.assertFalse((root / "wrong").exists())
            self.assertFalse((root / "completed").exists())

    def test_collapse_stops_after_saving_latest_and_best_models(self):
        from torchvision.datasets import FakeData
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = args_for("pcalm", root, "--epochs", "4", "--collapse-patience", "2")
            args.synthetic = False

            def data(*a, **kw):
                return FakeData(4, (3, 32, 32), 10, transform=kw["transform"])

            def gradients(model, *a, diagnostics, **kw):
                for p in model.parameters():
                    p.grad = torch.zeros_like(p)
                diagnostics.update(state_lr_min=.03, backtracks=0)
                return 2.4, 0.

            with patch.object(training, "CIFAR10", side_effect=data), \
                 patch.object(training, "evaluate", side_effect=[.3, .1, .1]), \
                 patch.object(training, "pcalm_gradients", side_effect=gradients):
                with self.assertRaisesRegex(RuntimeError, "collapsed near chance"):
                    self.quiet_run(args)
            latest = torch.load(root / "pcalm.pt", weights_only=True)
            best = torch.load(root / "pcalm_best.pt", weights_only=True)
            self.assertEqual(latest["epoch"], 3)
            self.assertEqual(best["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
