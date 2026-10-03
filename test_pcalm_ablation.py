"""Verify each single-factor experiment actually changes only one factor."""
import unittest
from run_pcalm_ablation import BASELINE, CHANGES, build_training_args, parse_args


class AblationTests(unittest.TestCase):
    def test_cases_have_identical_controls_and_single_changes(self):
        baseline = build_training_args(parse_args(["--case", "baseline"]))
        controls = ("epochs", "run_epochs", "c", "bs", "seed", "train_samples",
                    "validation_samples", "optimizer", "rho",
                    "label_smoothing", "log_gradients", "collapse_patience")
        outputs = set()
        for case in CHANGES:
            with self.subTest(case=case):
                args = build_training_args(parse_args(["--case", case]))
                for control in controls:
                    self.assertEqual(getattr(args, control), getattr(baseline, control))
                changed = {key for key in BASELINE if getattr(args, key) != getattr(baseline, key)}
                if case == "baseline":
                    self.assertEqual(changed, set())
                elif case == "combined":
                    self.assertEqual(changed, set(BASELINE) - {"weight_decay"})
                elif case == "inference_bundle":
                    self.assertEqual(changed, {"steps", "state_lr", "dual_lr"})
                elif case == "lr_inference":
                    self.assertEqual(changed, {"lr", "steps", "state_lr", "dual_lr"})
                else:
                    self.assertEqual(len(changed), 1)
                self.assertNotIn(args.output, outputs)
                outputs.add(args.output)
                self.assertEqual(args.epochs, 200)
                self.assertEqual(args.run_epochs, 12)


if __name__ == "__main__":
    unittest.main()
