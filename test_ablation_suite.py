"""Reports must distinguish observed collapse from incomplete/synthetic runs."""
import unittest
from run_ablation_suite import findings


class SuiteReportTests(unittest.TestCase):
    def status(self):
        return dict(synthetic=False, seed=42, target_epoch=12, cases=[
            dict(case="baseline", state="completed", completed_epoch=12, collapse_epoch=8,
                 final_accuracy=.1, initial_model_sha256="same"),
            dict(case="lr_only", state="completed", completed_epoch=12, collapse_epoch=None,
                 final_accuracy=.4, initial_model_sha256="same"),
        ])

    def test_single_factor_evidence_requires_finished_cases(self):
        status = self.status()
        report = findings(status)
        self.assertTrue(report["baseline_reproduced"])
        self.assertTrue(report["single_factor_evidence"][0]["initial_weights_match"])
        self.assertTrue(report["single_factor_evidence"][0]["learned_and_avoided_collapse"])
        self.assertAlmostEqual(report["contrasts"]["lr_effect_original_inference"], .3)
        status["cases"][1]["state"] = "running"
        self.assertEqual(findings(status)["single_factor_evidence"], [])
        self.assertEqual(findings(status)["contrasts"], {})

    def test_smoke_results_are_never_causal_evidence(self):
        status = self.status()
        status["synthetic"] = True
        report = findings(status)
        self.assertIsNone(report["baseline_reproduced"])
        self.assertEqual(report["single_factor_evidence"], [])
        self.assertEqual(report["contrasts"], {})


if __name__ == "__main__":
    unittest.main()
