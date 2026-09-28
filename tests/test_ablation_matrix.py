import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ablation_config import parse_seeds, selected_specs, selected_variants
from ablation_diagnostics import GammaConcentrationAccumulator, _spearman
from ablation_aggregate import aggregate
from ablation_runner import _filter_variants


class AblationMatrixTests(unittest.TestCase):
    def test_publication_matrix_has_19_nonduplicated_configs(self):
        variants = selected_variants(("all",))
        self.assertEqual(19, len(variants))
        self.assertEqual(19, len({variant.code for variant in variants}))
        self.assertEqual(665, len(variants) * len(selected_specs(("all",))) * len(parse_seeds(("0-4",))))

    def test_seed_range_and_tie_aware_spearman(self):
        self.assertEqual((0, 1, 2, 3, 4), parse_seeds(("0-4",)))
        self.assertAlmostEqual(1.0, _spearman(np.array([1, 1, 2, 3]), np.array([4, 4, 5, 8])))

    def test_streaming_channel_concentration_keeps_node_evidence(self):
        accumulator = GammaConcentrationAccumulator(10)
        # Different anomalies peak at different channels.  The aggregate can
        # be diffuse, but each node-level top-10% share must still be one.
        energy = np.zeros((2, 10), dtype=np.float64)
        energy[0, 0] = 3.0
        energy[1, 9] = 5.0
        accumulator.update(energy, np.array([1, 1]))
        result = accumulator.as_dict()
        self.assertAlmostEqual(
            1.0,
            result["anomaly"]["node_level_concentration"]["top10_percent_energy_share"]["mean"],
        )
        self.assertEqual(2, result["anomaly_count"])

    def test_exact_variant_filter_for_smoke_test(self):
        variants = _filter_variants(selected_variants(("all",)), ("L6",))
        self.assertEqual(("L6",), tuple(variant.code for variant in variants))

    def test_aggregate_exports_score_diagnostics_and_degree_buckets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs" / "loss" / "L6" / "facebook" / "seed_00"
            run_dir.mkdir(parents=True)
            diagnostics = {
                "alpha": {
                    "degree_diagnostics": {
                        "score_degree_pearson": 0.1,
                        "score_log_degree_pearson": 0.2,
                        "score_degree_spearman": 0.3,
                        "score_log_degree_spearman": 0.4,
                        "buckets": [{
                            "bucket": 1, "count": 10, "anomaly_count": 2,
                            "degree_min": 1.0, "degree_max": 3.0,
                            "auc": 0.75, "auprc": 0.5,
                        }],
                    }
                },
                "gamma": {"centered_channel_concentration": {"available": False}},
            }
            result = {
                "status": "completed", "study": "loss", "variant": "L6",
                "dataset": "Facebook", "dataset_key": "facebook", "seed": 0,
                "active_losses": ["alpha", "beta", "gamma"],
                "weights": {"alpha": 1.0, "beta": 0.15, "gamma": 0.6},
                "alpha_mode": "full", "gamma_mode": "full", "epoch_override": None,
                "metrics": {
                    "auc": 0.9, "auprc": 0.7,
                    "best_selection_epoch": 50, "best_selection_loss": 0.1,
                },
                "diagnostics_file": "diagnostics_attempt_01.json",
                "score_only_combinations": {
                    "alpha+beta+gamma": {
                        "auc": 0.9, "auprc": 0.7,
                        "weights": {"alpha": 1.0, "beta": 0.15, "gamma": 0.6},
                    }
                },
            }
            (run_dir / "diagnostics_attempt_01.json").write_text(
                json.dumps(diagnostics), encoding="utf-8"
            )
            (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (run_dir / "completed.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            manifest = aggregate(root)
            self.assertEqual(1, manifest["completed_seed_runs"])
            self.assertEqual(1, manifest["score_only_seed_rows"])
            self.assertEqual(1, manifest["degree_bucket_seed_rows"])
            self.assertTrue((root / "degree_bucket_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
