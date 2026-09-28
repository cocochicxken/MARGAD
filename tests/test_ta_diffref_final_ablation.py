from __future__ import annotations

import unittest

import torch

from model import AdaptiveWaveletAffinity
from large_graph import _sampled_raw_return_mass
from ta_diffref_ablation_config import DATASETS, DEFAULT_SEEDS, VARIANTS
from aggregate_ta_diffref_ablation import _paired_rows, _summary_rows
from utils import bidirect_unweighted


class FinalTaDiffRefOperatorTests(unittest.TestCase):
    def setUp(self):
        self.adjacency = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        self.features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 1.0],
                [2.0, -1.0],
            ],
            dtype=torch.float64,
        )

    def expected(self):
        b = self.adjacency
        degree = b.sum(dim=1)
        p = b / degree.unsqueeze(1)
        s = b / (degree.sqrt().unsqueeze(1) * degree.sqrt().unsqueeze(0))
        b2 = b @ b
        p2 = p @ p
        one = p @ self.features
        two = p2 @ self.features
        return_probability = p2.diagonal()
        anonymous = (
            two - return_probability.unsqueeze(1) * self.features
        ) / (1.0 - return_probability).unsqueeze(1)
        return {
            "one_hop": one,
            "symmetric_one_hop": s @ self.features,
            "raw_equal_multiscale": 0.5 * (b @ self.features + b2 @ self.features),
            "fixed_equal_no_target_anonymization": 0.5 * (one + two),
            "raw_equal_anonymous_no_renormalization": 0.5 * (
                b @ self.features
                + b2 @ self.features
                - b2.diagonal().unsqueeze(1) * self.features
            ),
            "fixed_equal_multiscale": 0.5 * (one + anonymous),
        }

    def test_a0_to_a5_match_declared_dense_operators(self):
        for mode, expected in self.expected().items():
            with self.subTest(mode=mode):
                actual = AdaptiveWaveletAffinity(mode=mode)(
                    self.features, self.adjacency
                )
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_a0_to_a5_match_declared_sparse_operators(self):
        sparse = self.adjacency.to_sparse_coo().coalesce()
        for mode, expected in self.expected().items():
            with self.subTest(mode=mode):
                actual = AdaptiveWaveletAffinity(mode=mode)(self.features, sparse)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_fixed_variants_keep_three_identical_heads(self):
        for variant in VARIANTS[:-1]:
            with self.subTest(variant=variant.code):
                model = AdaptiveWaveletAffinity(mode=variant.alpha_mode)
                _, references = model(
                    self.features,
                    self.adjacency,
                    return_filters=True,
                )
                self.assertEqual(tuple(references.shape), (3, 4, 2))
                torch.testing.assert_close(references[0], references[1])
                torch.testing.assert_close(references[1], references[2])

    def test_final_protocol_has_six_datasets_and_seven_variants(self):
        self.assertEqual([variant.code for variant in VARIANTS], [f"A{i}" for i in range(7)])
        self.assertEqual(
            [dataset.key for dataset in DATASETS],
            ["facebook", "reddit", "yelpchi", "tfinance", "elliptic", "tsocial"],
        )
        self.assertEqual(sum(dataset.gamma_centering for dataset in DATASETS), 5)
        self.assertEqual(DEFAULT_SEEDS, (0, 1, 2, 3, 4))

    def test_sampled_a4_counts_only_observed_return_paths(self):
        import dgl

        first = dgl.create_block(
            (
                torch.tensor([0, 3, 0, 2]),
                torch.tensor([1, 1, 2, 2]),
            ),
            num_src_nodes=4,
            num_dst_nodes=3,
        )
        first.srcdata[dgl.NID] = torch.tensor([0, 1, 2, 3])
        first.dstdata[dgl.NID] = torch.tensor([0, 1, 2])
        second = dgl.create_block(
            (torch.tensor([1, 2]), torch.tensor([0, 0])),
            num_src_nodes=3,
            num_dst_nodes=1,
        )
        second.srcdata[dgl.NID] = torch.tensor([0, 1, 2])
        second.dstdata[dgl.NID] = torch.tensor([0])

        return_mass = _sampled_raw_return_mass(
            [first, second], num_nodes=4, device=torch.device("cpu"), dtype=torch.float32
        )
        torch.testing.assert_close(return_mass, torch.tensor([2.0]))

    def test_a1_negative_edge_set_is_bidirected_before_s_normalization(self):
        directed = torch.sparse_coo_tensor(
            torch.tensor([[0, 2], [1, 1]]),
            torch.ones(2),
            (3, 3),
        ).coalesce()
        bidirected = bidirect_unweighted(directed).to_dense()
        expected = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        torch.testing.assert_close(bidirected, expected)
        operator, _ = AdaptiveWaveletAffinity(
            mode="symmetric_one_hop"
        ).symmetric_degree_terms(bidirected.to_sparse_coo(), torch.float32)
        self.assertTrue(torch.isfinite(operator.values()).all())

    def test_paired_aggregation_uses_common_seeds_and_population_std(self):
        spec = DATASETS[0]
        variants = (VARIANTS[2], VARIANTS[3])
        rows = []
        for seed, a2_auc, a3_auc, a2_pr, a3_pr in (
            (0, 70.0, 71.0, 20.0, 19.0),
            (1, 72.0, 74.0, 21.0, 22.0),
        ):
            for variant, auc, auprc in (
                (variants[0], a2_auc, a2_pr),
                (variants[1], a3_auc, a3_pr),
            ):
                rows.append({
                    "dataset": spec.cli_name,
                    "dataset_key": spec.key,
                    "variant": variant.code,
                    "alpha_mode": variant.alpha_mode,
                    "seed": seed,
                    "status": "completed",
                    "auc_percent": auc,
                    "auprc_percent": auprc,
                    "best_selection_loss": 1.0,
                    "train_seconds": 2.0,
                    "total_seconds": 3.0,
                })

        summaries = _summary_rows(rows, (spec,), variants)
        self.assertEqual([row["valid_n"] for row in summaries], [2, 2])
        paired, per_seed = _paired_rows(rows, (spec,), variants)
        auc = next(row for row in paired if row["metric"] == "AUROC")
        auprc = next(row for row in paired if row["metric"] == "AUPRC")
        self.assertEqual(auc["paired_n"], 2)
        self.assertAlmostEqual(auc["mean_difference_percent_points"], 1.5)
        self.assertAlmostEqual(auc["std_difference_percent_points"], 0.5)
        self.assertEqual(auc["positive_seed_count"], 2)
        self.assertAlmostEqual(auprc["mean_difference_percent_points"], 0.0)
        self.assertEqual(auprc["positive_seed_count"], 1)
        self.assertEqual(auprc["negative_seed_count"], 1)
        self.assertEqual(len(per_seed), 4)


if __name__ == "__main__":
    unittest.main()
