from __future__ import annotations

import unittest

from args import parameter_parser

try:
    import torch

    from training_common import gamma_components_need_centers, gamma_discrepancy
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


class GammaCenteringArgumentTests(unittest.TestCase):
    def test_fixed_toggle_defaults_to_centered(self):
        centered = parameter_parser(["--dataset", "Amazon"])
        self.assertTrue(centered.gamma_centering)

        tsocial_default = parameter_parser(["--dataset", "tsocial"])
        self.assertFalse(tsocial_default.gamma_centering)

        uncentered = parameter_parser([
            "--dataset", "tsocial", "--gamma_centering", "0",
        ])
        self.assertFalse(uncentered.gamma_centering)

    def test_final_affinity_retains_target_return(self):
        options = parameter_parser(["--dataset", "Facebook"])
        self.assertEqual(options.alpha_mode, "learned_no_target_anonymization")


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this Python environment")
class GammaCenteringTensorTests(unittest.TestCase):
    def test_switch_selects_centered_or_origin_anchored_cross_response(self):
        thick = torch.zeros(3, 2, 2)
        thin = torch.tensor([
            [[2.0, 0.0], [2.0, 0.0]],
            [[0.0, 3.0], [0.0, 3.0]],
            [[4.0, 4.0], [4.0, 4.0]],
        ])
        centered, _ = gamma_discrepancy(thick, thin, normalize_bands=False, center=True)
        uncentered, _ = gamma_discrepancy(thick, thin, normalize_bands=False, center=False)

        torch.testing.assert_close(centered, torch.zeros(2))
        torch.testing.assert_close(
            uncentered,
            torch.tensor([2.0 + 3.0 + 4.0 * 2.0**0.5] * 2) / 3.0,
        )
        self.assertEqual(gamma_components_need_centers("full", False), ())


if __name__ == "__main__":
    unittest.main()
