from __future__ import annotations

import unittest

from ta_diffref_format_config import RUN_VARIANTS, select_run_variants


class FinalFormatConfigurationTests(unittest.TestCase):
    def test_f5_is_final_retained_return_and_f3_is_anonymized_control(self):
        variants = {item.code: item for item in RUN_VARIANTS}
        self.assertEqual(tuple(variants), ("F0", "F1", "F2", "F3", "F4", "F5"))
        self.assertEqual(variants["F3"].alpha_mode, "full")
        self.assertEqual(variants["F3"].target_return, "removed")
        self.assertEqual(variants["F5"].alpha_mode, "learned_no_target_anonymization")
        self.assertEqual(variants["F5"].target_return, "retained")
        self.assertEqual(select_run_variants(), RUN_VARIANTS)


if __name__ == "__main__":
    unittest.main()
