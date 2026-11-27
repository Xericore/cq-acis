"""A trim on a stored-fit surface may sit up to three saved fit allowances from the fitted support.

Witness: face 51 on spline-surface $76 (fit 0.001 mm) with fitted intcurves; the projected trim deviates 0.00556 mm
against the former allowance 1e-6 + 2 x 0.002 = 0.004001 mm and is within 1e-6 + 3 x 0.002 = 0.006001 mm.
"""
import unittest
from typing import cast
from unittest.mock import patch

from cq_acis import AcisMetadata, AcisModel, cadquery
from cq_acis.model import AcisModelView


class StoredFitMarginTests(unittest.TestCase):
    def test_the_margin_is_three_saved_fit_allowances(self):
        self.assertEqual(cadquery._STORED_FIT_MARGIN, 3.0)

    def test_the_208bf9c_trim_is_inside_the_allowance_and_was_outside_the_former_one(self):
        tolerance, allowance, deviation = 1e-6, 0.002, 0.005556774800690702
        self.assertLessEqual(deviation, tolerance + allowance * cadquery._STORED_FIT_MARGIN)
        self.assertGreater(deviation, tolerance + allowance * 2.0)

    def test_a_larger_fit_allowance_is_local_to_one_converter(self):
        model = cast(AcisModelView, AcisModel(AcisMetadata(units_mm=1.0, resabs=1e-6), ()))
        with patch.object(cadquery, "_load_backend", return_value=(None, "occt")):
            preview = cadquery.CadQueryConverter(model, stored_fit_margin=64.0)
            normal = cadquery.CadQueryConverter(model)
        self.assertAlmostEqual(preview._stored_fit_limit(0.001), 0.064001)
        self.assertAlmostEqual(normal._stored_fit_limit(0.001), 0.003001)
        self.assertEqual(preview._stored_fit_limit(0.0), normal._stored_fit_limit(0.0))

    def test_an_invalid_margin_is_refused_before_loading_the_backend(self):
        model = cast(AcisModelView, AcisModel(AcisMetadata(), ()))
        for margin in (float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 65.0, True, "64"):
            with self.subTest(margin=margin), self.assertRaisesRegex(ValueError, "stored_fit_margin"):
                cadquery.CadQueryConverter(model, stored_fit_margin=cast(float, margin))


if __name__ == "__main__":
    unittest.main()
