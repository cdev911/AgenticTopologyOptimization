import unittest

from fenitop.config import normalize_boundary_conditions, validate_config


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_invalid_iteration_count(self):
        config = {"opt": {"max_iter": 0}, "fem": {}, "parameter_guidance": {}}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_volume_fraction_outside_bounds(self):
        config = {"opt": {"vol_frac": 1.2}, "fem": {}, "parameter_guidance": {}}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_normalizes_multiple_boundary_conditions(self):
        fem_config = {
            "dirichlet_bcs": [
                {"marker": lambda x: True, "value": [1.0, 0.0]},
                {"marker": lambda x: False, "value": [0.0, 1.0]},
            ],
            "traction_bcs": [
                {"value": [0.0, -1.0], "locator": lambda x: True},
            ],
        }
        dirichlet_bcs, traction_bcs = normalize_boundary_conditions(fem_config, dim=2)
        self.assertEqual(len(dirichlet_bcs), 2)
        self.assertEqual(traction_bcs[0][0], [0.0, -1.0])


if __name__ == "__main__":
    unittest.main()
