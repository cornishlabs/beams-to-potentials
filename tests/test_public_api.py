import subprocess
import sys
import unittest

import numpy as np

import beams_to_potentials as btp


class PublicApiTests(unittest.TestCase):
    def test_import_has_no_stdout_or_stderr_side_effects(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import beams_to_potentials"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_single_beam_potential_signs(self):
        beam_1065 = btp.Beam(
            wavelength_nm=1065,
            waist_x_um=1,
            waist_y_um=1,
            waist_z_um=1,
            power_mw=1,
        )
        beam_817 = beam_1065.with_updates(wavelength_nm=817)

        self.assertLess(btp.PotentialSystem([beam_1065], "Cs").potential((0, 0, 0)), 0)
        self.assertGreater(btp.PotentialSystem([beam_817], "Cs").potential((0, 0, 0)), 0)

    def test_vectorized_coordinates(self):
        beam = btp.Beam(
            wavelength_nm=1065,
            waist_x_um=1,
            waist_y_um=1,
            waist_z_um=1,
            power_mw=1,
        )
        system = btp.PotentialSystem([beam], "Rb")
        xs = np.linspace(-1, 1, 5)
        potential = system.potential((xs, 0, 0))

        self.assertEqual(potential.shape, xs.shape)
        self.assertTrue(np.all(np.isfinite(potential)))

    def test_intensity_is_species_independent_and_positive(self):
        beam = btp.Beam(
            wavelength_nm=1065,
            waist_x_um=1,
            waist_y_um=1,
            waist_z_um=1,
            power_mw=1,
        )
        rb_system = btp.PotentialSystem([beam], "Rb")
        cs_system = btp.PotentialSystem([beam], "Cs")

        self.assertGreater(rb_system.intensity((0, 0, 0)), 0)
        self.assertAlmostEqual(
            rb_system.intensity_kw_cm2((0, 0, 0)),
            cs_system.intensity_kw_cm2((0, 0, 0)),
        )

    def test_find_minimum_near_single_beam_centre(self):
        centre = (0.2, -0.1, 0.3)
        beam = btp.Beam(
            wavelength_nm=1065,
            waist_x_um=1,
            waist_y_um=1,
            waist_z_um=1,
            power_mw=1,
            center_um=centre,
        )
        system = btp.PotentialSystem([beam], "Cs")
        minimum = system.find_minimum((0, 0, 0))

        np.testing.assert_allclose(minimum, centre, atol=0.05)

    def test_trap_frequency_fit_on_synthetic_quadratic(self):
        xs = np.linspace(-1, 1, 101)
        curvature = 0.4
        potential = btp.quadratic_potential(xs, 0.15, curvature, -5)
        fit = btp.fit_trap_frequency(
            xs,
            potential,
            mass_amu=133,
            x_fit_threshold_um=0.8,
            x_bounds=(-0.6, 0.8),
        )
        expected = btp.trap_frequency_from_curvature_mhz_um2(curvature, 133)

        self.assertAlmostEqual(fit.center_um, 0.15, places=6)
        self.assertAlmostEqual(fit.frequency_mhz, expected, places=9)


if __name__ == "__main__":
    unittest.main()
