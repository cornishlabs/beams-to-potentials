"""Potential calculation and the high-level PotentialSystem API."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fmin_tnc

from .beams import Beam, gaussian_electric_field, group_beams_by_wavelength
from .constants import A0, C, EPSILON_0, H
from .species import Species, get_species

DEFAULT_MINIMUM_BOUNDS = ((-5, 5), (-5, 5), (-9, 9))


def potential_mhz(
    coord: Sequence[object], beams: Iterable[Beam], species: str | Species
):
    """Calculate optical potential in MHz for one species at ``coord``."""

    species_def = get_species(species)
    potential = 0

    for wavelength_nm, wavelength_beams in group_beams_by_wavelength(beams).items():
        electric_field = 0 + 0j
        for beam in wavelength_beams:
            electric_field += gaussian_electric_field(coord, beam)

        intensity = (1 / 2) * C * EPSILON_0 * np.abs(electric_field) ** 2
        polarizability_si = (
            species_def.polarizability_au(wavelength_nm) * 4 * np.pi * EPSILON_0 * A0**3
        )
        trap_depth_j = (polarizability_si / (2 * C * EPSILON_0)) * intensity
        potential += -(trap_depth_j / H / 1e6)

    return potential


def intensity_w_m2(coord: Sequence[object], beams: Iterable[Beam]):
    """Calculate total optical intensity in W/m^2 at ``coord``.

    Beams with the same wavelength are added coherently before converting to
    intensity; intensities from different wavelengths are summed.
    """

    intensity = 0
    for wavelength_beams in group_beams_by_wavelength(beams).values():
        electric_field = 0 + 0j
        for beam in wavelength_beams:
            electric_field += gaussian_electric_field(coord, beam)
        intensity += (1 / 2) * C * EPSILON_0 * np.abs(electric_field) ** 2
    return intensity


def intensity_kw_cm2(coord: Sequence[object], beams: Iterable[Beam]):
    """Calculate total optical intensity in kW/cm^2 at ``coord``."""

    return intensity_w_m2(coord, beams) / 1e7


@dataclass(frozen=True)
class PotentialSystem:
    """A set of beams evaluated for one trapped species.

    This is the main object used by examples. It accepts scalar coordinates or
    NumPy arrays, so the same ``potential((x, y, z))`` call works for a single
    point, a line cut, or a meshgrid.
    """

    beams: Sequence[Beam]
    species: str | Species

    def __post_init__(self) -> None:
        object.__setattr__(self, "beams", tuple(self.beams))
        object.__setattr__(self, "species", get_species(self.species))

    def potential(self, coord: Sequence[object]):
        """Optical potential in MHz at ``coord``."""

        return potential_mhz(coord, self.beams, self.species)

    def intensity(self, coord: Sequence[object]):
        """Total optical intensity in W/m^2 at ``coord``."""

        return intensity_w_m2(coord, self.beams)

    def intensity_kw_cm2(self, coord: Sequence[object]):
        """Total optical intensity in kW/cm^2 at ``coord``."""

        return intensity_kw_cm2(coord, self.beams)

    def find_minimum(
        self,
        initial_guess_um: Sequence[float],
        bounds: Sequence[tuple[float, float]] = DEFAULT_MINIMUM_BOUNDS,
        epsilon: float = 0.002,
    ) -> np.ndarray:
        """Find a local potential minimum near ``initial_guess_um``."""

        minimum, _, _ = fmin_tnc(
            lambda coord: self.potential(coord),
            initial_guess_um,
            approx_grad=True,
            bounds=bounds,
            epsilon=epsilon,
            disp=False,
        )
        return minimum

    def with_species(self, species: str | Species) -> "PotentialSystem":
        return PotentialSystem(self.beams, species)

    def with_beams(self, beams: Sequence[Beam]) -> "PotentialSystem":
        return PotentialSystem(beams, self.species)

    def trap_frequencies(self, centre_um: Sequence[float], **kwargs):
        from .analysis import trap_frequencies

        return trap_frequencies(self, centre_um, **kwargs)

    def analyze_trap(self, initial_guess_um: Sequence[float], **kwargs):
        from .analysis import analyze_trap

        return analyze_trap(self, initial_guess_um, **kwargs)


__all__ = [
    "DEFAULT_MINIMUM_BOUNDS",
    "PotentialSystem",
    "intensity_kw_cm2",
    "intensity_w_m2",
    "potential_mhz",
]
