"""Beam models and Gaussian electric-field helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .constants import C, EPSILON_0


@dataclass(frozen=True)
class Beam:
    """A single Gaussian beam.

    Units are deliberately explicit in the field names: wavelengths and
    positions/waists are in um or nm as named, power is in mW, theta and phase
    are in radians.
    """

    wavelength_nm: int
    waist_x_um: float
    waist_y_um: float
    waist_z_um: float
    power_mw: float
    theta_rad: float = 0.0
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase_rad: float = 0.0
    label: str = ""

    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm / 1e3

    @property
    def x0_um(self) -> float:
        return self.center_um[0]

    @property
    def y0_um(self) -> float:
        return self.center_um[1]

    @property
    def z0_um(self) -> float:
        return self.center_um[2]

    def with_updates(
        self, **changes: float | str | tuple[float, float, float]
    ) -> "Beam":
        """Return a copy of the beam with selected dataclass fields changed."""

        return replace(self, **changes)


def waist_um(z_um, waist_0_um: float, wavelength_um: float):
    """Gaussian beam waist at propagation distance ``z_um``."""

    z_rayleigh_um = rayleigh_range_um(waist_0_um, wavelength_um)
    return waist_0_um * np.sqrt(1 + (z_um / z_rayleigh_um) ** 2)


def rayleigh_range_um(waist_0_um: float, wavelength_um: float) -> float:
    """Rayleigh range in um for a waist and wavelength given in um."""

    return float(np.pi * waist_0_um**2 / wavelength_um)


def rayleigh_ranges_um(beam: Beam) -> tuple[float, float, float]:
    """Rayleigh ranges for ``beam`` from its x, y, and z waists."""

    return (
        rayleigh_range_um(beam.waist_x_um, beam.wavelength_um),
        rayleigh_range_um(beam.waist_y_um, beam.wavelength_um),
        rayleigh_range_um(beam.waist_z_um, beam.wavelength_um),
    )


def center_intensity_w_m2(beam: Beam) -> float:
    """Peak intensity of ``beam`` in W/m^2."""

    center_intensity_mw_um2 = (
        2 * beam.power_mw / (np.pi * beam.waist_x_um * beam.waist_y_um)
    )
    return (center_intensity_mw_um2 / 1e3) * (1e6) ** 2


def center_electric_field_v_m(beam: Beam) -> float:
    """Peak electric-field amplitude of ``beam`` in V/m."""

    return np.sqrt(2 * center_intensity_w_m2(beam) / (C * EPSILON_0))


def gaussian_electric_field(coord: Sequence[object], beam: Beam):
    """Complex electric field of a rotated Gaussian beam.

    This preserves the formula used in the original scripts, including the
    phase convention and omission of the commented wavefront-curvature terms.
    """

    x, y, z = coord
    wavelength_um = beam.wavelength_um
    electric_field_0 = center_electric_field_v_m(beam)

    # ``s`` is distance along the beam axis; ``xp`` is transverse in the x-z plane.
    s = (x - beam.x0_um) * np.sin(beam.theta_rad) + (z - beam.z0_um) * np.cos(
        beam.theta_rad
    )
    xp = (x - beam.x0_um) * np.cos(beam.theta_rad) - (z - beam.z0_um) * np.sin(
        beam.theta_rad
    )

    wx = waist_um(s, beam.waist_x_um, wavelength_um)
    wy = waist_um(s, beam.waist_y_um, wavelength_um)
    wz = waist_um(s, beam.waist_z_um, wavelength_um)

    k = 2 * np.pi / wavelength_um
    return (
        electric_field_0
        * np.exp(-(xp**2) / wx**2)
        * np.exp(-((y - beam.y0_um) ** 2) / wy**2)
        * (beam.waist_z_um / wz)
        * np.exp(-1j * k * s)
        * np.exp(-1j * beam.phase_rad)
    )


def group_beams_by_wavelength(beams: Iterable[Beam]) -> dict[int, tuple[Beam, ...]]:
    """Group beams so same-wavelength fields can interfere coherently."""

    groups: dict[int, list[Beam]] = defaultdict(list)
    for beam in beams:
        groups[int(beam.wavelength_nm)].append(beam)
    return {wavelength: tuple(group) for wavelength, group in groups.items()}


def filter_beams_by_wavelength(
    beams: Iterable[Beam], wavelength_nm: int
) -> tuple[Beam, ...]:
    """Return only beams at one wavelength."""

    return tuple(beam for beam in beams if beam.wavelength_nm == wavelength_nm)


__all__ = [
    "Beam",
    "center_electric_field_v_m",
    "center_intensity_w_m2",
    "filter_beams_by_wavelength",
    "gaussian_electric_field",
    "group_beams_by_wavelength",
    "rayleigh_range_um",
    "rayleigh_ranges_um",
    "waist_um",
]
