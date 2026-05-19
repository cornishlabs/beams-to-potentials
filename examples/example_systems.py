"""Shared system builders for example scripts.

Most lab scripts should start by editing a builder like this one: keep the
geometry and powers in one place, then let plotting/scan scripts call it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import beams_to_potentials as bpl


SHORT_TWEEZER_WAVELENGTH_NM = 817
LONG_TWEEZER_WAVELENGTH_NM = 1065

DEFAULT_SPECIES_PAIR = ("Rb", "Cs")
DEFAULT_POWER_817_MW = 2.5 * 0.78 * 0.6177 / 8
DEFAULT_POWER_LONG_MW = 30 * 0.43 / 9

LAB_BEAM_TEMPLATES = {
    "817-tweezer": bpl.Beam(
        wavelength_nm=817,
        waist_x_um=0.885,
        waist_y_um=0.985,
        waist_z_um=0.995,
        power_mw=1.0,
        label="817 tweezer",
    ),
    "1065-tweezer": bpl.Beam(
        wavelength_nm=1065,
        waist_x_um=1.05,
        waist_y_um=1.16,
        waist_z_um=1.19,
        power_mw=1.0,
        label="1065 tweezer",
    ),
    "1066-tweezer": bpl.Beam(
        wavelength_nm=1066,
        waist_x_um=1.05,
        waist_y_um=1.16,
        waist_z_um=1.19,
        power_mw=1.0,
        label="1066 tweezer",
    ),
    "1145-tweezer": bpl.Beam(
        wavelength_nm=1145,
        waist_x_um=1.7,
        waist_y_um=1.7,
        waist_z_um=1.7,
        power_mw=1.0,
        label="1145 tweezer",
    ),
    "1145-lattice": bpl.Beam(
        wavelength_nm=1145,
        waist_x_um=100,
        waist_y_um=40,
        waist_z_um=60,
        power_mw=1.0,
        theta_rad=np.pi / 2,
        label="1145 lattice beam",
    ),
}


def make_lab_beam(name: str, power_mw: float | None = None) -> bpl.Beam:
    """Return a named lab beam, optionally with a new power."""

    try:
        beam = LAB_BEAM_TEMPLATES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(LAB_BEAM_TEMPLATES))
        raise KeyError(f"Unknown lab beam {name!r}. Available: {choices}.") from exc

    if power_mw is None:
        return beam
    return beam.with_updates(power_mw=power_mw)


def species_options_for_wavelengths(wavelengths_nm: Sequence[int]) -> tuple[str, ...]:
    """Species with polarizabilities for every requested wavelength."""

    wavelengths = set(wavelengths_nm)
    return tuple(
        name
        for name, species in bpl.SPECIES.items()
        if wavelengths.issubset(species.polarizabilities_au)
    )


def make_two_tweezer_beams(
    power_817_mw: float = DEFAULT_POWER_817_MW,
    power_long_mw: float = DEFAULT_POWER_LONG_MW,
    x_long_um: float = 0.0,
    y_long_um: float = 0.0,
    z_long_um: float = 0.0,
    long_wavelength_nm: int = LONG_TWEEZER_WAVELENGTH_NM,
) -> tuple[bpl.Beam, ...]:
    """Return the common 817 plus long-wavelength tweezer overlap system."""

    return (
        bpl.Beam(
            wavelength_nm=SHORT_TWEEZER_WAVELENGTH_NM,
            waist_x_um=0.885,
            waist_y_um=0.985,
            waist_z_um=0.995,
            power_mw=power_817_mw,
            label="817 tweezer",
        ),
        bpl.Beam(
            wavelength_nm=long_wavelength_nm,
            waist_x_um=1.05,
            waist_y_um=1.16,
            waist_z_um=1.19,
            power_mw=power_long_mw,
            center_um=(x_long_um, y_long_um, z_long_um),
            label=f"{long_wavelength_nm} tweezer",
        ),
    )


def systems_for_species(
    beams: Sequence[bpl.Beam],
    species_pair: tuple[str, str] = DEFAULT_SPECIES_PAIR,
) -> tuple[bpl.PotentialSystem, bpl.PotentialSystem]:
    """Build one potential system for each species in ``species_pair``."""

    return tuple(bpl.PotentialSystem(beams, species) for species in species_pair)


def find_pair_minima(
    beams: Sequence[bpl.Beam],
    species_pair: tuple[str, str] = DEFAULT_SPECIES_PAIR,
    initial_guesses_um: tuple[Sequence[float], Sequence[float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """Find minima for the two species in a two-particle comparison."""

    systems = systems_for_species(beams, species_pair)
    return tuple(
        system.find_minimum(initial_guess)
        for system, initial_guess in zip(systems, initial_guesses_um)
    )
