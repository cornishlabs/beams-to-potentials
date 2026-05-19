"""Print trap properties for one named lab beam and one species."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence

import numpy as np

import beams_to_potentials as bpl
from example_systems import LAB_BEAM_TEMPLATES, make_lab_beam


def compatible_species(wavelength_nm: int) -> tuple[str, ...]:
    return tuple(
        name
        for name, species in bpl.SPECIES.items()
        if wavelength_nm in species.polarizabilities_au
    )


def axis_grids_for_beam(beam: bpl.Beam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rayleigh_z_um = bpl.rayleigh_ranges_um(beam)[2]
    axial_span_um = max(0.35 * rayleigh_z_um, 8.0)
    transverse_x_span_um = 3 * beam.waist_x_um

    # The beam propagates in the x-z plane. Choose line-cut spans that cover
    # both transverse and axial variation for the current beam angle.
    spans_um = (
        max(
            abs(np.cos(beam.theta_rad)) * transverse_x_span_um,
            abs(np.sin(beam.theta_rad)) * axial_span_um,
            2.0,
        ),
        max(3 * beam.waist_y_um, 2.0),
        max(
            abs(np.sin(beam.theta_rad)) * transverse_x_span_um,
            abs(np.cos(beam.theta_rad)) * axial_span_um,
            2.0,
        ),
    )
    return tuple(
        np.linspace(beam.center_um[index] - span, beam.center_um[index] + span, 301)
        for index, span in enumerate(spans_um)
    )


def format_frequency_khz(frequency_khz: float, curvature_mhz_per_um2: float) -> str:
    if curvature_mhz_per_um2 <= 0 or not np.isfinite(frequency_khz):
        return "not trapped"
    return f"{frequency_khz:.3g} kHz"


def print_beam_list() -> None:
    print("Available lab beams:")
    for name, beam in sorted(LAB_BEAM_TEMPLATES.items()):
        species = ", ".join(compatible_species(beam.wavelength_nm))
        print(
            f"  {name:14s} {beam.wavelength_nm:4d} nm, "
            f"waists=({beam.waist_x_um:g}, {beam.waist_y_um:g}, "
            f"{beam.waist_z_um:g}) um, species: {species}"
        )


def summarize_single_beam(
    beam_name: str,
    species: str,
    power_mw: float,
    fit_radius_um: float,
) -> None:
    beam = make_lab_beam(beam_name, power_mw=power_mw)
    if species not in compatible_species(beam.wavelength_nm):
        allowed = ", ".join(compatible_species(beam.wavelength_nm))
        raise ValueError(
            f"{species!r} has no polarizability at {beam.wavelength_nm} nm. "
            f"Try one of: {allowed}."
        )

    system = bpl.PotentialSystem([beam], species)
    centre_um = beam.center_um
    potential_mhz = float(system.potential(centre_um))
    depth_mhz = -potential_mhz
    depth_mk = bpl.mhz_to_mk(depth_mhz)
    intensity_kw_cm2 = float(system.intensity_kw_cm2(centre_um))
    rayleigh_ranges_um = bpl.rayleigh_ranges_um(beam)

    analysis = None
    fit_error = None
    if depth_mhz > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                analysis = bpl.analyze_trap(
                    system,
                    minimum_um=centre_um,
                    axis_grids=axis_grids_for_beam(beam),
                    fit_threshold_um=fit_radius_um,
                    fit_bounds_radius_um=fit_radius_um,
                )
        except Exception as exc:
            fit_error = str(exc)

    print(f"Beam: {beam_name}")
    print(f"Species: {system.species.name}")
    print(f"Power: {beam.power_mw:g} mW")
    print(f"Wavelength: {beam.wavelength_nm:g} nm")
    print(
        "Waists: "
        f"x={beam.waist_x_um:g} um, "
        f"y={beam.waist_y_um:g} um, "
        f"z={beam.waist_z_um:g} um"
    )
    print(
        "Rayleigh ranges: "
        f"x={rayleigh_ranges_um[0]:.4g} um, "
        f"y={rayleigh_ranges_um[1]:.4g} um, "
        f"z={rayleigh_ranges_um[2]:.4g} um"
    )
    print(f"Centre intensity: {intensity_kw_cm2:.4g} kW/cm^2")
    print(f"Potential at centre: {potential_mhz:.4g} MHz")
    print(f"Trap depth (-U): {depth_mhz:.4g} MHz = {depth_mk:.4g} mK")
    if depth_mhz <= 0:
        print(
            "Note: negative depth means this species is repelled by this beam centre."
        )

    print("Trap frequencies:")
    if analysis is None:
        for axis in bpl.AXES:
            print(f"  {axis}: not trapped")
        if fit_error is not None:
            print(f"Fit note: {fit_error}")
    else:
        for axis in analysis.axes:
            print(
                f"  {axis.axis}: "
                f"{format_frequency_khz(axis.fit.frequency_khz, axis.fit.curvature_mhz_per_um2)}"
            )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "beam",
        nargs="?",
        default="1065-tweezer",
        choices=sorted(LAB_BEAM_TEMPLATES),
        help="Named beam from examples/example_systems.py.",
    )
    parser.add_argument("--species", default="Cs", help="Species name, e.g. Rb or Cs.")
    parser.add_argument(
        "--power-mw",
        type=float,
        default=1.0,
        help="Beam power in mW.",
    )
    parser.add_argument(
        "--fit-radius-um",
        type=float,
        default=0.8,
        help="Half-width around the centre used for each trap-frequency fit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List named beams and compatible species, then exit.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.list:
        print_beam_list()
        return

    summarize_single_beam(
        parsed.beam,
        parsed.species,
        parsed.power_mw,
        parsed.fit_radius_um,
    )


if __name__ == "__main__":
    main()
