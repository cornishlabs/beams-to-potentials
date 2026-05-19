"""Interactive and scan examples for 817/1065 tweezer overlap.

This file replaces the three previous overlap forks with one implementation:

* ``atom-atom`` compares Rb and Cs in the overlap of 817 nm and 1065 nm beams.
* ``molecule-atom`` compares Rb and the ``RbCs33`` molecular state.
* ``ratio-scan`` reproduces the fixed-ratio displacement scan.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from matplotlib.widgets import Slider

import beams_to_potentials as bpl


AXES = ("x", "y", "z")
MODE_SPECIES = {
    "atom-atom": ("Rb", "Cs"),
    "molecule-atom": ("Rb", "RbCs33"),
}

DEFAULT_POWER_817_MW = 2.5 * 0.78 * 0.6177 / 8
DEFAULT_POWER_1065_MW = 30 * 0.43 / 9
RATIO_SCAN_POWER_817_MW = 0.25

U = scipy.constants.u
HBAR = scipy.constants.hbar


def make_overlap_tweezers(
    power_817_mw: float,
    power_1065_mw: float,
    x_1065_um: float = 0.0,
    y_1065_um: float = 0.0,
    z_1065_um: float = 0.0,
) -> dict[tuple[int, str], list[dict[str, float]]]:
    """Return the library-shaped beam dictionary for the overlap examples."""

    return {
        (817, "817 tweezer"): [
            {
                "wx0": 0.885,
                "wy0": 0.985,
                "wz0": 0.995,
                "power_mW": power_817_mw,
                "theta": 0.0,
                "x0": 0.0,
                "y0": 0.0,
                "z0": 0.0,
                "phase": 0.0,
            }
        ],
        (1065, "1065 tweezer"): [
            {
                "wx0": 1.05,
                "wy0": 1.16,
                "wz0": 1.19,
                "power_mW": power_1065_mw,
                "theta": 0.0,
                "x0": x_1065_um,
                "y0": y_1065_um,
                "z0": z_1065_um,
                "phase": 0.0,
            }
        ],
    }


def only_wavelength(
    tweezers: dict[tuple[int, str], list[dict[str, float]]],
    wavelength_nm: int,
) -> dict[tuple[int, str], list[dict[str, float]]]:
    return {tag: beams for tag, beams in tweezers.items() if tag[0] == wavelength_nm}


def distance_nm(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)) * 1e3)


def find_minima(
    tweezers: dict[tuple[int, str], list[dict[str, float]]],
    species_pair: tuple[str, str],
    second_guess: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    first_min = bpl.my_find_potential_minimum(
        (0.0, 0.0, 0.0), tweezers, species_pair[0]
    )
    second_min = bpl.my_find_potential_minimum(second_guess, tweezers, species_pair[1])
    return first_min, second_min


def harmonic_density_overlay(
    xs: np.ndarray,
    centre_um: float,
    centre_potential_mhz: float,
    trap_frequency_mhz: float,
    mass_amu: float,
    potential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    omega = 2 * np.pi * trap_frequency_mhz * 1e6
    wavefunction = (mass_amu * U * omega / (np.pi * HBAR)) ** 0.25
    wavefunction *= np.exp(
        -(mass_amu * U * omega * ((xs - centre_um) * 1e-6) ** 2) / (2 * HBAR)
    )
    density_per_um = wavefunction**2 * 1e-6
    density_scale = np.nanmax(density_per_um)
    if not np.isfinite(density_scale) or density_scale == 0:
        return xs, np.full_like(xs, centre_potential_mhz), centre_potential_mhz

    potential_span = np.nanmax(potential) - np.nanmin(potential)
    overlay = (
        centre_potential_mhz + density_per_um / density_scale * 0.15 * potential_span
    )
    return xs, overlay, centre_potential_mhz


def plot_axis_potential(
    ax,
    axis_index: int,
    atom: str,
    centre: np.ndarray,
    tweezers: dict[tuple[int, str], list[dict[str, float]]],
    colour: str,
) -> list[float | None]:
    grids = (
        np.linspace(-2, 2, 81),
        np.linspace(-2, 2, 82),
        np.linspace(-8, 8, 91),
    )
    axis_grid = grids[axis_index]
    axis_slice = tuple(centre[d] if d != axis_index else axis_grid for d in range(3))

    total = bpl.total_potential(axis_slice, tweezers, atom)
    potential_817 = bpl.total_potential(
        axis_slice, only_wavelength(tweezers, 817), atom
    )
    potential_1065 = bpl.total_potential(
        axis_slice, only_wavelength(tweezers, 1065), atom
    )

    ax.plot(axis_grid, total, color=colour, label="total")
    ax.plot(axis_grid, potential_817, color="purple", linestyle="--", label="817")
    ax.plot(axis_grid, potential_1065, color="green", linestyle="--", label="1065")
    ax.axvline(
        centre[axis_index], color="black", linestyle="--", linewidth=1, alpha=0.5
    )
    ax.set_xlabel(f"{AXES[axis_index]} distance from 817 centre ($\\mu$m)")
    ax.set_ylabel(f"{atom} potential (MHz)")

    mass = bpl.species[atom]["m"]
    try:
        trap_frequency_mhz, quad_popt = bpl.my_get_trap_frequency(
            axis_grid,
            total,
            mass,
            x_fit_threshold_um=0.6,
            y_fit_threshold_ratio=0.8,
            x_bounds=(centre[axis_index] - 0.8, centre[axis_index] + 0.8),
        )
    except Exception as exc:
        ax.text(
            0.02,
            0.9,
            f"fit failed: {exc}",
            transform=ax.transAxes,
            fontsize="x-small",
            color="red",
        )
        return [None, None]

    fit_centre_um = float(quad_popt[0])
    fit_min_mhz = float(quad_popt[2])
    fit_xs = np.linspace(fit_centre_um - 0.6, fit_centre_um + 0.6, 200)
    ax.plot(fit_xs, bpl.quad(fit_xs, *quad_popt), color="red", linewidth=1)
    ax.axvline(fit_centre_um, color="red", linestyle="--", linewidth=1)

    overlay_xs = np.linspace(fit_centre_um - 1.3, fit_centre_um + 1.3, 200)
    wf_xs, wf_overlay, wf_base = harmonic_density_overlay(
        overlay_xs,
        fit_centre_um,
        fit_min_mhz,
        trap_frequency_mhz,
        mass,
        total,
    )
    ax.plot(wf_xs, wf_overlay, color="black", alpha=0.25)
    ax.fill_between(wf_xs, wf_overlay, y2=wf_base, color="black", alpha=0.08)
    ax.text(
        0.02,
        0.9,
        f"{trap_frequency_mhz * 1e3:.1f} kHz",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    return [fit_centre_um, trap_frequency_mhz * 1e3]


def run_interactive(mode: str = "atom-atom") -> None:
    species_pair = MODE_SPECIES[mode]
    initial_x_1065 = 0.75 if mode == "molecule-atom" else 0.0
    colours = ("#2571E1", "#EE2519")

    fig, axs = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("")

    ax_p817 = fig.add_axes([0.1, 0.98, 0.8, 0.01])
    p817_slider = Slider(
        ax=ax_p817,
        label="p817",
        valmin=0,
        valmax=0.5,
        valinit=DEFAULT_POWER_817_MW,
        orientation="horizontal",
    )

    ax_p1065 = fig.add_axes([0.1, 0.96, 0.8, 0.01])
    p1065_slider = Slider(
        ax=ax_p1065,
        label="p1065",
        valmin=0,
        valmax=10,
        valinit=DEFAULT_POWER_1065_MW,
        orientation="horizontal",
    )

    ax_x1065 = fig.add_axes([0.1, 0.94, 0.8, 0.01])
    x1065_slider = Slider(
        ax=ax_x1065,
        label="x1065",
        valmin=0,
        valmax=2,
        valinit=initial_x_1065,
        orientation="horizontal",
    )

    ax_y1065 = fig.add_axes([0.1, 0.92, 0.8, 0.01])
    y1065_slider = Slider(
        ax=ax_y1065,
        label="y1065",
        valmin=0,
        valmax=2,
        valinit=0,
        orientation="horizontal",
    )

    ax_z1065 = fig.add_axes([0.1, 0.9, 0.8, 0.01])
    z1065_slider = Slider(
        ax=ax_z1065,
        label="z1065",
        valmin=0,
        valmax=10,
        valinit=0,
        orientation="horizontal",
    )

    def update(_value: float) -> None:
        for ax in axs.ravel():
            ax.clear()

        axs[0, 0].set_axis_off()
        axs[0, 1].set_axis_off()

        tweezers = make_overlap_tweezers(
            p817_slider.val,
            p1065_slider.val,
            x1065_slider.val,
            y1065_slider.val,
            z1065_slider.val,
        )
        second_guess = (x1065_slider.val, y1065_slider.val, z1065_slider.val)
        minima = find_minima(tweezers, species_pair, second_guess)
        fit_centres = []

        for column, (atom, centre, colour) in enumerate(
            zip(species_pair, minima, colours)
        ):
            axs[0, column].text(
                0.5,
                0.5,
                f"{atom} minimum: ({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}) um",
                ha="center",
                va="center",
                transform=axs[0, column].transAxes,
            )
            atom_fit_centres = []
            for axis_index in range(3):
                fit_centre, _trap_frequency = plot_axis_potential(
                    axs[axis_index + 1, column],
                    axis_index,
                    atom,
                    centre,
                    tweezers,
                    colour,
                )
                atom_fit_centres.append(fit_centre)
            fit_centres.append(atom_fit_centres)

        gradient_distance = distance_nm(minima[0], minima[1])
        if all(value is not None for row in fit_centres for value in row):
            fit_distance = distance_nm(fit_centres[0], fit_centres[1])
            fit_text = f", fit distance {fit_distance:.1f} nm"
        else:
            fit_text = ""
        fig.suptitle(
            f"{species_pair[0]} vs {species_pair[1]}: gradient distance "
            f"{gradient_distance:.1f} nm{fit_text}",
            y=0.86,
        )
        fig.canvas.draw_idle()

    for slider in (p817_slider, p1065_slider, x1065_slider, y1065_slider, z1065_slider):
        slider.on_changed(update)

    update(0)
    plt.show()


def overlap_distance_nm(
    power_1065_mw: float,
    x_distance_um: float,
    y_distance_um: float,
    z_distance_um: float,
    species_pair: tuple[str, str] = ("Rb", "Cs"),
    power_817_mw: float = RATIO_SCAN_POWER_817_MW,
) -> float:
    tweezers = make_overlap_tweezers(
        power_817_mw,
        power_1065_mw,
        x_distance_um,
        y_distance_um,
        z_distance_um,
    )
    minima = find_minima(
        tweezers,
        species_pair,
        (x_distance_um, y_distance_um, z_distance_um),
    )
    return distance_nm(minima[0], minima[1])


def run_ratio_scan() -> None:
    x_distances_um = np.linspace(0, 0.8, 50)
    axial_offsets_um = np.linspace(0.0, 0.1, 3)
    horizontal_offsets_um = np.linspace(0.01, 0.2, 3)
    power_1065_mw = 10 * RATIO_SCAN_POWER_817_MW

    fig, ax = plt.subplots(constrained_layout=True)
    for axial_offset_um in axial_offsets_um:
        for horizontal_offset_um in horizontal_offsets_um:
            distances_nm = [
                overlap_distance_nm(
                    power_1065_mw,
                    x_distance_um,
                    horizontal_offset_um,
                    axial_offset_um,
                )
                for x_distance_um in x_distances_um
            ]
            ax.plot(
                x_distances_um,
                distances_nm,
                label=f"dH={horizontal_offset_um:.2f} um, dAx={axial_offset_um:.2f} um",
            )

    ax.set_xlabel("Tweezer x displacement (um)")
    ax.set_ylabel("Atom distance (nm)")
    ax.legend(fontsize="x-small")
    ax.set_title(f"P1065/P817 = {power_1065_mw / RATIO_SCAN_POWER_817_MW:.1f}")
    plt.show()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("atom-atom", "molecule-atom", "ratio-scan"),
        default="atom-atom",
        help="Example mode to run.",
    )
    args = parser.parse_args(argv)

    if args.mode == "ratio-scan":
        run_ratio_scan()
    else:
        run_interactive(args.mode)


if __name__ == "__main__":
    main()
