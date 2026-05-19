"""Scan a parameter of the default two-tweezer system.

This example scans the x-position of the long-wavelength tweezer and tracks
interparticle separation, trap depth, and x-axis trap frequency for a few held
values of the other tweezer offsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import beams_to_potentials as bpl
from example_systems import (
    find_pair_minima,
    make_two_tweezer_beams,
    systems_for_species,
)


@dataclass(frozen=True)
class HeldParameters:
    label: str
    y_long_um: float = 0.0
    z_long_um: float = 0.0
    power_817_mw: float = 0.015
    power_long_mw: float = 0.05


@dataclass(frozen=True)
class ScanResult:
    holding: HeldParameters
    x_long_um: np.ndarray
    separation_nm: np.ndarray
    depths_mhz: np.ndarray
    frequencies_khz: np.ndarray
    minima_um: np.ndarray


X_LONG_UM = np.linspace(0, 2.0, 40)
SCAN_LONG_WAVELENGTH_NM = 1065
HELD_PARAMETERS = (
    HeldParameters("y=0.00 um, z=0.00 um", y_long_um=0.0, z_long_um=0.0),
    HeldParameters("y=0.1 um, z=0.00 um", y_long_um=0.1, z_long_um=0.0),
    HeldParameters("y=0.1 um, z=0.2 um", y_long_um=0.1, z_long_um=0.2),
)
SCAN_SPECIES_PAIR = ("Rb", "RbCs33")
AXIS_GRIDS = (
    np.linspace(-2, 2, 81),
    np.linspace(-2, 2, 81),
    np.linspace(-8, 8, 121),
)


def analyze_two_tweezer_scan(
    x_long_um: Sequence[float],
    holding: HeldParameters,
    species_pair: tuple[str, str] = SCAN_SPECIES_PAIR,
) -> ScanResult:
    separations = []
    depths = []
    frequencies = []
    minima = []
    guesses = (
        np.array((0.0, 0.0, 0.0)),
        np.array((x_long_um[0], holding.y_long_um, holding.z_long_um)),
    )

    for x_um in x_long_um:
        beams = make_two_tweezer_beams(
            power_817_mw=holding.power_817_mw,
            power_long_mw=holding.power_long_mw,
            x_long_um=x_um,
            y_long_um=holding.y_long_um,
            z_long_um=holding.z_long_um,
            long_wavelength_nm=SCAN_LONG_WAVELENGTH_NM,
        )
        systems = systems_for_species(beams, species_pair)
        point_minima = find_pair_minima(beams, species_pair, initial_guesses_um=guesses)
        guesses = point_minima

        point_depths = []
        point_frequencies = []
        for system, minimum in zip(systems, point_minima):
            analysis = bpl.analyze_trap(
                system,
                minimum_um=minimum,
                axis_grids=AXIS_GRIDS,
                fit_threshold_um=0.6,
                fit_bounds_radius_um=0.8,
            )
            point_depths.append(analysis.potential_minimum_mhz)
            point_frequencies.append(analysis.frequencies_khz)

        separations.append(bpl.distance_nm(point_minima[0], point_minima[1]))
        depths.append(point_depths)
        frequencies.append(point_frequencies)
        minima.append(point_minima)

    return ScanResult(
        holding=holding,
        x_long_um=np.asarray(x_long_um),
        separation_nm=np.asarray(separations),
        depths_mhz=np.asarray(depths),
        frequencies_khz=np.asarray(frequencies),
        minima_um=np.asarray(minima),
    )


def plot_scan_results(results: Sequence[ScanResult]) -> None:
    fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True, figsize=(8, 8))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    species_colours = ("#2571E1", "#EE2519")

    for result_index, result in enumerate(results):
        colour = colours[result_index % len(colours)]
        axs[0].plot(
            result.x_long_um,
            result.separation_nm,
            label=result.holding.label,
            color=colour,
        )

        for species_index, species in enumerate(SCAN_SPECIES_PAIR):
            linestyle = "-" if species_index == 0 else "--"
            axs[1].plot(
                result.x_long_um,
                -result.depths_mhz[:, species_index],
                color=species_colours[species_index],
                alpha=0.35 + 0.18 * result_index,
                linestyle=linestyle,
                label=f"{species}, {result.holding.label}",
            )
            axs[2].plot(
                result.x_long_um,
                result.frequencies_khz[:, species_index, 0],
                color=species_colours[species_index],
                alpha=0.35 + 0.18 * result_index,
                linestyle=linestyle,
                label=f"{species}, {result.holding.label}",
            )

    axs[0].set_ylabel("Separation (nm)")
    axs[1].set_ylabel("-U at minimum (MHz)")
    axs[2].set_ylabel("x trap frequency (kHz)")
    axs[2].set_xlabel(
        f"{SCAN_LONG_WAVELENGTH_NM} nm tweezer x-position (um)"
    )

    axs[0].legend(fontsize="x-small")
    axs[1].legend(fontsize="xx-small", ncol=2)
    axs[2].legend(fontsize="xx-small", ncol=2)
    plt.show()


def main() -> None:
    results = [
        analyze_two_tweezer_scan(X_LONG_UM, holding) for holding in HELD_PARAMETERS
    ]
    plot_scan_results(results)


if __name__ == "__main__":
    main()
