"""Fixed-ratio 817/1065 tweezer-overlap scan."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import beams_to_potentials as bpl
from example_systems import DEFAULT_SPECIES_PAIR, find_pair_minima, make_two_tweezer_beams


POWER_817_MW = 0.25
POWER_1065_TO_817_RATIO = 10


def overlap_distance_nm(
    power_1065_mw: float,
    x_distance_um: float,
    y_distance_um: float,
    z_distance_um: float,
    species_pair: tuple[str, str] = DEFAULT_SPECIES_PAIR,
    power_817_mw: float = POWER_817_MW,
) -> float:
    beams = make_two_tweezer_beams(
        power_817_mw,
        power_1065_mw,
        x_distance_um,
        y_distance_um,
        z_distance_um,
    )
    minima = find_pair_minima(
        beams,
        species_pair,
        initial_guesses_um=((0, 0, 0), (x_distance_um, y_distance_um, z_distance_um)),
    )
    return bpl.distance_nm(minima[0], minima[1])


def main() -> None:
    x_distances_um = np.linspace(0, 0.8, 50)
    axial_offsets_um = np.linspace(0.0, 0.1, 3)
    horizontal_offsets_um = np.linspace(0.01, 0.2, 3)
    power_1065_mw = POWER_1065_TO_817_RATIO * POWER_817_MW

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
    ax.set_title(f"P1065/P817 = {power_1065_mw / POWER_817_MW:.1f}")
    plt.show()


if __name__ == "__main__":
    main()
