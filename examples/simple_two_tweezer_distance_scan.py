"""Minimal scan of two-tweezer interparticle distance."""

import matplotlib.pyplot as plt
import numpy as np

import beams_to_potentials as bpl
from example_systems import find_pair_minima, make_two_tweezer_beams


SPECIES_PAIR = ("Rb", "RbCs33")
POWER_817_MW = 0.015
POWER_1065_MW = 0.05
XS_1065_UM = np.linspace(0, 3.0, 100)
OUTPUT_CSV = "simple_two_tweezer_distance_scan.csv"

HELD_SETTINGS = (
    ("y=0.00 um, z=0.00 um", "distance_y0p00_z0p00_nm", 0.0, 0.0),
    ("y=0.10 um, z=0.00 um", "distance_y0p10_z0p00_nm", 0.1, 0.0),
    ("y=0.10 um, z=0.20 um", "distance_y0p10_z0p20_nm", 0.1, 0.2),
)


def distance_vs_x_1065(
    x_1065_um,
    *,
    y_1065_um=0.0,
    z_1065_um=0.0,
    power_817_mw=POWER_817_MW,
    power_1065_mw=POWER_1065_MW,
    species_pair=SPECIES_PAIR,
):
    """Return interparticle distance in nm for each 1065 tweezer x-position."""

    x_values = np.asarray(x_1065_um, dtype=float)
    if x_values.size == 0:
        raise ValueError("x_1065_um must contain at least one value.")

    distances_nm = np.empty_like(x_values)

    # Reuse the previous minimum as the next optimizer guess to track the same traps.
    guesses = (
        np.array((0.0, 0.0, 0.0)),
        np.array((x_values[0], y_1065_um, z_1065_um)),
    )

    for index, x_um in enumerate(x_values):
        beams = make_two_tweezer_beams(
            power_817_mw=power_817_mw,
            power_long_mw=power_1065_mw,
            x_long_um=x_um,
            y_long_um=y_1065_um,
            z_long_um=z_1065_um,
            long_wavelength_nm=1065,
        )
        minima = find_pair_minima(beams, species_pair, initial_guesses_um=guesses)
        distances_nm[index] = bpl.distance_nm(minima[0], minima[1])

        # The next point is usually close to this one.
        guesses = minima

    return distances_nm


def main():
    fig, ax = plt.subplots(constrained_layout=True)
    csv_columns = [XS_1065_UM]
    csv_headers = ["x_1065_um"]

    # Plot one curve for each held y/z displacement of the 1065 tweezer.
    for label, csv_header, y_1065_um, z_1065_um in HELD_SETTINGS:
        distances_nm = distance_vs_x_1065(
            XS_1065_UM,
            y_1065_um=y_1065_um,
            z_1065_um=z_1065_um,
        )
        ax.plot(XS_1065_UM, distances_nm, label=label)
        csv_columns.append(distances_nm)
        csv_headers.append(csv_header)

    np.savetxt(
        OUTPUT_CSV,
        np.column_stack(csv_columns),
        delimiter=",",
        header=",".join(csv_headers),
        comments="",
    )
    print(f"Saved {OUTPUT_CSV}")

    ax.set_xlabel("1065 tweezer x displacement (um)")
    ax.set_ylabel("Interparticle distance (nm)")
    ax.legend(fontsize="small")
    plt.show()


if __name__ == "__main__":
    main()
