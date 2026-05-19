"""Tweezer plus state-dependent optical lattice example."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import beams_to_potentials as bpl


SPECIES_TO_COMPARE = ("Rb", "Cs")
COLOURS = ("blue", "red")


def make_beams(
    sal_power_mw: float = 200, sal_phase_rad: float = 1.9
) -> tuple[bpl.Beam, ...]:
    return (
        bpl.Beam(
            wavelength_nm=1066,
            waist_x_um=1.05,
            waist_y_um=1.16,
            waist_z_um=1.19,
            power_mw=4.4,
            label="1066 tweezer",
        ),
        bpl.Beam(
            wavelength_nm=817,
            waist_x_um=100,
            waist_y_um=40,
            waist_z_um=60,
            power_mw=sal_power_mw,
            theta_rad=np.pi / 2 + 7.5 * (2 * np.pi / 360),
            label="SAL +",
        ),
        bpl.Beam(
            wavelength_nm=817,
            waist_x_um=100,
            waist_y_um=40,
            waist_z_um=60,
            power_mw=sal_power_mw,
            theta_rad=np.pi / 2 - 7.5 * (2 * np.pi / 360),
            phase_rad=sal_phase_rad,
            label="SAL -",
        ),
    )


def systems_for_beams(beams: Sequence[bpl.Beam]) -> tuple[bpl.PotentialSystem, ...]:
    return tuple(bpl.PotentialSystem(beams, species) for species in SPECIES_TO_COMPARE)


def scan_sal_power(power_mw: float) -> tuple[bpl.PotentialSystem, ...]:
    return systems_for_beams(make_beams(sal_power_mw=power_mw))


def scan_sal_phase(phase_rad: float) -> tuple[bpl.PotentialSystem, ...]:
    return systems_for_beams(make_beams(sal_phase_rad=phase_rad))


def run_scan(
    values: Sequence[float], factory
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scan = bpl.scan_parameter(
        values,
        factory,
        initial_guesses_um=((0, 0, 0), (0, 0, 0)),
        fit_threshold_um=0.5,
        fit_bounds_radius_um=0.3,
    )
    return bpl.scan_arrays(scan)


def plot_trap_analysis(analyses: Sequence[bpl.TrapAnalysis]) -> None:
    fig, axs = plt.subplots(2, 3, height_ratios=(4, 1), sharey="row", sharex="none")

    for analysis, colour in zip(analyses, COLOURS):
        for axis in analysis.axes:
            ax, ax_low = axs.T[axis.axis_index]
            ax.plot(
                axis.grid_um,
                axis.potential_mhz,
                c=colour,
                label=analysis.system.species.name,
            )
            ax.axvline(axis.fit.center_um, lw=1, linestyle="--", color=colour)
            ax.axvline(
                analysis.minimum_um[axis.axis_index], lw=1, linestyle="-", color=colour
            )

            density_xs = np.linspace(
                axis.fit.center_um - 1.3, axis.fit.center_um + 1.3, 200
            )
            density = bpl.harmonic_oscillator_density_um(
                density_xs,
                axis.fit.center_um,
                axis.fit.frequency_mhz,
                analysis.system.species.mass_amu,
            )
            ax_low.text(
                axis.fit.center_um,
                6 - (3 if analysis.system.species.name == "Rb" else 0),
                f"TF {axis.fit.frequency_khz:.1f}(kHz)",
                size="x-small",
                color=colour,
                ha="right" if analysis.system.species.name == "Cs" else "left",
            )
            ax_low.plot(density_xs, density, c=colour, alpha=0.4)
            ax_low.fill_between(density_xs, density, y2=0, color=colour, alpha=0.2)

    for ax in axs[1, :]:
        ax.set_xlim(-1.2, 1.2)
    axs[0, 0].set_xlim(-4, 4)
    axs[1, 0].set_xlim(-4, 4)
    axs[0, 1].set_xlim(-2, 2)
    axs[0, 2].set_xlim(-10, 10)
    axs[1, 0].set_ylabel(r"$|\psi|$ (um$^{-1}$)")
    axs[0, 0].legend()

    for ax_pair, axis_name in zip(axs.T, bpl.AXES):
        ax_pair[-1].set_xlabel(f"{axis_name} (um)")

    axs[0, 0].set_ylabel("Trap depth (MHz)")
    plt.show()


def plot_potential_2d(beams: Sequence[bpl.Beam], species: str = "Cs") -> None:
    system = bpl.PotentialSystem(beams, species)
    fig, ax = plt.subplots(constrained_layout=True)

    x = np.linspace(-5, 5, 200)
    z = np.linspace(-20, 20, 200)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing="ij")

    potential = system.potential((x_mesh, 0, z_mesh))
    norm = mpl.colors.Normalize(vmin=-7, vmax=+3)

    contour = ax.contourf(
        x_mesh, z_mesh, potential, levels=1000, cmap="afmhot_r", norm=norm
    )
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Potential (MHz)")

    ax.set_xlabel(r"$x_{\mathrm{Twe}}$ (um)")
    ax.set_ylabel(r"$z_{\mathrm{Twe}}$ (um)")
    plt.show()


def plot_scan(
    values: Sequence[float],
    potential_minima_mhz: np.ndarray,
    fit_centres_um: np.ndarray,
    frequencies_khz: np.ndarray,
    xlabel: str,
) -> None:
    fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    for species_index, species in enumerate(SPECIES_TO_COMPARE):
        colour = COLOURS[species_index]
        axs[0].plot(
            values, -potential_minima_mhz[:, species_index], label=species, color=colour
        )
        for axis_index, axis_name in enumerate(bpl.AXES):
            linestyle = ["-", "--", ":"][axis_index]
            label = f"{species}, {axis_name}"
            axs[1].plot(
                values,
                fit_centres_um[:, species_index, axis_index],
                label=label,
                color=colour,
                ls=linestyle,
            )
            axs[2].plot(
                values,
                frequencies_khz[:, species_index, axis_index],
                label=label,
                color=colour,
                ls=linestyle,
            )

    axs[0].set_ylabel("Trap Depth (MHz)")
    axs[1].set_ylabel("Position (um)")
    axs[2].set_ylabel("Trap Frequency (KHz)")
    axs[1].legend(
        bbox_to_anchor=(0.85, 1.16),
        ncol=6,
        fancybox=True,
        shadow=False,
        fontsize="x-small",
    )
    axs[2].set_xlabel(xlabel)
    plt.show()


def main() -> None:
    no_power_scan = bpl.scan_parameter(
        [0],
        scan_sal_power,
        initial_guesses_um=((0, 0, 0), (0, 0, 0)),
        fit_threshold_um=0.5,
        fit_bounds_radius_um=0.3,
    )
    plot_trap_analysis(no_power_scan[0].analyses)

    plot_potential_2d(make_beams(), species="Cs")

    powers_mw = np.linspace(0, 500, 100)
    power_results = run_scan(powers_mw, scan_sal_power)
    plot_scan(powers_mw, *power_results, xlabel="SAL power per beam (mW)")

    phases_rad = np.linspace(-np.pi - 0.01, np.pi + 0.01, 100)
    phase_results = run_scan(phases_rad, scan_sal_phase)
    plot_scan(phases_rad, *phase_results, xlabel="Lattice phase (rad)")


if __name__ == "__main__":
    main()
