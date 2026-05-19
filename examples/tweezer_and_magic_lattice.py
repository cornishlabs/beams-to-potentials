"""RbCs tweezer and magic-lattice example."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import beams_to_potentials as bpl


def make_beams(
    lattice_power_mw: float = 25 * 7,
    tweezer_power_mw: float = 1,
    tweezer_spacing_um: float = 6,
) -> tuple[bpl.Beam, ...]:
    return (
        bpl.Beam(
            wavelength_nm=1145,
            waist_x_um=100,
            waist_y_um=40,
            waist_z_um=60,
            power_mw=lattice_power_mw,
            theta_rad=np.pi / 2,
            label="lattice +",
        ),
        bpl.Beam(
            wavelength_nm=1145,
            waist_x_um=100,
            waist_y_um=40,
            waist_z_um=60,
            power_mw=lattice_power_mw,
            theta_rad=np.pi / 2 + np.pi,
            label="lattice -",
        ),
        bpl.Beam(
            wavelength_nm=1145,
            waist_x_um=1.7,
            waist_y_um=1.7,
            waist_z_um=1.7,
            power_mw=tweezer_power_mw,
            center_um=(0.143, 0, tweezer_spacing_um),
            phase_rad=-1.9,
            label="tweezer +",
        ),
        bpl.Beam(
            wavelength_nm=1145,
            waist_x_um=1.7,
            waist_y_um=1.7,
            waist_z_um=1.7,
            power_mw=tweezer_power_mw,
            center_um=(-0.143, 0, -tweezer_spacing_um),
            label="tweezer -",
        ),
    )


def make_system(beams: Sequence[bpl.Beam]) -> bpl.PotentialSystem:
    return bpl.PotentialSystem(beams, "RbCs")


def plot_trap_analysis(analysis: bpl.TrapAnalysis) -> None:
    fig, axs = plt.subplots(2, 3, height_ratios=(4, 1), sharey="row", sharex="none")

    for axis in analysis.axes:
        ax, ax_low = axs.T[axis.axis_index]
        ax.plot(
            axis.grid_um,
            axis.potential_mhz,
            c="red",
            label=analysis.system.species.name,
        )
        ax.axvline(axis.fit.center_um, lw=1, linestyle="--", color="red")
        ax.axvline(
            analysis.minimum_um[axis.axis_index], lw=1, linestyle="-", color="red"
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
            10,
            f"TF {axis.fit.frequency_khz:.1f}(kHz)",
            size="x-small",
            color="red",
            ha="left",
        )
        ax_low.plot(density_xs, density, c="red", alpha=0.4)
        ax_low.fill_between(density_xs, density, y2=0, color="red", alpha=0.2)

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


def plot_potential_2d(
    system: bpl.PotentialSystem,
    analysis: bpl.TrapAnalysis | None = None,
    vmin: float = -2.5,
    vmax: float = 0,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.figure

    x = np.linspace(-5, 5, 200)
    z = np.linspace(-20, 20, 200)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing="ij")

    potential = system.potential((x_mesh, 0, z_mesh))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    contour = ax.contourf(
        x_mesh, z_mesh, potential, levels=1000, cmap="afmhot_r", norm=norm
    )
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Potential (MHz)")

    ax.set_xlabel(r"$x_{\mathrm{Twe}}$ (um)")
    ax.set_ylabel(r"$z_{\mathrm{Twe}}$ (um)")

    if analysis is not None:
        ellipse = mpl.patches.Ellipse(
            xy=(analysis.fit_centres_um[0], analysis.fit_centres_um[2]),
            width=analysis.confinement_lengths_um[0],
            height=analysis.confinement_lengths_um[2],
            edgecolor="red",
            fc="none",
            lw=2,
        )
        ax.add_patch(ellipse)

    return fig, ax


def track_tweezers(spacings_um: Sequence[float]) -> np.ndarray:
    positions = [[np.array([0.0, 0.0, -6.0]), np.array([0.0, 0.0, 6.0])]]
    length_scales = []

    for index, spacing_um in enumerate(spacings_um):
        system = make_system(make_beams(tweezer_spacing_um=spacing_um))
        previous_positions = positions[-1]
        positions.append([])
        length_scales.append([])

        fig_2d, ax_2d = plot_potential_2d(system, vmin=-3.8, vmax=0)
        fig, axs = plt.subplots(2, 3, height_ratios=(4, 1), sharey="row", sharex="none")

        for copy_index, previous_position in enumerate(previous_positions):
            analysis = bpl.analyze_trap(
                system,
                previous_position,
                axis_grids=(
                    np.linspace(-4, 4, 200),
                    np.linspace(-4, 4, 42),
                    np.linspace(-10, 10, 400),
                ),
                fit_threshold_um=0.5,
                fit_bounds_radius_um=0.3,
            )
            positions[-1].append(analysis.minimum_um)
            length_scales[-1].append(analysis.confinement_lengths_um)

            if copy_index == 0:
                for axis in analysis.axes:
                    ax, ax_low = axs.T[axis.axis_index]
                    ax.plot(axis.grid_um, axis.potential_mhz, c="red", label="RbCs")
                    ax.axvline(axis.fit.center_um, lw=1, linestyle="--", color="red")
                    density_xs = np.linspace(
                        axis.fit.center_um - 1.3, axis.fit.center_um + 1.3, 200
                    )
                    density = bpl.harmonic_oscillator_density_um(
                        density_xs,
                        axis.fit.center_um,
                        axis.fit.frequency_mhz,
                        system.species.mass_amu,
                    )
                    ax_low.text(
                        axis.fit.center_um,
                        10,
                        f"TF {axis.fit.frequency_khz:.1f}(kHz)",
                        size="x-small",
                        color="red",
                    )
                    ax_low.plot(density_xs, density, c="red", alpha=0.4)
                    ax_low.fill_between(
                        density_xs, density, y2=0, color="red", alpha=0.2
                    )

        for axes in axs.T:
            axes[-1].set_xlabel("position (um)")
        axs[1, 0].set_ylabel(r"$|\psi|$ (um$^{-1}$)")
        axs[0, 0].set_ylabel("Trap depth (MHz)")
        axs[0, 0].legend()

        for copy_index in range(2):
            ellipse = mpl.patches.Ellipse(
                xy=(positions[-1][copy_index][0], positions[-1][copy_index][2]),
                width=length_scales[-1][copy_index][0],
                height=length_scales[-1][copy_index][2],
                edgecolor="none",
                fc="green",
                lw=0.2,
            )
            ax_2d.add_patch(ellipse)

        fig_2d.savefig(f"{index}.png")
        plt.show()

    return np.array(positions)


def main() -> None:
    lattice_free_system = make_system(make_beams(lattice_power_mw=0))
    lattice_free_analysis = bpl.analyze_trap(
        lattice_free_system,
        (-0.143, 0, -5.5),
        axis_grids=(
            np.linspace(-4, 4, 200),
            np.linspace(-2, 2, 42),
            np.linspace(-15, 15, 400),
        ),
        fit_threshold_um=0.5,
        fit_bounds_radius_um=0.3,
    )
    plot_trap_analysis(lattice_free_analysis)

    full_system = make_system(make_beams())
    plot_potential_2d(full_system, lattice_free_analysis)
    plt.show()

    spacings = np.linspace(6, 1, 100)
    positions = track_tweezers(spacings)

    fig, ax = plt.subplots()
    ax.plot(spacings, positions[:, 0, 2][1:], label="z0")
    ax.plot(spacings, positions[:, 1, 2][1:], label="z1")
    ax.set_xlabel("tweezer spacing from zero (um)")
    ax.set_ylabel("z-particle position (um)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
