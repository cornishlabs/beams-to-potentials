"""Interactive and scan examples for 817/1065 tweezer overlap."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

import beams_to_potentials as bpl
from example_systems import (
    DEFAULT_POWER_817_MW,
    DEFAULT_POWER_LONG_MW,
    DEFAULT_SPECIES_PAIR,
    LONG_TWEEZER_WAVELENGTH_NM,
    SHORT_TWEEZER_WAVELENGTH_NM,
    find_pair_minima,
    make_two_tweezer_beams,
    species_options_for_wavelengths,
    systems_for_species,
)

SPECIES_OPTIONS = species_options_for_wavelengths(
    (SHORT_TWEEZER_WAVELENGTH_NM, LONG_TWEEZER_WAVELENGTH_NM)
)


class SpeciesSelector:
    """Compact species selector drawn in one Axes.

    This avoids nested Matplotlib ``Button`` widgets, which can fight over
    mouse-grab state when popup axes overlap sliders.
    """

    def __init__(
        self,
        fig,
        ax,
        label: str,
        options: Sequence[str],
        initial: str,
        on_changed,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.label = label
        self.options = tuple(options)
        self.value = initial
        self.on_changed = on_changed
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.draw()

    def draw(self) -> None:
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_xlim(-0.9, len(self.options))
        self.ax.set_ylim(0, 1)
        self.ax.text(
            -0.45,
            0.5,
            self.label,
            ha="center",
            va="center",
            fontsize="x-small",
            fontweight="bold",
        )

        for index, option in enumerate(self.options):
            selected = option == self.value
            self.ax.add_patch(
                Rectangle(
                    (index, 0.12),
                    1,
                    0.76,
                    facecolor="#d8e8ff" if selected else "#f5f5f5",
                    edgecolor="#4f6f93" if selected else "#bbbbbb",
                    linewidth=1,
                )
            )
            self.ax.text(
                index + 0.5,
                0.5,
                option,
                ha="center",
                va="center",
                fontsize="x-small",
            )
        self.fig.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        index = int(np.floor(event.xdata))
        if index < 0 or index >= len(self.options):
            return
        selected = self.options[index]
        if selected == self.value:
            return
        self.value = selected
        self.draw()
        self.on_changed(selected)


def find_minima(
    beams: Sequence[bpl.Beam],
    species_pair: tuple[str, str],
    second_guess: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return find_pair_minima(
        beams,
        species_pair,
        initial_guesses_um=((0.0, 0.0, 0.0), second_guess),
    )


def plot_axis_potential(
    ax,
    axis_index: int,
    system: bpl.PotentialSystem,
    centre_um: np.ndarray,
    colour: str,
) -> tuple[float | None, float | None]:
    grids = (
        np.linspace(-2, 2, 81),
        np.linspace(-2, 2, 82),
        np.linspace(-8, 8, 91),
    )
    grid = grids[axis_index]
    coord = bpl.axis_slice(centre_um, axis_index, grid)

    total = system.potential(coord)
    ax.plot(grid, total, color=colour, label="total")

    for wavelength_nm, line_colour in ((817, "purple"), (1065, "green")):
        beams = bpl.filter_beams_by_wavelength(system.beams, wavelength_nm)
        partial = system.with_beams(beams).potential(coord)
        ax.plot(grid, partial, color=line_colour, linestyle="--", label=wavelength_nm)

    ax.axvline(
        centre_um[axis_index], color="black", linestyle="--", linewidth=1, alpha=0.5
    )
    ax.set_xlabel(f"{bpl.AXES[axis_index]} distance from 817 centre (um)")
    ax.set_ylabel(f"{system.species.name} potential (MHz)")

    try:
        fit = bpl.fit_trap_frequency(
            grid,
            total,
            system.species.mass_amu,
            x_fit_threshold_um=0.6,
            y_fit_threshold_ratio=0.8,
            x_bounds=(centre_um[axis_index] - 0.8, centre_um[axis_index] + 0.8),
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
        return None, None

    fit_xs = np.linspace(fit.center_um - 0.6, fit.center_um + 0.6, 200)
    ax.plot(
        fit_xs,
        bpl.quadratic_potential(fit_xs, *fit.parameters),
        color="red",
        linewidth=1,
    )
    ax.axvline(fit.center_um, color="red", linestyle="--", linewidth=1)

    overlay_xs = np.linspace(fit.center_um - 1.3, fit.center_um + 1.3, 200)
    density = bpl.harmonic_oscillator_density_um(
        overlay_xs, fit.center_um, fit.frequency_mhz, system.species.mass_amu
    )
    density_scale = np.nanmax(density)
    if np.isfinite(density_scale) and density_scale > 0:
        potential_span = np.nanmax(total) - np.nanmin(total)
        overlay = fit.offset_mhz + density / density_scale * 0.15 * potential_span
        ax.plot(overlay_xs, overlay, color="black", alpha=0.25)
        ax.fill_between(
            overlay_xs, overlay, y2=fit.offset_mhz, color="black", alpha=0.08
        )

    fitted_point_um = bpl.axis_slice(centre_um, axis_index, fit.center_um)
    total_intensity_kw_cm2 = system.intensity_kw_cm2(fitted_point_um)
    gd_min_potential_mhz = float(system.potential(centre_um))

    ax.text(
        0.02,
        0.2,
        "$\\omega_{TF}$ = "
        f"{fit.frequency_khz:.1f} KHz\n"
        "$I_{total}$ = "
        f"{total_intensity_kw_cm2:.3g} kW/cm$^2$\n"
        "$U_{GDMin}$ = "
        f"{gd_min_potential_mhz:.3g} MHz",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    return fit.center_um, fit.frequency_khz


def run_interactive() -> None:
    selected_species = list(DEFAULT_SPECIES_PAIR)
    colours = ("#2571E1", "#EE2519")

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(
            rect=(0.04, 0.02, 0.94, 0.96),
            w_pad=1 / 72,
            h_pad=1 / 72,
            hspace=0.025,
            wspace=0.04,
        )

    grid = fig.add_gridspec(4, 1, height_ratios=[0.45, 1.6, 0.45, 7.5])
    species_grid = grid[0].subgridspec(1, 2)
    ax_left_species = fig.add_subplot(species_grid[0, 0])
    ax_right_species = fig.add_subplot(species_grid[0, 1])

    slider_axes = grid[1].subgridspec(5, 1).subplots()
    ax_status = fig.add_subplot(grid[2])
    ax_status.set_axis_off()
    status_text = ax_status.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize="small",
        transform=ax_status.transAxes,
    )
    axs = grid[3].subgridspec(3, 2).subplots()

    p817_slider = Slider(
        ax=slider_axes[0],
        label="p817",
        valmin=0,
        valmax=0.5,
        valinit=DEFAULT_POWER_817_MW,
        orientation="horizontal",
    )
    p817_slider.label.set_fontsize("x-small")
    p817_slider.valtext.set_fontsize("x-small")

    p1065_slider = Slider(
        ax=slider_axes[1],
        label="p1065",
        valmin=0,
        valmax=10,
        valinit=DEFAULT_POWER_LONG_MW,
        orientation="horizontal",
    )
    p1065_slider.label.set_fontsize("x-small")
    p1065_slider.valtext.set_fontsize("x-small")

    x1065_slider = Slider(
        ax=slider_axes[2],
        label="x1065",
        valmin=0,
        valmax=2,
        valinit=0.0,
        orientation="horizontal",
    )
    x1065_slider.label.set_fontsize("x-small")
    x1065_slider.valtext.set_fontsize("x-small")

    y1065_slider = Slider(
        ax=slider_axes[3],
        label="y1065",
        valmin=0,
        valmax=2,
        valinit=0,
        orientation="horizontal",
    )
    y1065_slider.label.set_fontsize("x-small")
    y1065_slider.valtext.set_fontsize("x-small")

    z1065_slider = Slider(
        ax=slider_axes[4],
        label="z1065",
        valmin=0,
        valmax=10,
        valinit=0,
        orientation="horizontal",
    )
    z1065_slider.label.set_fontsize("x-small")
    z1065_slider.valtext.set_fontsize("x-small")

    def set_left_species(species: str) -> None:
        selected_species[0] = species
        update(0)

    def set_right_species(species: str) -> None:
        selected_species[1] = species
        update(0)

    fig.species_selectors = (
        SpeciesSelector(
            fig,
            ax_left_species,
            "left",
            SPECIES_OPTIONS,
            selected_species[0],
            set_left_species,
        ),
        SpeciesSelector(
            fig,
            ax_right_species,
            "right",
            SPECIES_OPTIONS,
            selected_species[1],
            set_right_species,
        ),
    )

    def update(_value: float) -> None:
        for ax in axs.ravel():
            ax.clear()

        beams = make_two_tweezer_beams(
            p817_slider.val,
            p1065_slider.val,
            x1065_slider.val,
            y1065_slider.val,
            z1065_slider.val,
        )
        species_pair = (selected_species[0], selected_species[1])
        systems = systems_for_species(beams, species_pair)
        second_guess = (x1065_slider.val, y1065_slider.val, z1065_slider.val)
        minima = find_minima(beams, species_pair, second_guess)
        fit_centres = []

        for column, (system, centre, colour) in enumerate(
            zip(systems, minima, colours)
        ):
            atom_fit_centres = []
            for axis_index in range(3):
                fit_centre, _trap_frequency = plot_axis_potential(
                    axs[axis_index, column],
                    axis_index,
                    system,
                    centre,
                    colour,
                )
                atom_fit_centres.append(fit_centre)
            axs[0, column].set_title(
                f"{system.species.name} min: "
                f"({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}) um",
                fontsize="small",
            )
            fit_centres.append(atom_fit_centres)

        gradient_distance = bpl.distance_nm(minima[0], minima[1])
        if all(value is not None for row in fit_centres for value in row):
            fit_distance = bpl.distance_nm(fit_centres[0], fit_centres[1])
            fit_text = f", fit distance {fit_distance:.1f} nm"
        else:
            fit_text = ""
        status_text.set_text(
            f"{species_pair[0]} vs {species_pair[1]}: gradient distance "
            f"{gradient_distance:.1f} nm{fit_text}"
        )
        fig.canvas.draw_idle()

    for slider in (p817_slider, p1065_slider, x1065_slider, y1065_slider, z1065_slider):
        slider.on_changed(update)

    update(0)
    plt.show()


def main() -> None:
    run_interactive()


if __name__ == "__main__":
    main()
