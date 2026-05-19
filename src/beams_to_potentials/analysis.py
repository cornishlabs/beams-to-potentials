"""Reusable trap-analysis routines used by examples and downstream scripts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .constants import HBAR, U
from .fitting import TrapFrequencyFit, fit_trap_frequency
from .potentials import PotentialSystem

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class AxisTrapAnalysis:
    """Line cut and quadratic fit for one Cartesian axis."""

    axis: str
    axis_index: int
    grid_um: np.ndarray
    potential_mhz: np.ndarray
    fit: TrapFrequencyFit


@dataclass(frozen=True)
class TrapAnalysis:
    """Minimum, trap depth, and fitted frequencies for a system."""

    system: PotentialSystem
    minimum_um: np.ndarray
    potential_minimum_mhz: float
    axes: tuple[AxisTrapAnalysis, ...]

    @property
    def fit_centres_um(self) -> np.ndarray:
        return np.array([axis.fit.center_um for axis in self.axes])

    @property
    def frequencies_mhz(self) -> np.ndarray:
        return np.array([axis.fit.frequency_mhz for axis in self.axes])

    @property
    def frequencies_khz(self) -> np.ndarray:
        return self.frequencies_mhz * 1e3

    @property
    def confinement_lengths_um(self) -> np.ndarray:
        return np.array(
            [
                harmonic_length_um(axis.fit.frequency_mhz, self.system.species.mass_amu)
                for axis in self.axes
            ]
        )


@dataclass(frozen=True)
class ScanPoint:
    """Analysis results for one value in a parameter scan."""

    value: float
    analyses: tuple[TrapAnalysis, ...]


def default_axis_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Default line-cut grids used for x, y, and z trap fits."""

    return (
        np.linspace(-4, 4, 200),
        np.linspace(-2, 2, 42),
        np.linspace(-15, 15, 400),
    )


def axis_slice(
    centre_um: Sequence[float], axis_index: int, axis_values_um: Sequence[float]
) -> tuple[object, object, object]:
    """Coordinates for a one-dimensional cut through ``centre_um``."""

    return tuple(
        centre_um[index] if index != axis_index else axis_values_um
        for index in range(3)
    )


def fit_axis_trap(
    system: PotentialSystem,
    centre_um: Sequence[float],
    axis_index: int,
    axis_values_um: Sequence[float],
    fit_threshold_um: float = 0.5,
    fit_bounds_radius_um: float = 0.3,
    y_fit_threshold_ratio: float = 0.8,
) -> AxisTrapAnalysis:
    """Fit the potential along one axis through ``centre_um``."""

    grid = np.asarray(axis_values_um)
    potential = system.potential(axis_slice(centre_um, axis_index, grid))
    fit = fit_trap_frequency(
        grid,
        potential,
        system.species.mass_amu,
        x_fit_threshold_um=fit_threshold_um,
        y_fit_threshold_ratio=y_fit_threshold_ratio,
        x_bounds=(
            centre_um[axis_index] - fit_bounds_radius_um,
            centre_um[axis_index] + fit_bounds_radius_um,
        ),
    )
    return AxisTrapAnalysis(AXES[axis_index], axis_index, grid, potential, fit)


def trap_frequencies(
    system: PotentialSystem,
    centre_um: Sequence[float],
    axis_grids: Sequence[Sequence[float]] | None = None,
    fit_threshold_um: float = 0.5,
    fit_bounds_radius_um: float = 0.3,
    y_fit_threshold_ratio: float = 0.8,
) -> dict[str, TrapFrequencyFit]:
    """Return fitted trap frequencies for x, y, and z line cuts."""

    grids = axis_grids if axis_grids is not None else default_axis_grids()
    return {
        AXES[index]: fit_axis_trap(
            system,
            centre_um,
            index,
            grid,
            fit_threshold_um=fit_threshold_um,
            fit_bounds_radius_um=fit_bounds_radius_um,
            y_fit_threshold_ratio=y_fit_threshold_ratio,
        ).fit
        for index, grid in enumerate(grids)
    }


def analyze_trap(
    system: PotentialSystem,
    initial_guess_um: Sequence[float] | None = None,
    minimum_um: Sequence[float] | None = None,
    axis_grids: Sequence[Sequence[float]] | None = None,
    fit_threshold_um: float = 0.5,
    fit_bounds_radius_um: float = 0.3,
    y_fit_threshold_ratio: float = 0.8,
) -> TrapAnalysis:
    """Find or use a minimum, then fit trap frequencies along each axis."""

    if minimum_um is None:
        if initial_guess_um is None:
            raise ValueError("Provide either initial_guess_um or minimum_um.")
        minimum = system.find_minimum(initial_guess_um)
    else:
        minimum = np.asarray(minimum_um, dtype=float)

    grids = axis_grids if axis_grids is not None else default_axis_grids()
    axes = tuple(
        fit_axis_trap(
            system,
            minimum,
            index,
            grid,
            fit_threshold_um=fit_threshold_um,
            fit_bounds_radius_um=fit_bounds_radius_um,
            y_fit_threshold_ratio=y_fit_threshold_ratio,
        )
        for index, grid in enumerate(grids)
    )
    return TrapAnalysis(system, minimum, float(system.potential(minimum)), axes)


def scan_parameter(
    values: Sequence[float],
    system_factory: Callable[[float], Sequence[PotentialSystem]],
    initial_guesses_um: Sequence[Sequence[float]] | None = None,
    **analysis_kwargs,
) -> tuple[ScanPoint, ...]:
    """Run ``analyze_trap`` for each value produced by ``system_factory``.

    ``system_factory`` should return one or more ``PotentialSystem`` objects for
    the supplied scan value. This keeps scan scripts short while leaving the
    physical setup in ordinary Python code.
    """

    points = []
    for value in values:
        systems = tuple(system_factory(value))
        if initial_guesses_um is None:
            guesses = tuple((0.0, 0.0, 0.0) for _ in systems)
        else:
            guesses = tuple(initial_guesses_um)
        if len(guesses) != len(systems):
            raise ValueError("initial_guesses_um must match the number of systems.")

        analyses = tuple(
            analyze_trap(system, guess, **analysis_kwargs)
            for system, guess in zip(systems, guesses)
        )
        points.append(ScanPoint(float(value), analyses))
    return tuple(points)


def scan_arrays(scan: Sequence[ScanPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert scan results into arrays convenient for plotting."""

    potential_minima_mhz = np.array(
        [
            [analysis.potential_minimum_mhz for analysis in point.analyses]
            for point in scan
        ]
    )
    fit_centres_um = np.array(
        [[analysis.fit_centres_um for analysis in point.analyses] for point in scan]
    )
    frequencies_khz = np.array(
        [[analysis.frequencies_khz for analysis in point.analyses] for point in scan]
    )
    return potential_minima_mhz, fit_centres_um, frequencies_khz


def distance_um(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def distance_nm(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return distance_um(point_a, point_b) * 1e3


def harmonic_length_um(
    trap_frequency_mhz: float, mass_amu: float, sigma_scale: float = 3
) -> float:
    """Return ``sigma_scale`` harmonic-oscillator widths in um."""

    omega = 2 * np.pi * trap_frequency_mhz * 1e6
    sigma_m = np.sqrt(HBAR / (mass_amu * U * omega))
    return sigma_scale * sigma_m * 1e6


def harmonic_oscillator_density_um(
    xs_um: Sequence[float],
    centre_um: float,
    trap_frequency_mhz: float,
    mass_amu: float,
) -> np.ndarray:
    """One-dimensional harmonic oscillator probability density in um^-1."""

    xs = np.asarray(xs_um)
    omega = 2 * np.pi * trap_frequency_mhz * 1e6
    mass_kg = mass_amu * U
    displacement_m = (xs - centre_um) * 1e-6
    wavefunction = (mass_kg * omega / (np.pi * HBAR)) ** 0.25
    wavefunction *= np.exp(-(mass_kg * omega * displacement_m**2) / (2 * HBAR))
    return wavefunction**2 * 1e-6


__all__ = [
    "AXES",
    "AxisTrapAnalysis",
    "ScanPoint",
    "TrapAnalysis",
    "analyze_trap",
    "axis_slice",
    "default_axis_grids",
    "distance_nm",
    "distance_um",
    "fit_axis_trap",
    "harmonic_length_um",
    "harmonic_oscillator_density_um",
    "scan_arrays",
    "scan_parameter",
    "trap_frequencies",
]
