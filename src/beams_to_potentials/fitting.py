"""Fitting helpers for potentials."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .constants import H, U


def quadratic_potential(x, x0: float, curvature: float, y0: float):
    """Quadratic potential in MHz for position ``x`` in um."""

    return curvature / 2 * (x - x0) ** 2 + y0


@dataclass(frozen=True)
class TrapFrequencyFit:
    """Result of a one-dimensional harmonic trap fit."""

    frequency_mhz: float
    parameters: np.ndarray
    covariance: np.ndarray

    @property
    def center_um(self) -> float:
        return float(self.parameters[0])

    @property
    def curvature_mhz_per_um2(self) -> float:
        return float(self.parameters[1])

    @property
    def offset_mhz(self) -> float:
        return float(self.parameters[2])

    @property
    def frequency_khz(self) -> float:
        return self.frequency_mhz * 1e3


def trap_frequency_from_curvature_mhz_um2(curvature: float, mass_amu: float) -> float:
    """Convert quadratic curvature in MHz/um^2 into trap frequency in MHz."""

    return np.sqrt(curvature * 1e6 * H * (1e6) ** 2 / (mass_amu * U)) / 1e6 / 2 / np.pi


def fit_trap_frequency(
    xs_um: Sequence[float],
    potential_mhz: Sequence[float],
    mass_amu: float,
    x_fit_threshold_um: float = 1,
    y_fit_threshold_ratio: float = 0.8,
    x_bounds: tuple[float | None, float | None] = (None, None),
) -> TrapFrequencyFit:
    """Fit a quadratic near a potential minimum and return the trap frequency.

    The potential must be sampled in MHz on an ``xs_um`` grid. The fit first
    restricts to ``x_bounds`` if supplied, then fits points close to the lowest
    sampled value.
    """

    xs = np.asarray(xs_um)
    potential = np.asarray(potential_mhz)
    selector = np.full_like(xs, True, dtype=bool)
    if x_bounds[0] is not None:
        selector &= xs > x_bounds[0]
    if x_bounds[1] is not None:
        selector &= xs < x_bounds[1]

    fit_xs = xs[selector]
    fit_ys = potential[selector]
    if fit_xs.size < 3:
        raise ValueError(
            "Need at least three potential samples to fit a trap frequency."
        )

    index_min = int(np.argmin(fit_ys))
    y0 = fit_ys[index_min]
    x0 = fit_xs[index_min]

    near_minimum = np.abs(fit_xs - x0) < x_fit_threshold_um
    below_threshold = fit_ys < y_fit_threshold_ratio * y0
    turn_selector = near_minimum & below_threshold
    if np.count_nonzero(turn_selector) < 3:
        turn_selector = near_minimum
    if np.count_nonzero(turn_selector) < 3:
        raise ValueError("Need at least three points near the potential minimum.")

    parameters, covariance = curve_fit(
        quadratic_potential,
        fit_xs[turn_selector],
        fit_ys[turn_selector],
        p0=[x0, 0.2, y0],
    )
    frequency_mhz = trap_frequency_from_curvature_mhz_um2(parameters[1], mass_amu)
    return TrapFrequencyFit(frequency_mhz, parameters, covariance)


__all__ = [
    "TrapFrequencyFit",
    "fit_trap_frequency",
    "quadratic_potential",
    "trap_frequency_from_curvature_mhz_um2",
]
