import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import curve_fit, fmin, fmin_tnc
from copy import deepcopy
import scipy.constants

from scipy.interpolate import interp1d
from math import floor, log10


def sqrt(x, a):
    return a * x ** (1 / 2)


def get_valerr_string(val, err):
    prec = floor(log10(err))
    err = round(err / 10**prec) * 10**prec
    val = round(val / 10**prec) * 10**prec
    if prec > 0:
        valerr = "{:.0f}({:.0f})".format(val, err)
    else:
        valerr = "{:.{prec}f}({:.0f})".format(val, err * 10**-prec, prec=-prec)
    return valerr


epsilon_0 = scipy.constants.epsilon_0
c = scipy.constants.c
a0 = scipy.constants.physical_constants["Bohr radius"][0]
k_B = scipy.constants.Boltzmann
h = scipy.constants.h
u = scipy.constants.u

m_rb = 87
m_cs = 133
m_rbcs = m_rb + m_cs

species = {
    "Rb": {
        "m": m_rb,
        "pols": {
            817: 4690,
            1065: 687,
            1066: 687,
        },
    },
    "Cs": {
        "m": m_cs,
        "pols": {
            817: -3253,
            1065: 1168,
            1066: 1168,
        },
    },
    "RbCs": {
        "m": m_rbcs,
        "pols": {
            1145: 754,
        },
    },
    "RbCs00": {
        "m": m_rbcs,
        "pols": {
            817: 443,
            1065: 1825,
            1066: 1825,
        },
    },
    "RbCs33": {
        "m": m_rbcs,
        "pols": {
            817: 443 - (1 / 3) * (-2816),
            1065: 1825 - (1 / 3) * 1981,
            1066: 1825 - (1 / 3) * 1981,
        },
    },
}


def waist(z, w0, wavelength):
    zR = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1 + (z / zR) ** 2)


def E_gaussian_3d_rotated(x, y, z, x0, y0, z0, theta, wx0, wy0, wz0, E0, wavelength):
    # Beam propagation coordinate along the beam axis
    s = (x - x0) * np.sin(theta) + (z - z0) * np.cos(theta)
    # Transverse coordinate in the x-z plane
    xp = (x - x0) * np.cos(theta) - (z - z0) * np.sin(theta)

    # Compute the beam waists as functions of s
    zR = np.pi * wz0**2 / wavelength
    wx = waist(s, wx0, wavelength)
    wy = waist(s, wy0, wavelength)
    wz = waist(s, wz0, wavelength)

    k = 2 * np.pi / (wavelength)
    Rs = np.maximum(s * (1 + (zR / np.maximum(s, 0.01)) ** 2), 0.01)
    return (
        E0
        * np.exp(-(xp**2) / wx**2)
        * np.exp(-((y - y0) ** 2) / wy**2)
        * (wz0 / wz)
        * np.exp(-1j * (k * (s)))
    )  # +k*((xp)**2/(2*(Rs)))))#-np.arctan(np.maximum(s,0.01)/zR)))


def total_potential(coord, tweezers, atom):
    x = coord[0]
    y = coord[1]
    z = coord[2]
    potential = 0
    for tag, params_arr in deepcopy(tweezers).items():
        wavelength = float(tag[0]) / 1e3
        E = 0 + 0j
        for params in params_arr:
            center_intensity_mW_um2 = (
                2 * params["power_mW"] / (np.pi * params["wx0"] * params["wy0"])
            )
            center_intensity_W_m2 = (center_intensity_mW_um2 / 1e3) * (1e6) ** 2
            center_E_field_V_per_m = np.sqrt(
                2 * center_intensity_W_m2 / (c * epsilon_0)
            )

            E += E_gaussian_3d_rotated(
                x,
                y,
                z,
                params["x0"],
                params["y0"],
                params["z0"],
                params["theta"],
                params["wx0"],
                params["wy0"],
                params["wz0"],
                center_E_field_V_per_m,
                wavelength,
            ) * np.exp(-1j * params["phase"])
        intensity = (1 / 2) * c * epsilon_0 * np.abs(E) ** 2

        polarisability_au = species[atom]["pols"][tag[0]]
        polarisability_si = polarisability_au * 4 * np.pi * epsilon_0 * a0**3

        trap_depth_J = (polarisability_si / (2 * c * epsilon_0)) * intensity
        trap_depth_MHz = trap_depth_J / h / 1e6
        potential += -trap_depth_MHz

    return potential


def quad(x, x0, a, y0):
    return a / 2 * (x - x0) ** 2 + y0


def my_find_potential_minimum(min_guess, tweezers, atom):
    # xtol = 0.000001
    # ftol = 0.000001
    minr, _, _ = fmin_tnc(
        total_potential,
        min_guess,
        args=(tweezers, atom),
        approx_grad=True,
        bounds=[(-5, 5), (-5, 5), (-9, 9)],
        epsilon=0.002,
        disp=False,
    )
    return minr


def my_get_trap_frequency(
    xs,
    potential,
    mass,
    x_fit_threshold_um=1,
    y_fit_threshold_ratio=0.8,
    x_bounds=(None, None),
    plot=False,
):
    """Fits a quadratic to the minimum of a potential to find a trap frequency."""
    mass_si = mass * u

    df = pd.DataFrame()
    selector = np.full_like(xs, True, dtype=bool)
    if x_bounds[0] is not None:
        selector = selector * (xs > x_bounds[0])
    if x_bounds[1] is not None:
        selector = selector * (xs < x_bounds[1])
    df["x"] = xs[selector]
    df["y"] = potential[selector]

    idxmin = df["y"].idxmin()
    y0 = df["y"][idxmin]
    x0 = df["x"][idxmin]

    df_turn = df[(np.abs(df["x"] - x0) < x_fit_threshold_um)]
    df_turn = df_turn[(df_turn["y"] < y_fit_threshold_ratio * y0)]

    # fix_quad = lambda x,a : quad(x,x0,a,y0)
    # print(df_turn)
    popt, pcov = curve_fit(
        quad, df_turn["x"], df_turn["y"], p0=[x0, 0.2, y0]
    )  # ,p0=[0])

    trap_frequency_MHz = (
        np.sqrt(popt[1] * 1e6 * h * (1e6) ** 2 / mass_si) / 1e6 / 2 / np.pi
    )  # k is in MHz/um**2

    if plot:
        plt.plot(xs, potential, c="blue", linestyle="-")
        ylim = plt.gca().get_ylim()
        plt.plot(xs, quad(xs, *popt), c="red", linestyle="--")
        plt.ylabel("potential (MHz)")
        plt.xlabel("distance (um)")
        plt.ylim(ylim)
        plt.show()

    return trap_frequency_MHz, popt
