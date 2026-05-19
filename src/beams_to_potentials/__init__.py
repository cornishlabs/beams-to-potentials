"""Small public API for optical tweezer potential calculations.

Most scripts should only need ``Beam``, ``PotentialSystem``, and the analysis
helpers re-exported here. Lower-level implementation details live in the
submodules, but are intentionally not part of the top-level namespace.
"""

from .analysis import (
    AXES,
    AxisTrapAnalysis,
    ScanPoint,
    TrapAnalysis,
    analyze_trap,
    axis_slice,
    distance_nm,
    distance_um,
    harmonic_length_um,
    harmonic_oscillator_density_um,
    scan_arrays,
    scan_parameter,
    trap_frequencies,
)
from .beams import Beam, filter_beams_by_wavelength
from .fitting import (
    TrapFrequencyFit,
    fit_trap_frequency,
    quadratic_potential,
    trap_frequency_from_curvature_mhz_um2,
)
from .potentials import PotentialSystem
from .species import SPECIES, Species, get_species

__all__ = [
    "AXES",
    "SPECIES",
    "AxisTrapAnalysis",
    "Beam",
    "PotentialSystem",
    "ScanPoint",
    "Species",
    "TrapAnalysis",
    "TrapFrequencyFit",
    "analyze_trap",
    "axis_slice",
    "distance_nm",
    "distance_um",
    "filter_beams_by_wavelength",
    "fit_trap_frequency",
    "get_species",
    "harmonic_length_um",
    "harmonic_oscillator_density_um",
    "quadratic_potential",
    "scan_arrays",
    "scan_parameter",
    "trap_frequencies",
    "trap_frequency_from_curvature_mhz_um2",
]
