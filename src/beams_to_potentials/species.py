"""Species definitions and polarizability lookup."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .constants import U


@dataclass(frozen=True)
class Species:
    """A trapped atom or molecule species.

    Parameters use the units currently used throughout this package:
    mass in atomic mass units and polarizabilities in atomic units keyed by
    wavelength in nm.
    """

    name: str
    mass_amu: float
    polarizabilities_au: Mapping[int, float]

    def __post_init__(self) -> None:
        normalized = {int(k): float(v) for k, v in self.polarizabilities_au.items()}
        object.__setattr__(self, "polarizabilities_au", MappingProxyType(normalized))

    @property
    def mass_si(self) -> float:
        return self.mass_amu * U

    def polarizability_au(self, wavelength_nm: int) -> float:
        try:
            return self.polarizabilities_au[int(wavelength_nm)]
        except KeyError as exc:
            raise KeyError(
                f"No polarizability for {self.name} at {int(wavelength_nm)} nm."
            ) from exc


M_RB = 87
M_CS = 133
M_RBCS = M_RB + M_CS

SPECIES: dict[str, Species] = {
    "Rb": Species(
        "Rb",
        M_RB,
        {
            817: 4690,
            1065: 687,
            1066: 687,
        },
    ),
    "Cs": Species(
        "Cs",
        M_CS,
        {
            817: -3253,
            1065: 1168,
            1066: 1168,
        },
    ),
    "RbCs": Species(
        "RbCs",
        M_RBCS,
        {
            1145: 754,
        },
    ),
    "RbCs00": Species(
        "RbCs00",
        M_RBCS,
        {
            817: 443,
            1065: 1825,
            1066: 1825,
        },
    ),
    # -N/(2N+3) is the prefactor of alpha_2. See lab book 19 May 2026,
    # from on-diagonal elements of H_AC.
    "RbCs33": Species(
        "RbCs33",
        M_RBCS,
        {
            817: 443 - (1 / 3) * (-2816),
            1065: 1825 - (1 / 3) * 1981,
            1066: 1825 - (1 / 3) * 1981,
        },
    ),
}


def get_species(species: str | Species) -> Species:
    if isinstance(species, Species):
        return species
    try:
        return SPECIES[species]
    except KeyError as exc:
        available = ", ".join(sorted(SPECIES))
        raise KeyError(f"Unknown species {species!r}. Available: {available}.") from exc


__all__ = ["M_CS", "M_RB", "M_RBCS", "SPECIES", "Species", "get_species"]
