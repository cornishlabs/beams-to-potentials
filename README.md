beams-to-potentials
===================

`beams-to-potentials` is a small Python library for turning Gaussian laser beam
parameters into optical potentials for atoms and molecules. It is intended for
quick lab calculations: make an interactive plot to understand a geometry, then
scan beam powers/positions/phases and extract useful quantities such as trap
depths, trap frequencies, and interparticle separations.

The library is deliberately simple. Most scripts should only need:

```python
from beams_to_potentials import Beam, PotentialSystem
```

What it does
------------

- Represents each laser beam with a `Beam` dataclass.
- Calculates scalar optical potentials in MHz for a chosen species.
- Finds local minima by numerical optimization.
- Fits harmonic trap frequencies along `x`, `y`, and `z`.
- Provides helpers for common parameter scans.
- Includes examples for interactive tweezer overlap, fixed-ratio scans, a
  two-tweezer parameter scan, a tweezer plus SAL, and an RbCs magic-lattice
  geometry.

Units
-----

The public API uses explicit unit suffixes in field names:

| Quantity | Unit |
| --- | --- |
| `wavelength_nm` | nm |
| `waist_x_um`, `waist_y_um`, `waist_z_um` | um |
| `center_um` | um |
| `power_mw` | mW |
| `theta_rad`, `phase_rad` | radians |
| `PotentialSystem.potential(...)` | MHz |
| `PotentialSystem.intensity_kw_cm2(...)` | kW/cm^2 |

Installation with uv
--------------------

This project currently requires Python `>=3.14`. If uv cannot find a suitable
Python version, install one with:

```powershell
uv python install 3.14
```

From a fresh checkout of this repository:

```powershell
cd path\to\beams-to-potentials
uv sync --dev
```

That creates or updates the local virtual environment and installs the package
in editable mode, along with the development tools.

Run a script through the environment with:

```powershell
uv run python examples\interactive_tweezer_overlap.py
```

Run tests with:

```powershell
uv run python -m unittest
```

Run linting/formatting checks with:

```powershell
uv run ruff check src examples tests
uv run ruff format --check src examples tests
```

If you want to use this library from another uv project before it is published,
add it as an editable local dependency:

```powershell
cd path\to\your-other-project
uv add --editable path\to\beams-to-potentials
```

Then import it as:

```python
import beams_to_potentials as bpl
```

If the package is later published, the local path in `uv add --editable ...`
can be replaced by the package name.

Basic usage
-----------

```python
import numpy as np

from beams_to_potentials import Beam, PotentialSystem

beams = [
    Beam(
        wavelength_nm=1065,
        waist_x_um=1.05,
        waist_y_um=1.16,
        waist_z_um=1.19,
        power_mw=4.4,
        center_um=(0.0, 0.0, 0.0),
    )
]

system = PotentialSystem(beams=beams, species="Cs")

x = np.linspace(-2, 2, 101)
potential = system.potential((x, 0, 0))
minimum = system.find_minimum((0, 0, 0))
frequencies = system.trap_frequencies(minimum)

print(minimum)
print(frequencies["x"].frequency_khz)
```

`potential((x, y, z))` accepts scalars or NumPy arrays, so the same call works
for a single point, a line cut, or a meshgrid.

Example workflow
----------------

1. Put your lab geometry in a small builder function.
2. Make an interactive plot to get a feel for the system.
3. Turn the same builder into a parameter scan.
4. Plot derived quantities: trap depth, frequency, position, intensity, or
   interparticle separation.

The overlap examples share their default setup in:

```powershell
examples\example_systems.py
```

That is a good place to edit beam waists, default powers, wavelengths, and
species choices for a particular experiment.

Examples
--------

Interactive two-tweezer overlap:

```powershell
uv run python examples\interactive_tweezer_overlap.py
```

Fixed power-ratio overlap scan:

```powershell
uv run python examples\tweezer_overlap_ratio_scan.py
```

Scan the long-wavelength tweezer position and track separation, depths, and
trap frequencies:

```powershell
uv run python examples\two_tweezer_parameter_scan.py
```

Other example geometries:

```powershell
uv run python examples\tweezer_and_sal.py
uv run python examples\tweezer_and_magic_lattice.py
```

Species and polarizabilities
----------------------------

Built-in species are defined in:

```powershell
src\beams_to_potentials\species.py
```

Add or update entries in `SPECIES` when you have new polarizability data. Each
species needs a mass in atomic mass units and a mapping from wavelength in nm to
polarizability in atomic units.

For example:

```python
"MySpecies": Species(
    "MySpecies",
    mass_amu=100,
    polarizabilities_au={
        817: 1234,
        1065: 567,
    },
)
```

Development notes
-----------------

The package code lives in `src\beams_to_potentials`. Examples live in
`examples`. Tests live in `tests`.

The intended public API is intentionally small:

- `Beam`
- `Species`
- `PotentialSystem`
- `analyze_trap`
- `scan_parameter`
- `scan_arrays`
- `distance_um` / `distance_nm`
- `fit_trap_frequency`

Lower-level formula helpers are kept in submodules so day-to-day lab scripts do
not need to know about them.
