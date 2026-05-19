beams-to-potentials
===================

Small toolkit for calculating optical potentials from Gaussian beams and
tracking trap minima/frequencies for atoms and molecules.

Basic usage:

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
    )
]

system = PotentialSystem(beams=beams, species="Cs")

x = np.linspace(-2, 2, 101)
potential = system.potential((x, 0, 0))
minimum = system.find_minimum((0, 0, 0))
frequencies = system.trap_frequencies(minimum)
```

Examples live in `examples/`:

```powershell
python examples\interactive_tweezer_overlap.py
python examples\tweezer_overlap_ratio_scan.py
python examples\two_tweezer_parameter_scan.py
python examples\tweezer_and_sal.py
python examples\tweezer_and_magic_lattice.py
```

Run tests with:

```powershell
python -m unittest
```
