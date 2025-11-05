# LMP-LJ Simulations

This directory contains LAMMPS input scripts for Lennard-Jones simulations of a fluid confined between two solid plates with surfactant-like oil molecules and a gas phase. The systems are prepared from the shared `geo.dat` structure and make use of consistent force-field parameters and group definitions.

## `NVT/in.lammps`

This script performs an NVT (constant number of particles, volume, and temperature) run for the fluid components while the solid substrates remain fixed. It is typically used for equilibration at a target reduced temperature of 0.846 (≈101.3 K) before pressure coupling. Key features include:

- Lennard-Jones interactions with type-specific parameters for substrate, water, gas, and oil beads.
- FENE bonds for polymeric oil molecules.
- Velocity initialization for the fluid at the desired temperature.
- Thermodynamic output every 1000 steps and trajectory dumps for post-processing.

## `NPT/in.lammps`

This script continues from the same configuration but applies a force on the top solid plate to emulate an NPT-like ensemble while the fluid remains thermostatted at the same temperature. It is suited for studying the system under pressure control. Highlights are:

- Identical interaction parameters and group definitions to maintain consistency with the NVT run.
- Computation of the force balance between the fluid and the top substrate to control the applied pressure.
- Restart outputs every 500,000 steps and long production runs for sampling interfacial properties.

Run each input with the corresponding directory as the working directory so that relative paths resolve correctly (e.g., `lmp -in in.lammps`).

## `GEO/2D-GEO.py`

`GEO/2D-GEO.py` generates the shared initial configuration (`geo.dat`/`geo.xyz`) for the oil–gas–water slab simulations. The script builds a three-dimensional lattice, assigns atom types for the substrate, water, gas bubble, and oil beads, and creates intramolecular bonds within the oil phase. Key capabilities include:

- Parameterized grid dimensions and phase positioning to customize the slab geometry.
- Region-based atom typing with defaults that reproduce the published configuration.
- Optional bond creation rules that link adjacent oil beads across the mixed phase.
- Writers for both LAMMPS data files and XYZ snapshots, plus summary statistics for verification.

To regenerate the geometry, adjust the constants in `main()` as needed and run the script with Python 3 from within `LMP-LJ/GEO/`:

```bash
python 2D-GEO.py
```
