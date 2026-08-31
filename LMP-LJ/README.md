# LMP-LJ Simulations and Processed Cases

This directory collects reduced-unit Lennard-Jones examples and the processed
structures or inputs needed to interpret them. The two subdirectories under
`processed/` separate the confined oil-water-gas slab workflow from the
bubble-solid-particle interaction case:

```text
LMP-LJ/
├── processed/
│   ├── confined_oil_water_gas_slab/    # processed confined slab workflow
│   │   ├── GEO/2D-GEO.py
│   │   ├── NVT/{in.lammps,q.sh}
│   │   └── NPT/{in.lammps,q.sh}
│   └── bubble_solid_particle_interaction/ # processed GCMD case 91
│       ├── in2_91.min.txt
│       ├── bubble_solid_618nvt_91_first_frame.xyz
│       ├── Lennard-Jones_Potential_and_Reduced_Units.pdf
│       └── SHA256SUMS
└── README.md
```

Both directories contain processed or reusable reference artifacts; neither is
intended to be a complete raw-data mirror.

## `processed/confined_oil_water_gas_slab/`

This is the existing LAMMPS workflow for a fluid confined between two solid
plates with surfactant-like oil molecules and a gas phase. The systems use a
shared generated geometry and consistent force-field parameters.

### `processed/confined_oil_water_gas_slab/NVT/in.lammps`

This input performs an NVT run for the fluid while the solid substrates remain
fixed. It initializes velocities, applies type-specific Lennard-Jones and FENE
interactions, and writes thermodynamic and trajectory data for post-processing.

### `processed/confined_oil_water_gas_slab/NPT/in.lammps`

This input continues from the same configuration and applies a force to the top
solid plate to emulate pressure control while the fluid remains thermostatted.
The NVT and NPT stages share the same interaction definitions and groups.

Run an input from its own directory so that relative paths resolve correctly,
for example `lmp -in in.lammps`.

### `processed/confined_oil_water_gas_slab/GEO/2D-GEO.py`

This script generates the shared initial configuration (`geo.dat`/`geo.xyz`)
for the oil-gas-water slab. It builds the lattice, assigns atom types, creates
oil intramolecular bonds, and can write LAMMPS data and XYZ snapshots.

To regenerate the geometry, adjust the constants in `main()` and run Python 3
from `processed/confined_oil_water_gas_slab/GEO/`:

```bash
python 2D-GEO.py
```

## `processed/bubble_solid_particle_interaction/`

This folder is the processed initial-structure record for the SiO GCMD
bubble-solid example (case 91).

- `in2_91.min.txt` is a whitespace-normalized copy of the corresponding LAMMPS
  input. It documents the reduced Lennard-Jones parameters, groups, fixes,
  dump cadence, and run settings; commands and numerical values are unchanged.
- `bubble_solid_618nvt_91_first_frame.xyz` is the initial structure extracted
  from the first frame of `bubble_solid_618nvt_91.xyz`. It contains 230400 atoms
  and the frame comment `Atoms. Timestep: 0`.
- `Lennard-Jones_Potential_and_Reduced_Units.pdf` is the accompanying reference
  note for the reduced-unit potential and parameter example.
- `SHA256SUMS` records checksums for the three uploaded data/reference files.

The extracted XYZ contains atom type and Cartesian coordinates only. It does
not include velocities, periodic-box metadata, or the restart file referenced
by `in2_91.min.txt`; treat it as an initial-geometry artifact rather than a
standalone rerunnable package.

## Extraction provenance

- Source trajectory: `bubble_solid_618nvt_91.xyz`.
- Selection: lines 1-230402 inclusive, consisting of the XYZ atom count,
  comment line, and 230400 atom records.
- Source input: `in2_91.min.txt`; its XYZ dump cadence is 25000 steps and the
  selected frame is the timestep-0 frame.
- The full trajectory is intentionally not stored here because it is a large
  execution artifact; the compact first-frame structure is the public case
  record requested for data-oriented organization.
