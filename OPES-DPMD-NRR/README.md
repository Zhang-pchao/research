# OPES-DPMD-NRR

## Related paper

**Solvent effect on the electrocatalytic nitrogen reduction reaction: a deep potential molecular dynamics simulation with enhanced sampling for the case of the ruthenium single atom catalyst**  
Journal of Materials Chemistry A (2026)  
DOI: [10.1039/D5TA09029F](https://doi.org/10.1039/D5TA09029F)

## Abstract (brief)

This work uses deep-potential molecular dynamics with enhanced sampling to quantify how solvent environments affect electrocatalytic nitrogen reduction on a Ru single-atom catalyst. The simulations reproduce the rate-determining-step energy barrier in agreement with experimental observations, supporting the reliability of the OPES + DPMD workflow for mechanistic analysis of NRR.

## Citation (BibTeX)

```bibtex
@Article{Zhang_JMaterChemA_2026_v14_p7109,
    author =   {Bowen Zhang and Pengchao Zhang and Xuefei Xu},
    title =    {{Solvent effect on the electrocatalytic nitrogen reduction reaction: a
             deep potential molecular dynamics simulation with enhanced sampling
             for the case of the ruthenium single atom catalyst}},
    journal =  {J. Mater. Chem. A},
    year =     2026,
    volume =   14,
    number =   12,
    pages =    {7109--7120},
    doi =      {10.1039/D5TA09029F}
}
```

## Workflow overview

This directory collects the input decks, scripts, and reference data that support the neural network reactive dynamics workflow for nitrogen reduction using OPES and DeePMD-based force fields. The material is organized by stage of the pipeline.

## Directory Overview

### `DFT_Calculation/`
Contains the first-principles reference calculations used to seed or validate the machine-learning potential.

- `cp2k_input/input.inp` – CP2K input file that performs single-point energy and force evaluations (`RUN_TYPE ENERGY_FORCE`) with a Quickstep DFT setup including UKS multiplicity, 800 Ry plane-wave cutoff, and Fermi–Dirac smearing at 300 K. The file defines the electronic structure parameters, SCF convergence settings, and exchange–correlation treatment for the Ru-based catalytic model.
- `cp2k_input/coord.xyz` – Atomic coordinates for the simulated system, consumed by the CP2K workflow.

### `DP-GEN_Iteration/`
Holds the DP-GEN configuration for iterative active learning.

- `run.json` – Workflow configuration that enumerates the chemical species (`type_map`/`mass_map`), lists initial datasets and structure seeds, and specifies the DeepMD model hyperparameters (descriptor selections, fitting network layout, learning schedule, and model-deviation thresholds).
- `run_machine.json` – Execution profile describing the HPC environments for training, model deviation evaluation, and ab initio labeling. It includes scheduler types (Slurm/PBS), resource requests, and the commands for launching DeepMD (`dp`), LAMMPS (`lmp`), and CP2K labeling jobs.

### `DeePMD_Training/`
Stores standalone DeePMD training artifacts.

- `run.json` – Training recipe mirroring the production DP-GEN parameters for building a single-model potential from curated datasets.
- `partial_dataset/` – Example dataset shard (`data.010`) referenced by the training input.
- `frozen_model.pb` – Serialized DeepMD model used for downstream molecular dynamics.

### `Enhanced_Sampling/`
Resources for running OPES-enhanced molecular dynamics with the trained potential.

- `OPES_MD/input.lammps` – LAMMPS input script that loads the DeepMD model, sets up a 300 K NVT ensemble, and couples to PLUMED via the OPES bias (`fix dpgen_plm`).
- `OPES_MD/input.plumed` – PLUMED 2 control file defining Voronoi-based collective variables (`VORONOIS1`, `VORONOID1/2/3`), the OPES_METAD_EXPLORE bias, wall restraints, and COLVAR output.
- `OPES_MD/conf.lmp` – Initial atomic configuration compatible with the LAMMPS run.

### `Voronoi_collective_variables/`
Custom PLUMED collective-variable implementation.

- `VoronoiD3.cpp` – Source code extending PLUMED with the `VORONOID3` collective variable, including neighbor-list acceleration options and lambda-parameterized switching functions for counting reactive Voronoi environments.

## Software Environment

The workflow references the following software stack:

- **DeepMD-kit**: environment `dpmdkit_v2.1.5_0202`, providing the `dp` trainer and `python3.10` interpreter used in DP-GEN iterations (see `DP-GEN_Iteration/run_machine.json`).
- **LAMMPS**: invoked as `lmp` for model deviation screening and as the molecular dynamics engine for OPES runs (specified in `DP-GEN_Iteration/run_machine.json` and `Enhanced_Sampling/OPES_MD/input.lammps`).
- **CP2K**: module `cp2k/2023.1_plm` with toolchain setup for ab initio labeling tasks (listed in `DP-GEN_Iteration/run_machine.json`).
- **PLUMED**: custom collective variables are implemented against PLUMED 2 and used to couple with LAMMPS for OPES sampling (as indicated by `Voronoi_collective_variables/VoronoiD3.cpp` and `Enhanced_Sampling/OPES_MD/input.plumed`).

These inputs can be adapted to target infrastructures by updating filesystem paths, scheduler directives, and module names in the JSON descriptors.

