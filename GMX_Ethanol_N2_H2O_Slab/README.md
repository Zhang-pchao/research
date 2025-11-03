# GMX_Ethanol_N2_H2O_Slab Directory Guide

This directory collects input files, preparation utilities, and post-processing scripts for the ethanol/nitrogen/water slab molecular dynamics workflow. The resources are grouped into production (`md/`) and analysis (`analysis/`) subtrees. The sections below describe every file to clarify its role in the simulation pipeline.

## analysis/

Scripts and job submission helpers for processing trajectories produced by the slab simulation.

### density/
- **density.py** – Computes one-dimensional density profiles along the *z*-axis for ethanol, nitrogen, water, and the combined system using MDAnalysis. The script supports both per-atom and center-of-mass binning, writes `density_data.csv`, and generates `density_plot.png` visualisations.
- **q.pbs** – Minimal PBS submission script that loads the MDAnalysis environment and runs `density.py` on a cluster node.

### hydrogen_bond/
- **ethanol_hb_total.py** – Runs MDAnalysis `HydrogenBondAnalysis` to characterise ethanol–water hydrogen bonding in the slab. The script manages topology parsing, atom selections, trajectory iteration, result binning along *z*, and outputs CSV/PNG summaries.
- **q.pbs** – PBS job script that prepares the Python environment (via Anaconda) before executing `ethanol_hb_total.py`.

## md/

Input assets and workflow automation for building and running the molecular dynamics simulation.

### 189ethanol_600n2_5500h2o/
- **workflow.pbs** – Primary PBS workflow orchestrating the multi-stage GROMACS run (file preparation, energy minimisation, equilibration, production). Includes environment setup, state tracking, file validation, and idempotent reruns.

#### gmx_files/
Topology and parameter files consumed by GROMACS.
- **mix.top** – System topology that includes the Amber14SB/SPC/E force field files plus custom ethanol and nitrogen molecule definitions, and declares molecule counts (189 ethanol, 600 N₂, 5500 water).
- **ethanol.itp** – GROMACS include file defining ethanol atom types, charges, masses, bonds, angles, and dihedrals.
- **N2.itp** – Two-site nitrogen molecule include file with bond parameters.
- **em.mdp** – Energy minimisation parameters (conjugate gradient integrator, 10,000 steps, PME electrostatics, dispersion corrections).
- **eq.mdp** – NVT equilibration parameters with temperature annealing from 0 K to 298.15 K, V-rescale thermostat, and hydrogen bond constraints.
- **prod.mdp** – Production MD parameters (30 ns with 1 fs timestep) under NVT conditions using the same thermostat and cut-off scheme as equilibration.

#### packmol/
Utilities for constructing the initial slab coordinates.
- **189C2H6O_600N2_5500H2O_1.pdb** – Packmol-generated mixed configuration containing two ethanol slabs, a nitrogen layer, and water reservoirs.
- **pack.inp.org** – Template Packmol input describing the spatial regions and molecule counts used to build the slab.
- **loop.sh** – Convenience shell loop that regenerates Packmol inputs/output for multiple water box identifiers by editing `pack.inp` and running Packmol.
- **pdb_modify_pbc.py** – Inserts a `CRYST1` record with box dimensions (40 × 40 × 159.33 Å³) into each PDB in the directory to define periodic boundaries.

## Repository-level README
For an overview of the entire research repository, refer to the root-level `README.md`.
