# OPES-DPMD Bubble TiO2 Analysis Toolkit

This directory contains the in-house tools used to analyze OPES-DPMD simulations of nitrogen bubbles interacting with a TiO2 surface. The collection is organized into three top-level areas:

- Analysis_Scripts: Python utilities for frame-wise and post-processed analyses.
- DeePMD_Training: training configurations and helper scripts to generate Deep Potential models used by DPMD simulations.
- Enhanced_Sampling: OPES / enhanced-sampling inputs and helper scripts (see note below).

All Python tools rely on a scientific Python stack (NumPy, SciPy, Matplotlib, ASE, and MDAnalysis). Each analysis subdirectory typically contains a run.sh (or similar) template showing expected command-line arguments and example conda environment activation.

## Paper
[Nanobubble Nucleation and Dissolution Near the Anatase (101)–Water Interface](https://pubs.acs.org/articlesonrequest/AOR-BDZI3EDX8DYQMRPCX5I8)

```bibtex
@article{Zhang_JAmChemSoc_2026,
  title        = {{Nanobubble Nucleation and Dissolution Near the Anatase (101)–Water Interface}},
  author       = {Pengchao Zhang, Yawen Gao, Changsheng Chen, Xiangdang Guo, Chao Sun, Xuefei Xu},
  year         = 2026,
  journal      = {J. Am. Chem. Soc.},
  volume       = 148,
  number       = 17,
  pages        = {18507--18517},
  doi          = {10.1021/jacs.6c05480},
}
```

Contents and descriptions

## Analysis_Scripts/init_analysis

- analyze_surface_species/analyze_surface_species.py
  - Parses extended XYZ snapshots or LAMMPS data+trajectory pairs.
  - Identifies top-surface Ti atoms and classifies nearby adsorbates by counting O–H, H–Ti, H–H, O–O, and N–N contacts using KDTree searches with periodic boundary conditions.
  - Produces per-frame statistics for surface hydroxyls, protons, hydronium ions, and other species; writes diagnostic plots and logs.

- analyze_surface_species/run.sh
  - Minimal wrapper activating the analysis environment and invoking analyze_surface_species.py with LAMMPS inputs and an explicit atom_style argument.

- bubble_centroids/calculate_bubble_centroids.py
  - Loads MD trajectories with MDAnalysis, finds nitrogen atoms and clusters them using a union–find structure with periodic-distance criteria.
  - Computes bubble centroids using angular averaging to respect periodic boundaries; records time-resolved bubble positions and cluster sizes.
  - Optionally reads ion-coordinate files to ensure consistent simulation cell dimensions.

- bubble_centroids/run.sh
  - Example launch script that calculates bubble centroids for each stored trajectory frame (frame interval = 1) between specified start and end indices.

- find_ion_matrix/analyze_all_ion_species_matrix.py
  - Frame-by-frame classification of ionic species in solution and at the TiO2 surface using contact-based heuristics similar to the surface species analysis.
  - Produces time-resolved species matrices for building transition statistics and occupancy histograms.

- find_ion_matrix/run.sh
  - Wrapper script that runs the ion-matrix analysis for user-defined frame ranges and atom-style metadata.

- h2o_orientation/analyze_h2o_orientation.py
  - Reconstructs water molecules from O–H connectivity (with optional filtering of molecules above the TiO2 surface).
  - Computes per-molecule dipole vectors and their alignment relative to the surface normal; aggregates results into configurable z-bins.
  - Writes raw measurements, binned statistics, and pickled summaries for later plotting.

- h2o_orientation/run.sh
  - Demonstrates a full invocation including logging, z-bin size control, Ti–O cutoff selection, and output directory parameters.

## Analysis_Scripts/post_analysis

- bubble_centroids/batch_ion_analysis.py
  - Aggregates ion–bubble and ion–interface distance outputs across time-window subdirectories produced by the initial centroid analysis.
  - Computes distributions, ion-specific charge profiles, and exports publication-style plots.

- bubble_centroids/run.sh
  - Reference command to run the batch ion analysis over a simulation campaign and write figures to a central results directory.

- find_ion_matrix/transition_matrix_analysis.py
  - Reconstructs per-frame molecule identities from XYZ files and tracks transitions between bulk, surface hydroxide, and surface water states.
  - Builds count and probability transition matrices and includes checks to distinguish true proton hops from bulk re-association events; saves full statistics and figures.

- find_ion_matrix/run.sh
  - Example batch command pointing to a simulation archive and output directory for transition-matrix reports.

- h2o_orientation/plot_custom_orientation.py
  - Loads the pickled binned-orientation dataset produced during initial analysis and offers plotting modes (violin, box, mean±error, histogram) for cosθ or θ distributions.
  - Uses a Nature-style Matplotlib configuration and exports publication-ready figures.

## DeePMD_Training (detailed)

This folder contains the DeePMD-kit training configuration and helper scripts used to fit Deep Potential models for the DPMD simulations.

- run.json
  - Model and training highlights extracted from the file:
    - type_map: ["H","O","N","Na","Cl","Ti"].
    - Descriptor:
      - type: se_e2_a
      - sel (neighbor caps per species): [88, 44, 36, 12, 12, 38]
      - rcut_smth: 0.5 Å
      - rcut: 6.0 Å
      - neuron: [25, 50, 100]
      - activation_function: tanh
      - axis_neuron: 16
      - seed: 20250210
    - Fitting network:
      - type: ener
      - neuron: [240, 240, 240]
      - activation_function: tanh
      - resnet_dt: true
      - seed: 20250210
    - Loss settings:
      - start_pref_e: 0.02, limit_pref_e: 1.0
      - start_pref_f: 1000, limit_pref_f: 1.0
      - start_pref_v: 0.0, limit_pref_v: 0.0
    - Learning rate:
      - type: exp
      - start_lr: 1e-3, stop_lr: 3e-8
    - Training runtime:
      - numb_steps: 10,000,000
      - save_freq: 10,000
      - save_ckpt: model.ckpt
      - disp_freq: 1,000 (training curve output: lcurve.out)
      - profiling: false (profiling_file: timeline.json)
    - Training dataset:
      - A long list of systems and initial snapshots is referenced (bubble_ion/noNaCl and NaCl datasets, many init.* snapshots, and multiple tio2_dp datasets). These paths point to the on-disk training datasets used to build a combined model covering water, ions, nanobubbles, and TiO2-containing systems.
    - Batch selection:
      - batch_size: "auto"
      - auto_prob: "prob_sys_size; 0:16:0.41; 16:68:0.10; 68:92:0.24; 92:158:0.25"

  - Note: If you want a concise summary inserted in the README (e.g., network architecture, number of training steps, principal dataset groups), I included those key fields above. I can also extract and summarize any other run.json entries you want highlighted.

- q.sh
  - Example HPC / local training helper script:
    - Activates conda environment: source activate /your_envs_path/dpmdkit_v2.2.10
    - Runs training: dp train run.json --skip-neighbor-stat
    - Freezes the trained model: dp freeze
    - Compresses the model: dp compress
  - This script shows the intended training workflow and that dp (DeePMD-kit CLI) commands complete the standard train → freeze → compress sequence.

## Enhanced_Sampling

This directory groups input files and helper scripts for the OPES / enhanced-sampling part of the project: OPES / PLUMED inputs, collective-variable definitions, LAMMPS input templates, and post-processing helpers (CV analysis scripts, reweighting helpers, plotting scripts).

Note: I attempted to list and read the contents of OPES-DPMD-Bubble-TiO2/Enhanced_Sampling but received an access/"not found" response from the repository API. If you want per-file, line-level descriptions added to this README I need an accessible listing or the specific files. Please either:

- Confirm the correct path to Enhanced_Sampling (case-sensitive), or
- Give me read permission / make the files available in the repository, or
- Paste the file list (or the important files) here and I will incorporate precise descriptions.

## Usage notes and environment

- Python analyses expect a scientific Python environment with NumPy, SciPy, Matplotlib, ASE, and MDAnalysis.
- DeePMD-kit workflow (see q.sh): activate a conda env with dp installed (example env: dpmdkit_v2.2.10), run dp train on run.json, then dp freeze and dp compress to produce the final ready-to-use Deep Potential model.
- Many scripts include run.sh templates demonstrating example invocations and environment activation. Consult these for command-line options and typical runtime parameters.
