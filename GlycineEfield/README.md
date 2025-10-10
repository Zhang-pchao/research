## Overview

This repository accompanies the study on *Modulation of Electric Field and Interface on Competitive Reaction Mechanisms*. It gathers the input files, scripts, and trained models used to generate the results reported in the paper listed below. Use the directory map to quickly locate simulation inputs, training data, and analysis utilities.

## Dataset and models

The data sets for training the DPLR model and DW model are uploaded to [Zenodo](https://zenodo.org/records/14469805).

## Repository layout

- **`Analysis_Scripts/`** – Post-processing utilities used throughout the project.
  - `IR/collect_dipole/` contains the raw trajectory (`ZwiGlycine128H2O_opt_full4mda.data`) together with scripts (`lmptrj2dipole.py`, `lmptrj2dipole_vcv.py`) to extract dipole moments, while `IR/dipole2ir/` provides `dipole2ir.py` for converting dipole time series into infrared spectra.
  - `dp_data/` houses tools for preparing DeePMD datasets, including format conversions between CP2K, LAMMPS, and DeePMD (`sp_cp2k2dp_newmap.py`, `sp_lmp2dp_newmap.py`, `lammpstrj2xyz.py`, etc.), utilities for splitting and converting datasets (`split_npy.py`, `dpdata2exyz.py`, `xyz2dpdata.py`), and reference DFT folders (e.g., `dft/pbe2m062x/`).
  - `fes_cv/` groups scripts and submission helpers for free-energy surface reconstruction and collective variable handling (`find_cat_colvar.py`, `sub_cvfig.sh`) alongside example outputs (`fes1D/`, `fes2D/`, `cvfig/`, `interface_cv/`).
  - `lammps_data/` offers data-conversion helpers for LAMMPS simulations, such as `atomic2full_data4dplr.py`, `convert_xyz_to_lmp_data.py`, and restart-file sorting scripts.
- **`DeePMD_LongRange_Training/`** – DeePMD-kit inputs for the long-range potential, including `run.json`, training diagnostics (`lcurve.out`), the submission script `q.sh`, and the exported model `frozen_model.pb`.
- **`DeePWannier_Model_Training/`** – DeePWannier training configuration (`run.json`), learning-curve output (`lcurve.out`), and the associated job script (`q.sh`).
- **`Enhanced_Sampling_MD/`** – PLUMED-enhanced sampling setups. `LongRange_Bulk_Efield/` and `LongRange_Interface/` each provide LAMMPS inputs (`in.lmp`), PLUMED control files (`input.plumed`), job scripts, and equilibrated data files (`*.data`).
- **`IR_MD/`** – Input deck for IR spectrum molecular dynamics simulations, including the system topology (`ZwiGlycine128H2O_opt_full.data`), the LAMMPS input (`in.glycine`), and submission script (`q.sh`).

## Packages Used

### 1. plumed_v2.8.1_patch
Requiring the installation of the [OPES](https://www.plumed.org/doc-v2.8/user-doc/html/_o_p_e_s.html) module.

To use Voronoi CVs code, put the three .cpp files above into /your_plumed_path/plumed/src/colvar, and then compile plumed.

The Voronoi CV VORONOID2.cpp, VORONOIS1.cpp [code files](https://github.com/Zhang-pchao/GlycineTautomerism/tree/main/Voronoi_collective_variables) are linked to CVs named s_d, s_p as illustrated in paper.

Other Voronoi CVs can be used to calculate the [diffusion coefficient](https://github.com/Zhang-pchao/OilWaterInterface/tree/main/Ion_Diffusion_Coefficient) for H₃O⁺ or OH⁻ ions, and the [water autoionization](https://github.com/Zhang-pchao/OilWaterInterface/tree/main) process.

### 2. deepmd-kit_v2.2.6

- **Re-compile PLUMED**
  Incorporate LAMMPS and PLUMED by following the [plumed-feedstock](https://github.com/Zhang-pchao/plumed-feedstock/tree/devel) recipe to overlay the default PLUMED version.

- **No Re-compile (quick test)**
  If you do **not** want to re-compile PLUMED, use the [LOAD](https://www.plumed.org/doc-v2.8/user-doc/html/_l_o_a_d.html) command at runtime.

### 3. deepks-kit_v0.1

### 4. abacus_v3.0.5

### 5. cp2k_v9.1
Incorporating plumed

## Paper

Modulation of electric field and interface on competitive reaction mechanisms. [ACS Articles on Request link](https://pubs.acs.org/articlesonrequest/AOR-NMVU6VAHH7GKQHZNMPMC) [JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00705)

```bibtex
@article{Zhang_JChemTheoryComput_2025_v21_p6584,
  title        = {{Modulation of Electric Field and Interface on Competitive Reaction Mechanisms}},
  author       = {Pengchao Zhang and Xuefei Xu},
  year         = 2025,
  journal      = {J. Chem. Theory Comput.},
  volume       = 21,
  number       = 13,
  pages        = {6584--6593},
  doi          = {10.1021/acs.jctc.5c00705},
}
