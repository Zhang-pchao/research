## Dataset and model

The data sets for training the DPLR model and DW model are uploaded to [Zenodo](https://zenodo.org/records/14469805).

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
