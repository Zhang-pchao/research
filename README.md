# Research Reproducibility Archive

This repository curates the computational assets underlying recent publications on aqueous interfaces, biomolecular tautomerism, and electric-field-driven chemistry. The goal is to make **simulation inputs, trained models, and analysis workflows** permanently accessible so that other researchers can reproduce and extend the studies with minimal friction.

---

## Published Works

Each entry links to the original article and the companion GitHub repository where input/output files, scripts, and post-processing notebooks are maintained.

| No. | Title & Link | GitHub | Year | Journal |
|:---:|:-------------|:--------|:----:|:--------|
| 1 | [Double-layer distribution of hydronium and hydroxide ions in the air-water interface](https://pubs.acs.org/doi/10.1021/acsphyschemau.3c00076) | [🔗 GitHub](https://github.com/Zhang-pchao/DoubleLayerAirWater) | 2024 | *ACS Phys. Chem Au* |
| 2 | [Intramolecular and water mediated tautomerism of solvated glycine](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00273) | [🔗 GitHub](https://github.com/Zhang-pchao/GlycineTautomerism) | 2024 | *J. Chem. Inf. Model.* |
| 3 | [Hydroxide and hydronium ions modulate the dynamic evolution of nitrogen nanobubbles in water](https://pubs.acs.org/doi/10.1021/jacs.4c06641) | [🔗 GitHub](https://github.com/Zhang-pchao/N2BubbleIon) | 2024 | *J. Am. Chem. Soc.* |
| 4 | [Propensity of water self-ions at air(oil)-water interfaces revealed by deep potential molecular dynamics with enhanced sampling](https://pubs.acs.org/doi/full/10.1021/acs.langmuir.4c05004) | [🔗 GitHub](https://github.com/Zhang-pchao/OilWaterInterface) | 2025 | *Langmuir* |
| 5 | [Modulation of electric field and interface on competitive reaction mechanisms](https://doi.org/10.1021/acs.jctc.5c00705) | [🔗 GitHub](https://github.com/Zhang-pchao/research/tree/main/GlycineEfield) | 2025 | *J. Chem. Theory Comput.* |

---

## How to Use This Archive

1. **Select a publication** from the table above to identify the scientific context and journal venue.
2. **Follow the GitHub link** to obtain the complete simulation package, including input geometries, PLUMED configuration files, Deep Potential/Deep Wannier training sets, and analysis notebooks prepared by the authors.
3. **Consult companion datasets** where provided. For example, the `GlycineEfield` entry in this repository points to Zenodo-hosted training data for the Deep Potential Long-Range (DPLR) and Deep Wannier (DW) models associated with the JCTC study.
4. **Replicate or extend** the calculations by adapting the published workflows to your systems of interest. The repositories are organized so that new systems, sampling strategies, or machine-learning force fields can be integrated with minimal restructuring.

## Publication Capsules

### 1. Double-layer distribution of hydronium and hydroxide ions in the air-water interface (2024, *ACS Phys. Chem Au*)
- **Research theme:** Characterizes how self-ions partition at the air–water boundary and influence double-layer structure.
- **Shared resources:** The linked GitHub repository provides the enhanced-sampling inputs, model checkpoints, and post-processing scripts required to reproduce the reported double-layer profiles.
- **Extending the work:** Replace the interfacial composition or add co-solutes within the same workflow to explore how environmental factors reshape the double layer.

### 2. Intramolecular and water mediated tautomerism of solvated glycine (2024, *J. Chem. Inf. Model.*)
- **Research theme:** Maps the competition between neutral and zwitterionic glycine tautomers under explicit-solvent conditions.
- **Shared resources:** Simulation inputs and Voronoi collective variables (CVs) for enhanced sampling are hosted in the companion GitHub repository, enabling direct reuse in new amino-acid or peptide studies.
- **Extending the work:** Substitute different biomolecules or introduce external fields to probe tautomer populations using the supplied CVs and sampling recipes.

### 3. Hydroxide and hydronium ions modulate the dynamic evolution of nitrogen nanobubbles in water (2024, *J. Am. Chem. Soc.*)
- **Research theme:** Investigates how water self-ions regulate nanobubble stabilization in aqueous media.
- **Shared resources:** Input decks for reactive molecular dynamics, bubble-tracking analysis utilities, and figure-generation scripts are archived in the associated GitHub repository.
- **Extending the work:** Apply the same pipelines to different dissolved gases or ion concentrations to interrogate interfacial gas dynamics.

### 4. Propensity of water self-ions at air(oil)-water interfaces revealed by deep potential molecular dynamics with enhanced sampling (2025, *Langmuir*)
- **Research theme:** Quantifies the positioning and kinetics of hydronium and hydroxide ions at complex air–oil–water interfaces via machine-learning potentials.
- **Shared resources:** The GitHub repository hosts Deep Potential training inputs, OPES-based enhanced-sampling configurations, and diffusion-analysis scripts tailored to interfacial systems.
- **Extending the work:** Swap the organic phase, adjust ionic strength, or couple to additional order parameters to test new hypotheses about interfacial ion behavior.

### 5. Modulation of electric field and interface on competitive reaction mechanisms (2025, *J. Chem. Theory Comput.*)
- **Research theme:** Dissects how interfacial electric fields bias competing reaction channels.
- **Shared resources:** This repository contains the `GlycineEfield` package with analysis scripts, enhanced-sampling molecular dynamics setups, infrared trajectory inputs, and Deep Potential/Deep Wannier training workflows. Zenodo hosts the large DPLR and DW datasets referenced therein.
- **Extending the work:** Introduce alternative reactants, field strengths, or interfacial compositions by editing the provided PLUMED, LAMMPS, and CP2K inputs while retraining the ML potentials as needed.

## Extending the Archive

- **Adding new publications:** Duplicate the table row format, include direct links to the article and resource repository, and summarize the study using the capsule template above.
- **Documenting datasets:** Note any external hosting services (e.g., Zenodo) alongside in-repository folders so future users can locate raw data and model checkpoints.
- **Maintaining workflows:** Keep simulation scripts and analysis notebooks synchronized with the published results; version tags or release notes in the companion repositories help track updates.
- **Community feedback:** Issues and pull requests are welcome for clarifications, missing files, or suggestions on how to broaden the reusable components of each study.
