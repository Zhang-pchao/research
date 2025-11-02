# OPES-DPMD Bubble TiO$_2$ Analysis Toolkit

This repository directory collects the in-house Python utilities that were used
to analyze OPES-DPMD simulations of nitrogen bubbles interacting with a
TiO$_2$ surface.  The scripts are grouped under
[`Analysis_Scripts`](Analysis_Scripts) and split into the two main stages of
our workflow:

* **Initial analysis** (`init_analysis/`): scripts that ingest raw LAMMPS
  trajectories or extended XYZ snapshots to extract chemical and structural
  descriptors for individual frames or for a subset of frames.
* **Post analysis** (`post_analysis/`): scripts that process the archived
  outputs of the initial analysis in order to build aggregate statistics or to
  generate publication-quality figures.

All Python tools rely on a scientific Python stack (NumPy, SciPy, Matplotlib,
ASE, and MDAnalysis).  Each subdirectory contains a `run.sh` template that
shows the expected command-line arguments and a typical `conda` environment
activation.

Below is a detailed description of every script.

## `Analysis_Scripts/init_analysis`

### `analyze_surface_species/analyze_surface_species.py`
* Parses either extended XYZ snapshots or pairs of LAMMPS data and trajectory
  files.
* Identifies top-surface Ti atoms and classifies nearby adsorbates by counting
  O–H, H–Ti, H–H, O–O, and N–N contacts via KDTree searches under periodic
  boundary conditions.
* Generates per-frame statistics of surface hydroxyls, protons, hydronium ions,
  and other species, and stores diagnostic plots and logs.

### `analyze_surface_species/run.sh`
* Minimal execution wrapper that activates the analysis environment and calls
  `analyze_surface_species.py` with the LAMMPS inputs and an explicit
  `atom_style` definition.

### `bubble_centroids/calculate_bubble_centroids.py`
* Loads MD trajectories with MDAnalysis, locates nitrogen atoms, and groups
  them into clusters using a union–find data structure with periodic-distance
  criteria.
* Computes bubble centroids with angular averaging to respect periodic boundary
  conditions, and records time-resolved bubble positions and cluster sizes.
* Can also read optional ion-coordinate files to ensure consistent simulation
  cell dimensions.

### `bubble_centroids/run.sh`
* Example launch script that calculates bubble centroids for every stored
  trajectory frame (frame interval 1) between the provided start and end
  indices.

### `find_ion_matrix/analyze_all_ion_species_matrix.py`
* Performs frame-by-frame classification of ionic species in the solution and
  at the TiO$_2$ surface using contact-based heuristics similar to the surface
  species analysis.
* Produces time-resolved species matrices that can later be used to construct
  transition statistics and occupancy histograms.

### `find_ion_matrix/run.sh`
* Wrapper that drives the ion matrix analysis with user-defined frame ranges
  and atom-style metadata.

### `h2o_orientation/analyze_h2o_orientation.py`
* Builds O–H connectivity to reconstruct individual water molecules and, when
  requested, filters to molecules above the TiO$_2$ surface.
* Calculates per-molecule dipole vectors, their alignment relative to the
  surface normal, and aggregates the results into configurable $z$-direction
  bins.
* Stores raw measurements, binned statistics, and pickled summaries for later
  visualization.

### `h2o_orientation/run.sh`
* Demonstrates a complete command with logging enabled, $z$-bin size control,
  Ti–O cutoff selection, and output directories for the orientation analysis.

## `Analysis_Scripts/post_analysis`

### `bubble_centroids/batch_ion_analysis.py`
* Walks through multiple time-window subdirectories that contain ion distance
  outputs from the initial centroid analysis.
* Aggregates raw ion–bubble and ion–interface distance files, computes
  distributions, ion-specific charge profiles, and produces Nature-style plots
  for each species and time range.

### `bubble_centroids/run.sh`
* Reference script for running the batch ion analysis on a chosen simulation
  campaign and writing the figure outputs into a central results folder.

### `find_ion_matrix/transition_matrix_analysis.py`
* Scans all time-resolved species folders produced by the ion-matrix analysis.
* Reconstructs per-frame molecule identities from XYZ files, tracks transitions
  between bulk, surface hydroxide, and surface water states, and builds both
  count and probability transition matrices.
* Includes specialized checks to distinguish genuine proton hops from bulk
  water re-association events and saves comprehensive statistics and figures.

### `find_ion_matrix/run.sh`
* Example batch command pointing to a full simulation archive and the desired
  output directory for the transition-matrix reports.

### `h2o_orientation/plot_custom_orientation.py`
* Loads the pickled binned-orientation dataset generated during the initial
  analysis and offers multiple plotting modes (violin, box, mean with error,
  histograms) for either $ \cos\theta$ or $ \theta$ distributions.
* Applies a Nature-style Matplotlib configuration and exports publication-ready
  figures to a user-selected directory.

