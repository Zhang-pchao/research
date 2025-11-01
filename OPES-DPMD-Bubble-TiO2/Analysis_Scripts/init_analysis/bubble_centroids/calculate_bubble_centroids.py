#!/usr/bin/env python3
"""
Script for calculating nitrogen bubble centroids.
Combines MDAnalysis LAMMPS trajectory reading with union-find clustering analysis.

Update notes (PBC support):
- Supports reading extended xyz ion files (including PBC information)
- Automatically parses lattice="a 0.0 0.0 0.0 b 0.0 0.0 0.0 c" formatted box dimensions
- Prioritizes box dimensions read from ion files when calculating ion-bubble distances
- Ensures all distance calculations properly account for periodic boundary conditions
"""

import os
import sys
import numpy as np
import argparse
import time
import re
from collections import defaultdict
import logging

# Import MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

class UnionFind:
    """Union-find data structure for clustering analysis"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

class BubbleCentroidCalculator:
    """Bubble centroid calculator"""
    
    def __init__(self, cutoff_distance=5.5):
        self.cutoff_distance = cutoff_distance
        
        # Mapping from LAMMPS atom types to chemical elements
        self.type_to_element = {
            '1': 'H',   
            '2': 'O',     
            '3': 'N',  
            '4': 'Na',   
            '5': 'Cl',
            '6': 'Ti',
        }
        
        # Determine the type ID for nitrogen atoms
        self.nitrogen_type = None
        for atom_type, element in self.type_to_element.items():
            if element == 'N':
                self.nitrogen_type = int(atom_type)
                break
        
        # Initialize ion box data storage
        self.ion_box_data = {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_progress(self, message, flush=True):
        """Print progress messages with timestamps"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        if flush:
            sys.stdout.flush()
    
    def periodic_distance(self, coord1, coord2, box_dims):
        """Calculate distance with periodic boundary conditions"""
        diff = coord1 - coord2
        for i in range(3):
            box_length = box_dims[i]
            diff[i] = diff[i] - box_length * round(diff[i] / box_length)
        return np.linalg.norm(diff)
    
    def cluster_nitrogen_atoms(self, n_coords, box_dims):
        """Perform nitrogen atom clustering with union-find"""
        if len(n_coords) == 0:
            return [], []
        
        n_points = len(n_coords)
        uf = UnionFind(n_points)
        
        self.logger.info(f"Starting clustering analysis for {n_points} nitrogen atoms...")
        
        # Build adjacency relationships
        for i in range(n_points):
            for j in range(i+1, n_points):
                if self.periodic_distance(n_coords[i], n_coords[j], box_dims) <= self.cutoff_distance:
                    uf.union(i, j)
        
        # Collect cluster results
        clusters_dict = defaultdict(list)
        for atom in range(n_points):
            clusters_dict[uf.find(atom)].append(atom)
        
        # Convert to list format and sort by size
        clusters = sorted(clusters_dict.values(), key=len, reverse=True)
        
        self.logger.info(f"Found {len(clusters)} nitrogen clusters")
        if clusters:
            self.logger.info(f"Largest cluster contains {len(clusters[0])} nitrogen atoms")
        
        return clusters
    
    def calculate_centroid_pbc(self, coords, box_dims):
        """Calculate centroid with periodic boundary conditions"""
        centroid = np.zeros(3)
        
        for dim in range(3):
            box_length = box_dims[dim]
            
            # Convert to angular coordinates
            angles = 2 * np.pi * coords[:, dim] / box_length
            
            # Compute average angle
            cos_mean = np.mean(np.cos(angles))
            sin_mean = np.mean(np.sin(angles))
            mean_angle = np.arctan2(sin_mean, cos_mean)
            
            # Convert back to Cartesian coordinates
            centroid[dim] = (mean_angle * box_length) / (2 * np.pi)
            if centroid[dim] < 0:
                centroid[dim] += box_length
        
        return centroid
    
    def read_lammps_with_mda(self, data_file, traj_file, atom_style=None):
        """Read LAMMPS files with MDAnalysis"""
        try:
            self.logger.info("Reading LAMMPS files with MDAnalysis...")
            self.logger.info(f"Data file: {data_file}")
            self.logger.info(f"Trajectory file: {traj_file}")

            # Use the user-specified atom_style when provided
            if atom_style:
                self.logger.info(f"Using user-specified atom_style='{atom_style}'")
                u = mda.Universe(data_file, traj_file,
                               atom_style=atom_style,
                               format='LAMMPSDUMP')
            else:
                # Attempt multiple approaches to read the LAMMPS files
                try:
                    self.logger.info("Trying atom_style='id type x y z'")
                    u = mda.Universe(data_file, traj_file,
                                   atom_style='id type x y z',
                                   format='LAMMPSDUMP')
                except Exception as e1:
                    self.logger.warning(f"Approach 1 failed: {e1}")
                    try:
                        self.logger.info("Trying atom_style='atomic'")
                        u = mda.Universe(data_file, traj_file,
                                       atom_style='atomic',
                                       format='LAMMPSDUMP')
                    except Exception as e2:
                        self.logger.warning(f"Approach 2 failed: {e2}")
                        self.logger.info("Trying atom_style='full'")
                        u = mda.Universe(data_file, traj_file,
                                       atom_style='full',
                                       format='LAMMPSDUMP')

            self.logger.info(f"Successfully read {len(u.atoms)} atoms")
            self.logger.info(f"Trajectory contains {len(u.trajectory)} frames")

            return u

        except Exception as e:
            self.logger.error(f"Error reading LAMMPS files with MDAnalysis: {str(e)}")
            raise
    
    def process_trajectory(self, data_file, traj_file, atom_style=None, output_file="bubble_centroids.txt",
                         step_interval=1, start_frame=0, end_frame=-1, ion_files=None, ions_analysis_output=None):
        """Process the trajectory and calculate bubble centroids with optional ion analysis"""

        # Ensure the nitrogen atom type was located
        if self.nitrogen_type is None:
            raise ValueError("Nitrogen atom type not found in type_to_element mapping")

        self.logger.info(f"Nitrogen atom type from type_to_element mapping: {self.nitrogen_type}")

        # Load the trajectory
        u = self.read_lammps_with_mda(data_file, traj_file, atom_style)

        # Determine frames to analyze
        total_frames = len(u.trajectory)
        actual_end_frame = total_frames if end_frame == -1 else min(end_frame, total_frames)
        frames_to_analyze = list(range(start_frame, actual_end_frame, step_interval))

        self.logger.info(f"Total trajectory frames: {total_frames}")
        self.logger.info(f"Frame range analyzed: {start_frame} - {actual_end_frame-1}")
        self.logger.info(f"Frame interval: {step_interval}")
        self.logger.info(f"Frames to analyze: {len(frames_to_analyze)}")

        if not frames_to_analyze:
            raise ValueError("No frames to analyze; please check the frame range settings")

        # Read ion data if provided
        ion_frames_data = {}
        if ion_files:
            self.logger.info("Reading ion files with extended xyz PBC information...")
            ion_configs = {
                'H3O': {'file': ion_files.get('h3o'), 'atoms_per_molecule': 4},
                'bulk_OH': {'file': ion_files.get('bulk_oh'), 'atoms_per_molecule': 2},
                'surface_OH': {'file': ion_files.get('surface_oh'), 'atoms_per_molecule': 2},
                'surface_H': {'file': ion_files.get('surface_h'), 'atoms_per_molecule': 1},
                'Na': {'file': ion_files.get('na'), 'atoms_per_molecule': 1},
                'Cl': {'file': ion_files.get('cl'), 'atoms_per_molecule': 1}
            }

            for ion_name, config in ion_configs.items():
                if config['file']:
                    ion_frames_data[ion_name] = self.read_xyz_file_with_frame_filter(
                        config['file'], set(frames_to_analyze), config['atoms_per_molecule'], ion_name)

            # Report box information statistics
            if hasattr(self, 'ion_box_data') and self.ion_box_data:
                self.logger.info(f"Parsed box dimensions from {len(self.ion_box_data)} frames in ion files")
                sample_frame = next(iter(self.ion_box_data))
                sample_box = self.ion_box_data[sample_frame]
                self.logger.info(f"Sample box dimensions (frame {sample_frame}): {sample_box}")
            else:
                self.logger.warning("No box dimensions parsed from ion files; using LAMMPS trajectory box dimensions")

        # Prepare output data
        centroids_data = []
        times = []
        bubble_sizes = []
        frame_numbers = []

        # Ion analysis data
        ions_distance_data = {ion_name: [] for ion_name in ion_frames_data.keys()}

        self.logger.info("Processing trajectory frames...")

        # Iterate over selected frames
        for i, frame_idx in enumerate(frames_to_analyze):
            # Jump to the specified frame
            u.trajectory[frame_idx]
            ts = u.trajectory.ts

            self.logger.info(f"Processing selected frame {i+1}/{len(frames_to_analyze)} (frame index: {frame_idx}, time: {ts.time})")

            # Retrieve box dimensions
            box_dims = u.dimensions[:3]  # Use only the first three dimensions (x, y, z)

            # Select nitrogen atoms
            nitrogen_atoms = u.select_atoms(f"type {self.nitrogen_type}")

            if len(nitrogen_atoms) == 0:
                self.logger.warning(f"No nitrogen atoms found in frame {frame_idx}")
                continue

            # Extract nitrogen coordinates
            n_coords = nitrogen_atoms.positions

            # Perform clustering analysis
            clusters = self.cluster_nitrogen_atoms(n_coords, box_dims)

            if not clusters:
                self.logger.warning(f"No nitrogen clusters found in frame {frame_idx}")
                continue

            # Identify the largest nitrogen bubble
            largest_cluster_indices = clusters[0]  # Already sorted by size
            largest_cluster_coords = n_coords[largest_cluster_indices]

            # Calculate centroid of the largest bubble
            centroid = self.calculate_centroid_pbc(largest_cluster_coords, box_dims)

            # Record data
            times.append(ts.time)
            centroids_data.append(centroid)
            bubble_sizes.append(len(largest_cluster_indices))
            frame_numbers.append(frame_idx)

            self.logger.info(f"Frame {frame_idx}: largest bubble contains {len(largest_cluster_indices)} nitrogen atoms")
            self.logger.info(f"Centroid coordinates: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")

            # Ion analysis when ion data are available
            if ion_frames_data:
                # Identify bubble surface atoms
                surface_coords = self.find_bubble_surface_n2_atoms(largest_cluster_coords, box_dims)

                # Analyze each ion type
                for ion_name, ion_frames in ion_frames_data.items():
                    if frame_idx in ion_frames:
                        molecules = ion_frames[frame_idx]
                        if molecules:
                            # Compute ion distances, passing frame_idx to use appropriate box dimensions
                            distances_data = self.calculate_ion_bubble_distances(
                                molecules, centroid, surface_coords, box_dims, ion_name, frame_idx)

                            ions_distance_data[ion_name].extend(distances_data)

                            self.logger.info(f"Frame {frame_idx}: analyzed {len(molecules)} {ion_name} molecules/ions")

        # Save bubble centroid results
        self.save_results(times, centroids_data, bubble_sizes, frame_numbers, output_file,
                         step_interval, start_frame, actual_end_frame)

        # Save raw ion distance data when available
        if any(ions_distance_data.values()) and ions_analysis_output:
            self.save_raw_ion_distances(ions_distance_data, ions_analysis_output)

        # Analyze and plot ion distributions when data exist
        if any(ions_distance_data.values()) and ions_analysis_output:
            total_ions = sum(len(distances) for distances in ions_distance_data.values())
            self.logger.info(f"Starting ion distribution analysis with {total_ions} data points")
            self.plot_all_ions_distance_distributions(ions_distance_data, ions_analysis_output)
        elif ion_files and not any(ions_distance_data.values()):
            self.logger.warning("Ion files were provided but no valid ion data were found")

        return times, centroids_data, bubble_sizes
    
    def save_results(self, times, centroids_data, bubble_sizes, frame_numbers, output_file,
                   step_interval, start_frame, end_frame):
        """Save computed results"""

        # Save centroid coordinates
        centroids_file = output_file
        with open(centroids_file, 'w') as f:
            f.write("# FrameIndex Time(ps) X(Å) Y(Å) Z(Å) BubbleSize\n")
            for frame_idx, time, centroid, size in zip(frame_numbers, times, centroids_data, bubble_sizes):
                f.write(f"{frame_idx} {time:.1f} {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} {size}\n")

        self.logger.info(f"Centroid coordinates saved to: {centroids_file}")

        # Save statistics
        stats_file = output_file.replace('.txt', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("# Bubble centroid calculation statistics\n")
            f.write(f"# Number of analyzed frames: {len(times)}\n")
            f.write(f"# Frame selection settings:\n")
            f.write(f"#   Start frame: {start_frame}\n")
            f.write(f"#   End frame: {end_frame}\n")
            f.write(f"#   Frame interval: {step_interval}\n")
            f.write(f"# Cutoff distance: {self.cutoff_distance} Å\n")
            f.write(f"# Nitrogen atom type: {self.nitrogen_type} (from type_to_element mapping)\n")
            if times:
                f.write(f"# Time range: {min(times):.1f} - {max(times):.1f} ps\n")
                f.write(f"# Frame index range: {min(frame_numbers)} - {max(frame_numbers)}\n")
                f.write(f"# Average bubble size: {np.mean(bubble_sizes):.1f} nitrogen atoms\n")
                f.write(f"# Maximum bubble size: {max(bubble_sizes)} nitrogen atoms\n")
                f.write(f"# Minimum bubble size: {min(bubble_sizes)} nitrogen atoms\n")

        self.logger.info(f"Statistics saved to: {stats_file}")
    
    def parse_lattice_from_extended_xyz(self, header_line):
        """Parse lattice information from an extended xyz header line"""
        try:
            # Search for lattice information
            lattice_match = re.search(r'lattice="([^"]+)"', header_line)
            if lattice_match:
                lattice_str = lattice_match.group(1)
                lattice_values = list(map(float, lattice_str.split()))

                # Orthorhombic box format: "a 0.0 0.0 0.0 b 0.0 0.0 0.0 c"
                if len(lattice_values) == 9:
                    box_dims = np.array([lattice_values[0], lattice_values[4], lattice_values[8]])
                    self.logger.debug(f"Box dimensions parsed from extended xyz: {box_dims}")
                    return box_dims
                else:
                    self.logger.warning(f"Unsupported lattice format: {lattice_str}")
                    return None
            else:
                self.logger.warning("Lattice information not found")
                return None

        except Exception as e:
            self.logger.error(f"Failed to parse lattice information: {e}")
            return None

    def read_xyz_file_with_frame_filter(self, xyz_file, frames_to_analyze, molecules_per_group, molecule_name):
        """Read xyz files with frame filtering and extended xyz PBC information"""
        frames_data = {}  # {frame: [molecules]}
        frames_box_data = {}  # {frame: box_dims}

        if not os.path.exists(xyz_file):
            self.logger.warning(f"{molecule_name} file does not exist: {xyz_file}")
            return frames_data

        self.logger.info(f"Reading {molecule_name} file: {xyz_file}")
        
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                # Read number of atoms
                if lines[i].strip().isdigit():
                    n_atoms = int(lines[i].strip())
                    i += 1

                    # Read frame information and extended xyz header
                    frame_line = lines[i].strip()
                    frame_match = re.search(r'Frame[=\s](\d+)', frame_line)
                    if frame_match:
                        frame = int(frame_match.group(1))

                        # Parse box dimension information
                        box_dims = self.parse_lattice_from_extended_xyz(frame_line)

                        i += 1

                        # Only process frames scheduled for analysis
                        if frame in frames_to_analyze:
                            # Read atomic coordinates
                            atoms = []
                            for j in range(n_atoms):
                                if i + j < len(lines):
                                    parts = lines[i + j].strip().split()
                                    if len(parts) >= 4:
                                        element = parts[0]
                                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                        atoms.append((element, x, y, z))

                            # Parse molecules
                            molecules = self.parse_molecules(atoms, molecules_per_group, molecule_name)
                            frames_data[frame] = molecules

                            # Store box dimension information
                            if box_dims is not None:
                                frames_box_data[frame] = box_dims
                            
                        i += n_atoms
                    else:
                        i += 1
                else:
                    i += 1
            
            self.logger.info(f"Successfully read {len(frames_data)} frames of {molecule_name} data")
            total_molecules = sum(len(molecules) for molecules in frames_data.values())
            self.logger.info(f"Total {molecule_name} molecules/ions: {total_molecules}")

            # Store box information for later use
            # If box data do not yet exist, use those from the current file
            if not hasattr(self, 'ion_box_data') or not self.ion_box_data:
                self.ion_box_data = frames_box_data
            else:
                # Merge box data, preferring existing entries (assuming matching boxes)
                for frame, box_dims in frames_box_data.items():
                    if frame not in self.ion_box_data:
                        self.ion_box_data[frame] = box_dims

        except Exception as e:
            self.logger.error(f"Failed to read {molecule_name} file: {e}")

        return frames_data

    def parse_molecules(self, atoms, molecules_per_group, molecule_name):
        """Parse molecular structure"""
        molecules = []

        if molecule_name == "H3O":
            # H3O: O H H H (groups of four atoms)
            for atom_idx in range(0, len(atoms), 4):
                if atom_idx + 3 < len(atoms):
                    o_atom = atoms[atom_idx]
                    h1_atom = atoms[atom_idx + 1]
                    h2_atom = atoms[atom_idx + 2] 
                    h3_atom = atoms[atom_idx + 3]
                    
                    if (o_atom[0] == 'O' and h1_atom[0] == 'H' and 
                        h2_atom[0] == 'H' and h3_atom[0] == 'H'):
                        o_coord = np.array([o_atom[1], o_atom[2], o_atom[3]])
                        h_coords = [
                            np.array([h1_atom[1], h1_atom[2], h1_atom[3]]),
                            np.array([h2_atom[1], h2_atom[2], h2_atom[3]]),
                            np.array([h3_atom[1], h3_atom[2], h3_atom[3]])
                        ]
                        molecules.append(('O', o_coord, h_coords))
        
        elif molecule_name in ["bulk_OH", "surface_OH"]:
            # OH: O H (pairs of atoms)
            for atom_idx in range(0, len(atoms), 2):
                if atom_idx + 1 < len(atoms):
                    o_atom = atoms[atom_idx]
                    h_atom = atoms[atom_idx + 1]
                    
                    if o_atom[0] == 'O' and h_atom[0] == 'H':
                        o_coord = np.array([o_atom[1], o_atom[2], o_atom[3]])
                        h_coord = np.array([h_atom[1], h_atom[2], h_atom[3]])
                        molecules.append(('O', o_coord, [h_coord]))
        
        elif molecule_name == "surface_H":
            # H: single hydrogen atoms
            for atom in atoms:
                if atom[0] == 'H':
                    h_coord = np.array([atom[1], atom[2], atom[3]])
                    molecules.append(('H', h_coord, []))

        elif molecule_name in ["Na", "Cl"]:
            # Ions: individual atoms
            expected_element = 'Na' if molecule_name == 'Na' else 'Cl'
            for atom in atoms:
                if atom[0] == expected_element:
                    coord = np.array([atom[1], atom[2], atom[3]])
                    molecules.append((expected_element, coord, []))
        
        return molecules
    
    def find_bubble_surface_n2_atoms(self, largest_cluster_coords, box_dims):
        """Identify N2 atoms on the bubble surface (outer layer)"""
        if len(largest_cluster_coords) < 2:
            return largest_cluster_coords

        # Calculate bubble centroid
        centroid = self.calculate_centroid_pbc(largest_cluster_coords, box_dims)

        # Compute distance from each nitrogen atom to the centroid
        distances_to_center = []
        for coord in largest_cluster_coords:
            dist = self.periodic_distance(coord, centroid, box_dims)
            distances_to_center.append(dist)

        # Use 80% of the maximum distance as the surface threshold
        max_dist = max(distances_to_center)
        surface_threshold = max_dist * 0.8

        # Select surface atoms
        surface_indices = [i for i, dist in enumerate(distances_to_center)
                          if dist >= surface_threshold]
        surface_coords = largest_cluster_coords[surface_indices]

        self.logger.debug(f"Number of nitrogen atoms on bubble surface: {len(surface_coords)}/{len(largest_cluster_coords)}")

        return surface_coords

    def calculate_ion_bubble_distances(self, molecules, centroid, surface_coords, box_dims, molecule_name, frame_idx=None):
        """Calculate ion distances relative to the bubble using appropriate PBC"""
        distances_data = []

        # Prefer box dimensions read from ion files
        ion_box_dims = box_dims  # Default to LAMMPS trajectory box dimensions
        if hasattr(self, 'ion_box_data') and frame_idx is not None and frame_idx in self.ion_box_data:
            ion_box_dims = self.ion_box_data[frame_idx]
            self.logger.debug(f"Frame {frame_idx}: using box dimensions from {molecule_name} file: {ion_box_dims}")
        else:
            self.logger.debug(f"Frame {frame_idx}: using LAMMPS trajectory box dimensions: {box_dims}")

        for molecule in molecules:
            element, center_coord, other_coords = molecule

            # Distance from the central atom to the bubble centroid (d_centroid)
            d_centroid = self.periodic_distance(center_coord, centroid, ion_box_dims)

            # Distance from the central atom to the nearest N2 on the bubble surface (d_interface)
            min_surface_dist = float('inf')
            for surface_coord in surface_coords:
                dist = self.periodic_distance(center_coord, surface_coord, ion_box_dims)
                if dist < min_surface_dist:
                    min_surface_dist = dist

            d_interface = min_surface_dist
            distances_data.append((d_centroid, d_interface))

        return distances_data
    
    def save_raw_ion_distances(self, ions_distance_data, output_dir):
        """Save raw ion distance data per ion type"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save by ion type
        total_count = 0
        for ion_name, distances in ions_distance_data.items():
            if distances:
                ion_file = os.path.join(output_dir, f"raw_{ion_name}_distances.txt")
                with open(ion_file, 'w') as f:
                    f.write(f"# Raw distance data for {ion_name} ions\n")
                    f.write("# Format: d_centroid(Å) d_interface(Å)\n")
                    f.write("d_centroid\td_interface\n")

                    for d_cent, d_int in distances:
                        f.write(f"{d_cent:.6f}\t{d_int:.6f}\n")

                total_count += len(distances)
                self.logger.info(f"{ion_name} ion distance data saved: {ion_file} ({len(distances)} data points)")

        self.logger.info(f"All ion raw distance data saved ({total_count} data points)")

    def setup_nature_style(self):
        """Configure matplotlib parameters in a Nature-style aesthetic"""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'Arial',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'axes.spines.right': False,
            'axes.spines.top': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
            'text.usetex': False,
            'mathtext.default': 'regular'
        })
    
    def plot_all_ions_distance_distributions(self, ions_distance_data, output_dir):
        """Plot number density distributions for all ion distances"""
        if not ions_distance_data:
            self.logger.warning("No ion distance data available for plotting")
            return

        # Configure plotting style
        self.setup_nature_style()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define ion colors and markers
        ion_styles = {
            'H3O': {'color': '#1f77b4', 'marker': 'o', 'label': r'$\mathrm{H_3O^+}$'},
            'bulk_OH': {'color': '#ff7f0e', 'marker': 's', 'label': r'$\mathrm{OH^-(bulk)}$'},
            'surface_OH': {'color': '#2ca02c', 'marker': '^', 'label': r'$\mathrm{OH^-(surf)}$'},
            'surface_H': {'color': '#d62728', 'marker': 'v', 'label': r'$\mathrm{H^+(surf)}$'},
            'Na': {'color': '#9467bd', 'marker': 'D', 'label': r'$\mathrm{Na^+}$'},
            'Cl': {'color': '#8c564b', 'marker': 'p', 'label': r'$\mathrm{Cl^-}$'}
        }
        
        bins = 50

        # Plot two subplots: d_centroid and d_interface
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Process each ion type
        for ion_name, distances in ions_distance_data.items():
            if not distances:
                continue
                
            all_d_centroid = [d[0] for d in distances]
            all_d_interface = [d[1] for d in distances]
            
            style = ion_styles.get(ion_name, {'color': 'black', 'marker': 'o', 'label': ion_name})
            
            # d_centroid distribution
            hist_centroid, bin_edges_centroid = np.histogram(all_d_centroid, bins=bins, density=True)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2

            ax1.plot(bin_centers_centroid, hist_centroid, 
                    marker=style['marker'], color=style['color'], 
                    linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # d_interface distribution
            hist_interface, bin_edges_interface = np.histogram(all_d_interface, bins=bins, density=True)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2

            ax2.plot(bin_centers_interface, hist_interface,
                    marker=style['marker'], color=style['color'],
                    linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
        
        # Configure d_centroid subplot
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Density', fontsize=14)
        ax1.set_title('Ion Distance to Bubble Centroid', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=11)

        # Configure d_interface subplot
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Density', fontsize=14)
        ax2.set_title('Ion Distance to Bubble Surface', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=11)
        
        plt.tight_layout()
        
        # Save figure
        plot_file = os.path.join(output_dir, "all_ions_bubble_distance_distributions.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Save numerical data
        data_file = os.path.join(output_dir, "all_ions_distance_distribution_data.txt")
        with open(data_file, 'w') as f:
            f.write("# All ions distance distribution data\n")
            f.write("# Format: ion_type d_centroid_center d_centroid_density d_interface_center d_interface_density\n")

            for ion_name, distances in ions_distance_data.items():
                if not distances:
                    continue
                    
                all_d_centroid = [d[0] for d in distances]
                all_d_interface = [d[1] for d in distances]
                
                hist_centroid, bin_edges_centroid = np.histogram(all_d_centroid, bins=bins, density=True)
                bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
                
                hist_interface, bin_edges_interface = np.histogram(all_d_interface, bins=bins, density=True)
                bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
                
                f.write(f"\n# {ion_name} data\n")
                max_len = max(len(bin_centers_centroid), len(bin_centers_interface))
                for i in range(max_len):
                    centroid_center = bin_centers_centroid[i] if i < len(bin_centers_centroid) else ""
                    centroid_density = hist_centroid[i] if i < len(hist_centroid) else ""
                    interface_center = bin_centers_interface[i] if i < len(bin_centers_interface) else ""
                    interface_density = hist_interface[i] if i < len(hist_interface) else ""
                    f.write(f"{ion_name}\t{centroid_center}\t{centroid_density}\t{interface_center}\t{interface_density}\n")
        

        
        self.logger.info(f"Saved ion distance distribution plots: {plot_file}")
        self.logger.info(f"Saved distribution data: {data_file}")

        # Print statistics
        for ion_name, distances in ions_distance_data.items():
            if distances:
                all_d_centroid = [d[0] for d in distances]
                all_d_interface = [d[1] for d in distances]
                self.logger.info(f"{ion_name} distance statistics:")
                self.logger.info(f"  d_centroid: mean={np.mean(all_d_centroid):.3f}±{np.std(all_d_centroid):.3f} Å")
                self.logger.info(f"  d_interface: mean={np.mean(all_d_interface):.3f}±{np.std(all_d_interface):.3f} Å")

def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Calculate nitrogen bubble centroids from LAMMPS trajectories')

    # Required parameters
    parser.add_argument('--traj_file', default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_nacl_tio2_water_layer/6/0-1.6ns/bubble_1k.lammpstrj',  help='Path to the LAMMPS trajectory file')

    # Optional parameters
    parser.add_argument('--data', default='../model_atomic.data', help='Path to the LAMMPS data file (default: ../model_atomic.data)')
    parser.add_argument('--output', default='bubble_centroids.txt', help='Output filename')
    parser.add_argument('--cutoff', type=float, default=5.5, help='Nitrogen clustering cutoff distance (Å)')
    parser.add_argument('--atom_style', default="id type x y z", help='LAMMPS atom_style (e.g., "id type x y z", "atomic", "full")')

    # Trajectory frame selection parameters
    parser.add_argument('--step_interval', type=int, default=1, help='Frame interval for analysis (analyze every n frames)')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame for analysis')
    parser.add_argument('--end_frame', type=int, default=-1, help='Ending frame for analysis; -1 analyzes to the final frame')

    # Ion analysis parameters
    ion_base_path = '../find_ion_4/ion_analysis_results'
    parser.add_argument('--h3o_file', default=f'{ion_base_path}/solution_bulk_h3o.xyz', help='Trajectory file for H3O ions')
    parser.add_argument('--bulk_oh_file', default=f'{ion_base_path}/solution_bulk_oh.xyz', help='Trajectory file for bulk OH ions')
    parser.add_argument('--surface_oh_file', default=f'{ion_base_path}/solution_surface_oh.xyz', help='Trajectory file for surface OH ions')
    parser.add_argument('--surface_h_file', default=f'{ion_base_path}/tio2_surface_h.xyz', help='Trajectory file for surface H ions')
    parser.add_argument('--na_file', default=f'{ion_base_path}/na_ions.xyz', help='Trajectory file for Na ions')
    parser.add_argument('--cl_file', default=f'{ion_base_path}/cl_ions.xyz', help='Trajectory file for Cl ions')
    parser.add_argument('--ions_output', default='ions_analysis', help='Output directory for ion analysis results')
    parser.add_argument('--disable_ions', action='store_true', help='Disable ion analysis and only compute bubble centroids')

    return parser.parse_args()

def main():
    """Entry point"""
    args = get_args()

    # Validate input files
    if not os.path.exists(args.traj_file):
        print(f"Error: trajectory file {args.traj_file} does not exist")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"Error: data file {args.data} does not exist")
        print("Hint: use the --data argument to specify the correct data file path")
        sys.exit(1)

    # Check ion files (optional)
    ion_files = {}
    if not args.disable_ions:
        ion_file_configs = {
            'h3o': args.h3o_file,
            'bulk_oh': args.bulk_oh_file,
            'surface_oh': args.surface_oh_file,
            'surface_h': args.surface_h_file,
            'na': args.na_file,
            'cl': args.cl_file
        }

        for ion_name, file_path in ion_file_configs.items():
            if file_path and os.path.exists(file_path):
                ion_files[ion_name] = file_path
                print(f"Found {ion_name} file: {file_path}")
            elif file_path:
                # Na and Cl ions may be absent in some systems; skip quietly
                if ion_name in ['na', 'cl']:
                    print(f"Note: {ion_name} file does not exist; skipping analysis for this ion (may be absent in some systems)")
                else:
                    print(f"Warning: {ion_name} file {file_path} does not exist; skipping this ion analysis")

        if not ion_files:
            print("Warning: no valid ion files found; only bubble centroids will be calculated")
    else:
        print("Ion analysis disabled; only bubble centroids will be calculated")

    # Instantiate calculator
    calculator = BubbleCentroidCalculator(
        cutoff_distance=args.cutoff
    )

    # Use provided data file
    data_file = args.data

    try:
        # Process trajectory
        times, centroids, sizes = calculator.process_trajectory(
            data_file=data_file,
            traj_file=args.traj_file,
            atom_style=args.atom_style,
            output_file=args.output,
            step_interval=args.step_interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            ion_files=ion_files if ion_files else None,
            ions_analysis_output=args.ions_output if ion_files else None
        )
        
        print("\n" + "="*50)
        print("Calculation complete!")
        print(f"Processed {len(times)} frames")
        if times:
            print(f"Time range: {min(times):.1f} - {max(times):.1f} ps")
            print(f"Average bubble size: {np.mean(sizes):.1f} nitrogen atoms")
        print(f"Bubble centroid results saved to: {args.output}")

        if ion_files:
            print(f"Ion analysis results saved to: {args.ions_output}/")
            print("  - raw_[ion_type]_distances.txt: raw distance data for each ion type")
            print("  - all_ions_bubble_distance_distributions.png: distance distribution plot")
            print("  - all_ions_distance_distribution_data.txt: distribution statistics")
            print(f"  Ion types analyzed: {', '.join(ion_files.keys())}")

        print("="*50)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()