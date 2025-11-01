#!/usr/bin/env python3
"""
Transition matrix analysis script.

Features:
1. Read xyz trajectory data for multiple time segments (solution_bulk_oh, solution_surface_oh, solution_surface_h2o).
2. Extract atom index information to track species transitions.
3. Compute the transition matrix (based on O atom indices).
4. Visualize the transition matrix (counts and probabilities).
5. Special analysis: check whether the three indices of OHH in solution_surface_h2o are consecutive when solution_surface_oh → solution_surface_h2o.
   - Consecutive: proton transfer (back-and-forth hopping).
   - Non-consecutive: bulk_h2o transfers H to surface_oh.
6. Save all data and statistics.
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import glob
import re
from collections import defaultdict
import argparse

def setup_nature_style():
    """Configure matplotlib parameters to mimic Nature journal styling."""
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

class TransitionAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.time_segments = []
        
        # Store molecule data for each time segment: {segment_key: {frame: [molecules]}}
        # molecule: {'coords': [(x,y,z), ...], 'symbols': ['O', 'H', ...], 'indices': [idx1, idx2, ...]}
        self.bulk_oh_data = {}
        self.surface_oh_data = {}
        self.surface_h2o_data = {}
        
        # Locate all time segments
        self._find_time_segments()
        
    def _find_time_segments(self):
        """Find all time-segment directories."""
        pattern = os.path.join(self.base_path, "*ns")
        dirs = glob.glob(pattern)
        
        # Sort in chronological order
        time_pattern = r'(\d+\.?\d*)-(\d+\.?\d*)ns'
        segments = []
        for d in dirs:
            dir_name = os.path.basename(d)
            match = re.search(time_pattern, dir_name)
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                # Check for the find_ion_4_matrix directory
                matrix_dir = os.path.join(d, "find_ion_4_matrix", "ion_analysis_results")
                if os.path.exists(matrix_dir):
                    segments.append((start_time, end_time, d))
        
        segments.sort(key=lambda x: x[0])
        self.time_segments = segments
        
        print(f"Found {len(self.time_segments)} time segments:")
        for start, end, path in self.time_segments:
            print(f"  {start}-{end}ns: {os.path.basename(path)}")
    
    def parse_xyz_with_indices(self, xyz_file):
        """
        Read an xyz file and extract molecule information (including atom indices).

        Returns: {frame: [molecules]}
        molecule: {'coords': [(x,y,z), ...], 'symbols': ['O', 'H', ...], 'indices': [idx1, idx2, ...]}
        """
        frame_data = {}

        if not os.path.exists(xyz_file):
            print(f"  Warning: file not found: {xyz_file}")
            return frame_data
        
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                # Read the atom count
                if lines[i].strip() and lines[i].strip().isdigit():
                    n_atoms = int(lines[i].strip())
                    i += 1
                    
                    if i >= len(lines):
                        break
                    
                    # Read frame information
                    frame_line = lines[i].strip()
                    frame_match = re.search(r'Frame[=\s]+(\d+)', frame_line)
                    if frame_match:
                        frame = int(frame_match.group(1))
                        i += 1
                        
                        # Check whether an atom_index column is present
                        has_indices = 'atom_index' in frame_line.lower()
                        
                        # Read atom data
                        atoms = []
                        for j in range(n_atoms):
                            if i + j < len(lines):
                                parts = lines[i + j].strip().split()
                                if len(parts) >= 4:
                                    element = parts[0]
                                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                    atom_idx = int(parts[4]) if len(parts) >= 5 and has_indices else None
                                    atoms.append({'symbol': element, 'coord': (x, y, z), 'index': atom_idx})
                        
                        # Parse molecules
                        molecules = self._group_atoms_into_molecules(atoms)
                        frame_data[frame] = molecules
                        
                        i += n_atoms
                    else:
                        i += 1
                else:
                    i += 1
        
        except Exception as e:
            print(f"  Error: failed to read {xyz_file}: {e}")
        
        return frame_data
    
    def _group_atoms_into_molecules(self, atoms):
        """Group a list of atoms into molecules."""
        molecules = []
        
        # Determine the molecular size
        if not atoms:
            return molecules
        
        # Count the number of O and H atoms
        o_count = sum(1 for atom in atoms if atom['symbol'] == 'O')
        h_count = sum(1 for atom in atoms if atom['symbol'] == 'H')
        
        # Determine the number of atoms per molecule
        if o_count > 0:
            atoms_per_mol = len(atoms) // o_count
        else:
            return molecules
        
        # Group by molecule
        for i in range(0, len(atoms), atoms_per_mol):
            if i + atoms_per_mol <= len(atoms):
                mol_atoms = atoms[i:i+atoms_per_mol]
                
                        # Validate the molecule
                symbols = [a['symbol'] for a in mol_atoms]
                coords = [a['coord'] for a in mol_atoms]
                indices = [a['index'] for a in mol_atoms if a['index'] is not None]
                
                # Validate molecular integrity
                if atoms_per_mol == 2:  # OH
                    if symbols == ['O', 'H'] and len(indices) == 2:
                        molecules.append({
                            'symbols': symbols,
                            'coords': coords,
                            'indices': indices
                        })
                elif atoms_per_mol == 3:  # H2O
                    if symbols == ['O', 'H', 'H'] and len(indices) == 3:
                        molecules.append({
                            'symbols': symbols,
                            'coords': coords,
                            'indices': indices
                        })
        
        return molecules
    
    def load_all_data(self):
        """Load trajectory data for all time segments."""
        print("\nStarting to load trajectory data...")

        for start_time, end_time, segment_path in self.time_segments:
            segment_key = f"{start_time}-{end_time}ns"
            print(f"\nLoading time segment: {segment_key}")

            results_dir = os.path.join(segment_path, "find_ion_4_matrix", "ion_analysis_results")

            # Load bulk OH
            bulk_oh_file = os.path.join(results_dir, "solution_bulk_oh.xyz")
            print(f"  Reading solution_bulk_oh.xyz...")
            self.bulk_oh_data[segment_key] = self.parse_xyz_with_indices(bulk_oh_file)
            print(f"    Loaded {len(self.bulk_oh_data[segment_key])} frames")

            # Load surface OH
            surface_oh_file = os.path.join(results_dir, "solution_surface_oh.xyz")
            print(f"  Reading solution_surface_oh.xyz...")
            self.surface_oh_data[segment_key] = self.parse_xyz_with_indices(surface_oh_file)
            print(f"    Loaded {len(self.surface_oh_data[segment_key])} frames")

            # Load surface H2O
            surface_h2o_file = os.path.join(results_dir, "solution_surface_h2o.xyz")
            print(f"  Reading solution_surface_h2o.xyz...")
            self.surface_h2o_data[segment_key] = self.parse_xyz_with_indices(surface_h2o_file)
            print(f"    Loaded {len(self.surface_h2o_data[segment_key])} frames")

        print("\nData loading complete!")
    
    def build_unified_frame_sequence(self):
        """
        Build a unified frame sequence by combining all time segments.

        Returns: [(segment_key, frame), ...] sorted in chronological order.
        """
        frame_sequence = []

        for start_time, end_time, segment_path in self.time_segments:
            segment_key = f"{start_time}-{end_time}ns"

            # Gather all frames for this time segment
            frames = set()
            if segment_key in self.bulk_oh_data:
                frames.update(self.bulk_oh_data[segment_key].keys())
            if segment_key in self.surface_oh_data:
                frames.update(self.surface_oh_data[segment_key].keys())
            if segment_key in self.surface_h2o_data:
                frames.update(self.surface_h2o_data[segment_key].keys())

            # Append to the sequence
            for frame in sorted(frames):
                frame_sequence.append((segment_key, frame))
        
        return frame_sequence
    
    def build_transition_matrix(self):
        """
        Construct the species transition matrix.

        Returns:
        - transition_counts: transition count matrix
        - species_names: list of species names
        - transition_details: detailed transition records
        - oh_to_h2o_analysis: detailed analysis for OH→H2O conversions
        """
        species_names = ['solution_bulk_oh', 'solution_surface_oh', 'solution_surface_h2o']
        n_species = len(species_names)
        
        # Initialize the transition count matrix
        transition_counts = np.zeros((n_species, n_species), dtype=int)

        # Store detailed transition information
        transition_details = []

        # OH → H2O conversion analysis
        oh_to_h2o_transitions = {
            'consecutive_indices': 0,  # consecutive indices (proton transfer)
            'non_consecutive_indices': 0,  # non-consecutive indices (bulk H2O transfers H)
            'details': []
        }

        # Obtain the unified frame sequence
        frame_sequence = self.build_unified_frame_sequence()

        print(f"\nStarting transition matrix calculation...")
        print(f"Total frames: {len(frame_sequence)}")

        # Iterate over neighboring frames
        for idx in range(len(frame_sequence) - 1):
            current_segment, current_frame = frame_sequence[idx]
            next_segment, next_frame = frame_sequence[idx + 1]
            
            # Build the mapping from O atom index to species for the current frame
            current_o_to_species = {}
            current_o_to_mol = {}  # Map O atom indices to full molecule information
            
            # Bulk OH
            if current_segment in self.bulk_oh_data and current_frame in self.bulk_oh_data[current_segment]:
                for mol in self.bulk_oh_data[current_segment][current_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]  # The O atom is always first
                        current_o_to_species[o_idx] = 0
                        current_o_to_mol[o_idx] = mol
            
            # Surface OH
            if current_segment in self.surface_oh_data and current_frame in self.surface_oh_data[current_segment]:
                for mol in self.surface_oh_data[current_segment][current_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        current_o_to_species[o_idx] = 1
                        current_o_to_mol[o_idx] = mol
            
            # Surface H2O
            if current_segment in self.surface_h2o_data and current_frame in self.surface_h2o_data[current_segment]:
                for mol in self.surface_h2o_data[current_segment][current_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        current_o_to_species[o_idx] = 2
                        current_o_to_mol[o_idx] = mol
            
            # Build the mapping for the next frame
            next_o_to_species = {}
            next_o_to_mol = {}
            
            # Bulk OH
            if next_segment in self.bulk_oh_data and next_frame in self.bulk_oh_data[next_segment]:
                for mol in self.bulk_oh_data[next_segment][next_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        next_o_to_species[o_idx] = 0
                        next_o_to_mol[o_idx] = mol
            
            # Surface OH
            if next_segment in self.surface_oh_data and next_frame in self.surface_oh_data[next_segment]:
                for mol in self.surface_oh_data[next_segment][next_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        next_o_to_species[o_idx] = 1
                        next_o_to_mol[o_idx] = mol
            
            # Surface H2O
            if next_segment in self.surface_h2o_data and next_frame in self.surface_h2o_data[next_segment]:
                for mol in self.surface_h2o_data[next_segment][next_frame]:
                    if mol['indices']:
                        o_idx = mol['indices'][0]
                        next_o_to_species[o_idx] = 2
                        next_o_to_mol[o_idx] = mol
            
            # Accumulate transitions
            for o_idx, from_species in current_o_to_species.items():
                if o_idx in next_o_to_species:
                    to_species = next_o_to_species[o_idx]
                    transition_counts[from_species, to_species] += 1

                    # Record detailed transition information
                    if from_species != to_species:
                        transition_info = {
                            'segment': current_segment,
                            'frame': current_frame,
                            'o_index': o_idx,
                            'from': species_names[from_species],
                            'to': species_names[to_species]
                        }
                        transition_details.append(transition_info)
                        
                        # Special analysis: surface OH → surface H2O
                        if from_species == 1 and to_species == 2:  # surface_oh → surface_h2o
                            # Check whether the three indices in H2O are consecutive
                            h2o_mol = next_o_to_mol[o_idx]
                            h2o_indices = sorted(h2o_mol['indices'])

                            # Determine whether the indices are consecutive
                            is_consecutive = (h2o_indices[1] == h2o_indices[0] + 1 and 
                                            h2o_indices[2] == h2o_indices[1] + 1)
                            
                            if is_consecutive:
                                oh_to_h2o_transitions['consecutive_indices'] += 1
                            else:
                                oh_to_h2o_transitions['non_consecutive_indices'] += 1
                            
                            oh_to_h2o_transitions['details'].append({
                                'segment': current_segment,
                                'frame': current_frame,
                                'o_index': o_idx,
                                'oh_indices': current_o_to_mol[o_idx]['indices'],
                                'h2o_indices': h2o_indices,
                                'is_consecutive': is_consecutive
                            })
        
        print(f"Transition matrix calculation finished!")
        print(f"Total transition events: {len(transition_details)}")
        print(f"surface_oh → surface_h2o transitions: {oh_to_h2o_transitions['consecutive_indices'] + oh_to_h2o_transitions['non_consecutive_indices']}")
        
        return transition_counts, species_names, transition_details, oh_to_h2o_transitions
    
    def plot_transition_matrix(self, transition_counts, species_names, output_file):
        """Plot the transition matrix."""
        setup_nature_style()

        # Calculate the transition probability matrix
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_prob = transition_counts / row_sums

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Simplify species names for display
        display_names = [
            r'Bulk OH$^-$',
            r'Surface OH$^-$',
            r'Surface H$_2$O'
        ]

        # Plot the transition count matrix
        im1 = ax1.imshow(transition_counts, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(species_names)))
        ax1.set_yticks(range(len(species_names)))
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        ax1.set_yticklabels(display_names)
        ax1.set_xlabel('To Species')
        ax1.set_ylabel('From Species')
        ax1.set_title('Transition Counts')

        # Add values to each cell
        for i in range(len(species_names)):
            for j in range(len(species_names)):
                text = ax1.text(j, i, int(transition_counts[i, j]),
                              ha="center", va="center", 
                              color="black" if transition_prob[i, j] < 0.5 else "white",
                              fontsize=14, fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Counts', rotation=270, labelpad=20)
        
        # Plot the transition probability matrix
        im2 = ax2.imshow(transition_prob, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(species_names)))
        ax2.set_yticks(range(len(species_names)))
        ax2.set_xticklabels(display_names, rotation=45, ha='right')
        ax2.set_yticklabels(display_names)
        ax2.set_xlabel('To Species')
        ax2.set_ylabel('From Species')
        ax2.set_title('Transition Probability')
        
        # Add probability values to each cell
        for i in range(len(species_names)):
            for j in range(len(species_names)):
                text = ax2.text(j, i, f'{transition_prob[i, j]:.3f}',
                              ha="center", va="center", 
                              color="black" if transition_prob[i, j] < 0.5 else "white",
                              fontsize=14, fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Probability', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Transition matrix figure saved to: {output_file}")
    
    def plot_transition_probability_matrix_only(self, transition_counts, species_names, output_file):
        """Plot only the transition probability matrix (refined version)."""
        setup_nature_style()

        # Calculate the transition probability matrix
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_prob = transition_counts / row_sums

        # Create a square figure
        fig, ax = plt.subplots(figsize=(5, 5))

        # Improved species labels - ion name (location)
        display_names = [
            r'OH$^-$ (bulk)',
            r'OH$^-$ (surf)',
            r'H$_2$O (surf)'
        ]

        # Use a white-to-blue gradient
        im = ax.imshow(transition_prob, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(species_names)))
        ax.set_yticks(range(len(species_names)))
        ax.set_xticklabels(display_names, fontsize=14)
        #ax.set_yticklabels(display_names, fontsize=14)
        ax.set_yticklabels(display_names, fontsize=14, rotation=90,va='center', ha='center',x=-0.05)
        ax.set_xlabel('To Species', fontsize=15)
        ax.set_ylabel('From Species', fontsize=15, rotation=90)
        
        # Add probability values to each cell
        for i in range(len(species_names)):
            for j in range(len(species_names)):
                # Automatically adjust text color based on the background
                text_color = "white" if transition_prob[i, j] > 0.6 else "black"
                text = ax.text(j, i, f'{transition_prob[i, j]:.3f}',
                               ha="center", va="center",
                               color=text_color,
                               fontsize=15)

        # Place the colorbar horizontally at the top with a slight offset downward
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        #cbar.set_label('Transition Probability', fontsize=13, labelpad=12)
        cbar.ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Transition probability matrix (standalone) saved to: {output_file}")
    
    def plot_oh_to_h2o_analysis(self, oh_to_h2o_transitions, output_file):
        """Plot the OH→H2O conversion mechanism analysis."""
        setup_nature_style()
        
        total = (oh_to_h2o_transitions['consecutive_indices'] + 
                oh_to_h2o_transitions['non_consecutive_indices'])
        
        if total == 0:
            print("Warning: no surface_oh → surface_h2o transition events")
            return
        
        consecutive_ratio = oh_to_h2o_transitions['consecutive_indices'] / total
        non_consecutive_ratio = oh_to_h2o_transitions['non_consecutive_indices'] / total
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        labels = [
            f'Proton Transfer\n(consecutive indices)\n{consecutive_ratio:.1%}',
            f'Bulk H$_2$O Transfer\n(non-consecutive indices)\n{non_consecutive_ratio:.1%}'
        ]
        sizes = [oh_to_h2o_transitions['consecutive_indices'],
                oh_to_h2o_transitions['non_consecutive_indices']]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0.05)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax1.set_title(r'Surface OH$^-$ → Surface H$_2$O Mechanism', fontsize=14, fontweight='bold')
        
        # Bar chart
        categories = ['Proton\nTransfer', 'Bulk H$_2$O\nTransfer']
        counts = [oh_to_h2o_transitions['consecutive_indices'],
                 oh_to_h2o_transitions['non_consecutive_indices']]

        bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(r'Surface OH$^-$ → Surface H$_2$O Transition Counts',
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add counts on top of the bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"OH→H2O mechanism analysis figure saved to: {output_file}")
    
    def plot_oh_to_h2o_bar_only(self, oh_to_h2o_transitions, output_file):
        """Plot only the OH→H2O conversion bar chart (refined, percentage normalized)."""
        setup_nature_style()
        
        total = (oh_to_h2o_transitions['consecutive_indices'] + 
                oh_to_h2o_transitions['non_consecutive_indices'])
        
        if total == 0:
            print("Warning: no surface_oh → surface_h2o transition events")
            return
        
        consecutive_ratio = oh_to_h2o_transitions['consecutive_indices'] / total * 100
        non_consecutive_ratio = oh_to_h2o_transitions['non_consecutive_indices'] / total * 100
        
        # Create a narrower figure
        fig, ax = plt.subplots(figsize=(2, 4))

        # Improved label descriptions
        #categories = [
        #    'Surface H$^+$\nhopping',  # Surface proton hopping
        #    'Proton exchange\nwith bulk H$_2$O'  # Proton exchange with bulk H2O
        #]
        categories = [
            'PT(surf)',  # Surface proton hopping
            'PT(bulk)'  # Proton exchange with bulk H2O
        ]

        percentages = [consecutive_ratio, non_consecutive_ratio]

        # Nature-style color palette - more professional colors
        colors = ['#E64B35', '#4DBBD5']  # Red and cyan, high contrast
        
        bars = ax.bar(categories, percentages, color=colors, alpha=0.85, 
                     edgecolor='black', linewidth=1, width=0.3)
        
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.set_ylim(0, 65)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        
        # Add percentage values on top of the bars
        #for bar, percentage in zip(bars, percentages):
        #    height = bar.get_height()
        #    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
        #            f'{percentage:.1f}%',
        #            ha='center', va='bottom', fontsize=15)

        # Adjust x-axis label styling
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"OH→H2O mechanism bar chart (standalone) saved to: {output_file}")
    
    def save_transition_data(self, transition_counts, species_names, transition_details,
                            oh_to_h2o_transitions, output_dir):
        """Save transition matrix data and associated statistics."""

        os.makedirs(output_dir, exist_ok=True)

        # 1. Save the transition count matrix
        counts_file = os.path.join(output_dir, "transition_counts.txt")
        with open(counts_file, 'w') as f:
            f.write("Transition Counts Matrix\n")
            f.write("From \\ To\t" + "\t".join(species_names) + "\n")
            for i, from_species in enumerate(species_names):
                f.write(f"{from_species}\t")
                f.write("\t".join(str(int(transition_counts[i, j])) for j in range(len(species_names))))
                f.write("\n")
        print(f"Transition count matrix saved to: {counts_file}")

        # 2. Save the transition probability matrix
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_prob = transition_counts / row_sums

        prob_file = os.path.join(output_dir, "transition_probabilities.txt")
        with open(prob_file, 'w') as f:
            f.write("Transition Probability Matrix\n")
            f.write("From \\ To\t" + "\t".join(species_names) + "\n")
            for i, from_species in enumerate(species_names):
                f.write(f"{from_species}\t")
                f.write("\t".join(f"{transition_prob[i, j]:.6f}" for j in range(len(species_names))))
                f.write("\n")
        print(f"Transition probability matrix saved to: {prob_file}")

        # 3. Save detailed transition information
        details_file = os.path.join(output_dir, "transition_details.txt")
        with open(details_file, 'w') as f:
            f.write("Segment\tFrame\tO_Index\tFrom_Species\tTo_Species\n")
            for detail in transition_details:
                f.write(f"{detail['segment']}\t{detail['frame']}\t{detail['o_index']}\t"
                       f"{detail['from']}\t{detail['to']}\n")
        print(f"Detailed transition information saved to: {details_file}")

        # 4. Save the transition statistics summary
        summary_file = os.path.join(output_dir, "transition_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== Species Transition Summary ===\n\n")
            
            total_transitions = np.sum(transition_counts) - np.trace(transition_counts)
            f.write(f"Total transitions (excluding self-transitions): {int(total_transitions)}\n\n")
            
            for i, species in enumerate(species_names):
                f.write(f"\n{species}:\n")
                total_from = np.sum(transition_counts[i, :])
                total_to = np.sum(transition_counts[:, i])
                f.write(f"  Total transitions from this species: {int(total_from)}\n")
                f.write(f"  Total transitions to this species: {int(total_to)}\n")
                
                if total_from > 0:
                    f.write(f"  Main transitions from {species}:\n")
                    for j in range(len(species_names)):
                            if i != j and transition_counts[i, j] > 0:
                                prob = transition_prob[i, j]
                                f.write(f"    -> {species_names[j]}: {int(transition_counts[i, j])} ({prob:.2%})\n")
        print(f"Transition summary saved to: {summary_file}")

        # 5. Save the OH→H2O conversion mechanism analysis
        oh_h2o_file = os.path.join(output_dir, "oh_to_h2o_mechanism_analysis.txt")
        total = (oh_to_h2o_transitions['consecutive_indices'] +
                oh_to_h2o_transitions['non_consecutive_indices'])
        
        with open(oh_h2o_file, 'w') as f:
            f.write("=== Surface OH- → Surface H2O Mechanism Analysis ===\n\n")
            f.write(f"Total transitions: {total}\n\n")
            
            if total > 0:
                f.write("Mechanism 1: Proton Transfer (consecutive indices)\n")
                f.write(f"  Count: {oh_to_h2o_transitions['consecutive_indices']}\n")
                f.write(f"  Percentage: {oh_to_h2o_transitions['consecutive_indices']/total:.2%}\n")
                f.write("  Interpretation: OH- ⇌ H2O proton transfer (back-and-forth)\n\n")
                
                f.write("Mechanism 2: Bulk H2O H-transfer (non-consecutive indices)\n")
                f.write(f"  Count: {oh_to_h2o_transitions['non_consecutive_indices']}\n")
                f.write(f"  Percentage: {oh_to_h2o_transitions['non_consecutive_indices']/total:.2%}\n")
                f.write("  Interpretation: Bulk H2O transfers H to surface OH-\n\n")
                
                f.write("\n=== Detailed Events ===\n")
                f.write("Segment\tFrame\tO_Index\tOH_Indices\tH2O_Indices\tIs_Consecutive\tMechanism\n")
                for detail in oh_to_h2o_transitions['details']:
                    mechanism = "Proton Transfer" if detail['is_consecutive'] else "Bulk H2O Transfer"
                    f.write(f"{detail['segment']}\t{detail['frame']}\t{detail['o_index']}\t"
                           f"{detail['oh_indices']}\t{detail['h2o_indices']}\t"
                           f"{detail['is_consecutive']}\t{mechanism}\n")

        print(f"OH→H2O mechanism analysis saved to: {oh_h2o_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze the species transition matrix')
    parser.add_argument('--base_path', type=str,
                       default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4',
                       help='Base directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4/analysis_ion/find_ion_4_matrix/results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Species Transition Matrix Analysis")
    print("="*80)
    
    # Create the analyzer
    analyzer = TransitionAnalyzer(args.base_path)

    # Load data
    analyzer.load_all_data()

    # Compute the transition matrix
    transition_counts, species_names, transition_details, oh_to_h2o_transitions = \
        analyzer.build_transition_matrix()

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot the transition matrix (full version)
    matrix_plot = os.path.join(args.output_dir, "transition_matrix.png")
    analyzer.plot_transition_matrix(transition_counts, species_names, matrix_plot)

    # Plot only the transition probability matrix (refined)
    prob_matrix_only = os.path.join(args.output_dir, "transition_probability_matrix.png")
    analyzer.plot_transition_probability_matrix_only(transition_counts, species_names, prob_matrix_only)

    # Plot the OH→H2O mechanism analysis (full version)
    oh_h2o_plot = os.path.join(args.output_dir, "oh_to_h2o_mechanism.png")
    analyzer.plot_oh_to_h2o_analysis(oh_to_h2o_transitions, oh_h2o_plot)

    # Plot only the OH→H2O mechanism bar chart (refined)
    oh_h2o_bar_only = os.path.join(args.output_dir, "oh_to_h2o_bar.png")
    analyzer.plot_oh_to_h2o_bar_only(oh_to_h2o_transitions, oh_h2o_bar_only)

    # Save data
    analyzer.save_transition_data(transition_counts, species_names, transition_details,
                                  oh_to_h2o_transitions, args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

