#!/usr/bin/env python3
"""
Script for batch analysis of ion distance distributions.
Reads ion distance data across multiple time ranges to generate comprehensive distribution plots, time-grouped distributions, and charge distribution plots.
Supports command-line arguments to specify the input path and output directory.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from collections import defaultdict
import re
import glob
import argparse

class BatchIonAnalyzer:
    """Batch ion distance analyzer."""
    
    def __init__(self, base_dir="/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4", output_dir=None):
        self.base_dir = base_dir
        
        # Configure the output directory
        if output_dir is None:
            self.output_dir = os.path.join(base_dir, "analysis_ion/centroids_density")
        else:
            # If it is a relative path, resolve it relative to base_dir; otherwise use the absolute path directly
            if os.path.isabs(output_dir):
                self.output_dir = output_dir
            else:
                self.output_dir = os.path.join(base_dir, output_dir)
        
        # Define ion plotting styles
        self.ion_styles = {
            'H3O': {'color': '#1f77b4', 'marker': 'o', 'label': r'$\mathrm{H_3O^+}$'},
            'bulk_OH': {'color': '#ff7f0e', 'marker': 's', 'label': r'$\mathrm{OH^-(bulk)}$'},
            'surface_OH': {'color': '#2ca02c', 'marker': '^', 'label': r'$\mathrm{OH^-(surf)}$'},
            'surface_H': {'color': '#d62728', 'marker': 'v', 'label': r'$\mathrm{H^+(surf)}$'},
            'Na': {'color': '#9467bd', 'marker': 'D', 'label': r'$\mathrm{Na^+}$'},
            'Cl': {'color': '#8c564b', 'marker': 'p', 'label': r'$\mathrm{Cl^-}$'}
        }
        
        # Define ion charges
        self.ion_charges = {
            'H3O': 1,
            'bulk_OH': -1,
            'surface_OH': -1,
            'surface_H': 1,
            'Na': 1,
            'Cl': -1
        }
        
        # Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_nature_style(self):
        """Configure matplotlib parameters in a Nature-style format."""
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
    
    def find_time_directories(self):
        """Find all time-range directories and sort them chronologically."""
        pattern = re.compile(r'(\d+\.?\d*)-(\d+\.?\d*)ns')
        time_dirs = []
        
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path) and pattern.match(item):
                match = pattern.match(item)
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                time_dirs.append((start_time, end_time, item, item_path))
        
        # Sort by the starting time
        time_dirs.sort(key=lambda x: x[0])
        
        print(f"Found {len(time_dirs)} time-range directories:")
        total_time = 0
        for start, end, dirname, _ in time_dirs:
            duration = end - start
            total_time += duration
            print(f"  {dirname}: {start}-{end}ns (duration: {duration}ns)")
        print(f"Total simulation time: {total_time}ns")
        
        return time_dirs, total_time
    
    def read_ion_distances(self, time_dirs, centroids_subpath="centroids_density_2"):
        """Read ion distance data for all time ranges."""
        all_ion_data = defaultdict(list)  # {ion_type: [(d_centroid, d_interface, time_period), ...]}
        time_grouped_data = defaultdict(lambda: defaultdict(list))  # {time_period: {ion_type: [(d_centroid, d_interface), ...]}}
        
        for i, (start_time, end_time, dirname, dir_path) in enumerate(time_dirs):
            ions_analysis_dir = os.path.join(dir_path, f"{centroids_subpath}/ions_analysis")
            
            if not os.path.exists(ions_analysis_dir):
                print(f"Warning: {ions_analysis_dir} does not exist, skipping")
                continue

            print(f"Processing time range {i+1}/{len(time_dirs)}: {dirname}")

            # Find all raw distance files
            distance_files = glob.glob(os.path.join(ions_analysis_dir, "raw_*_distances.txt"))

            for file_path in distance_files:
                # Extract ion type from the filename
                filename = os.path.basename(file_path)
                ion_match = re.search(r'raw_(.+)_distances\.txt', filename)
                if not ion_match:
                    continue

                ion_type = ion_match.group(1)

                # Read the data
                try:
                    # First check whether the file contains valid data
                    with open(file_path, 'r') as f:
                        lines = f.readlines()

                    # Skip the header and check for data rows
                    data_lines = [line.strip() for line in lines[3:] if line.strip() and not line.startswith('#')]

                    if not data_lines:
                        print(f"    Skipping {ion_type}: no data in file (the system may not contain this ion)")
                        continue

                    # Load numerical data
                    data = np.loadtxt(file_path, skiprows=3, dtype=float)
                    if data.size == 0:
                        print(f"    Skipping {ion_type}: no valid numerical data")
                        continue

                    if len(data.shape) == 1:
                        data = data.reshape(1, -1)

                    print(f"    Successfully read {ion_type}: {len(data)} data points")

                    # Add to the aggregated dataset
                    for row in data:
                        # According to the new definition, d_interface is the distance from the ion to the nearest N atom.
                        # This script assumes upstream data generation already adopts this definition, with column two containing it.
                        d_centroid, d_interface = row[0], row[1]
                        all_ion_data[ion_type].append((d_centroid, d_interface, i))
                        time_grouped_data[i][ion_type].append((d_centroid, d_interface))

                except Exception as e:
                    print(f"    Error reading {ion_type}: {e}")
                    # Provide a specific reminder for Na or Cl ions
                    if ion_type in ['Na', 'Cl']:
                        print(f"      Note: Some systems may not contain {ion_type} ions")

        # Report which ion types were found
        found_ions = list(all_ion_data.keys())
        expected_ions = ['H3O', 'bulk_OH', 'surface_OH', 'surface_H', 'Na', 'Cl']
        missing_ions = [ion for ion in expected_ions if ion not in found_ions]

        print(f"\nData loading summary:")
        print(f"  Ion types successfully read: {', '.join(found_ions) if found_ions else 'None'}")
        if missing_ions:
            print(f"  Ion types without data: {', '.join(missing_ions)}")
            if 'Na' in missing_ions or 'Cl' in missing_ions:
                print(f"  Note: The system may not contain NaCl ions")

        return all_ion_data, time_grouped_data, len(time_dirs)
    
    def read_bubble_centroids(self, time_dirs, centroids_subpath="centroids_density_2"):
        """Read bubble centroid data across all time ranges to obtain BubbleSize evolution."""
        all_bubble_data = []  # [(frame_index, time_ns, bubble_size, time_period_idx), ...]
        
        for i, (start_time, end_time, dirname, dir_path) in enumerate(time_dirs):
            centroids_file = os.path.join(dir_path, f"{centroids_subpath}/bubble_centroids.txt")
            
            if not os.path.exists(centroids_file):
                print(f"Warning: {centroids_file} does not exist, skipping")
                continue

            print(f"Reading bubble data {i+1}/{len(time_dirs)}: {dirname}")

            try:
                # Load the bubble_centroids.txt file
                data = np.loadtxt(centroids_file, skiprows=1)  # Skip header comments

                if data.size == 0:
                    print(f"    Warning: {centroids_file} contains no data")
                    continue

                if len(data.shape) == 1:
                    data = data.reshape(1, -1)

                # Extract frame index, time, and bubble size data
                frame_indices = data[:, 0]  # FrameIndex column
                times_ps_column = data[:, 1]  # Time(ps) column (actually frame count × 1000)
                bubble_sizes = data[:, 5]  # BubbleSize column

                # Compute the real time: starting time + frames × 1 ps
                # Each subdirectory starts frames from 1, 1 ps per frame
                times_ns = start_time + (frame_indices - 1) / 1000.0  # Convert to ns

                print(f"    Successfully read: {len(frame_indices)} time points")
                print(f"    Frame range: {frame_indices[0]:.0f} - {frame_indices[-1]:.0f}")
                print(f"    Time range: {times_ns[0]:.3f} - {times_ns[-1]:.3f} ns")
                print(f"    Bubble size range: {bubble_sizes[0]/2:.0f} - {bubble_sizes[-1]/2:.0f} N2 molecules")

                # Add to the aggregated data, recording the time-range index and absolute frame number
                for frame_idx, time_ns, bubble_size in zip(frame_indices, times_ns, bubble_sizes):
                    # Compute the absolute frame number across all time ranges
                    absolute_frame = int(time_ns * 1000)  # Time (ns) * 1000 = absolute frame index
                    all_bubble_data.append((absolute_frame, time_ns, bubble_size, i))

            except Exception as e:
                print(f"    Error reading {centroids_file}: {e}")

        if not all_bubble_data:
            print("Error: No bubble data were read")
            return None

        # Sort by time
        all_bubble_data.sort(key=lambda x: x[1])  # Sort by time in ns

        print(f"\nBubble data summary:")
        print(f"  Total number of time points: {len(all_bubble_data)}")
        print(f"  Time range: {all_bubble_data[0][1]:.3f} - {all_bubble_data[-1][1]:.3f} ns")
        print(f"  Frame range: {all_bubble_data[0][0]} - {all_bubble_data[-1][0]}")
        print(f"  Initial bubble size: {all_bubble_data[0][2]/2:.0f} $\\mathrm{{N_2}}$ molecules")
        print(f"  Final bubble size: {all_bubble_data[-1][2]/2:.0f} $\\mathrm{{N_2}}$ molecules")

        # Count the number of points per time range
        period_stats = defaultdict(int)
        for _, _, _, period_idx in all_bubble_data:
            period_stats[period_idx] += 1

        print(f"  Data points per time range:")
        for period_idx in sorted(period_stats.keys()):
            if period_idx < len(time_dirs):
                dirname = time_dirs[period_idx][2]
                print(f"    {dirname}: {period_stats[period_idx]} data points")

        return all_bubble_data

    def save_plot_data(self, filename, headers, data_dict):
        """
        Save plot data to a text file.
        :param filename: Output file name
        :param headers: Header comments (list of strings)
        :param data_dict: Dictionary containing plot data, e.g., {'x_axis_label': x_data, 'y1_label': y1_data, ...}
        """
        try:
            with open(filename, 'w') as f:
                for header in headers:
                    f.write(f"# {header}\n")
                
                column_labels = list(data_dict.keys())
                f.write("# " + "\t".join(column_labels) + "\n")
                
                # Retrieve data columns and transpose
                columns = list(data_dict.values())
                if not all(isinstance(c, np.ndarray) for c in columns):
                     print(f"Warning: not all data columns in {filename} are numpy arrays, skipping save.")
                     return

                # Ensure all columns share the same length, using the shortest as reference
                min_len = min(len(col) for col in columns if col is not None)
                
                for i in range(min_len):
                    line_parts = []
                    for col in columns:
                        if col is not None and i < len(col):
                            line_parts.append(f"{col[i]:.6e}")
                        else:
                            line_parts.append("NaN")
                    f.write("\t".join(line_parts) + "\n")
            print(f"Plot data saved: {filename}")
        except Exception as e:
            print(f"Error saving plot data to {filename}: {e}")

    def plot_bubble_evolution(self, bubble_data, args):
        """Plot the evolution of bubble size over time."""
        if not bubble_data:
            print("No bubble data available to plot the evolution curve")
            return None

        self.setup_nature_style()

        # Extract data
        frame_indices = [item[0] for item in bubble_data]
        times_ns = [item[1] for item in bubble_data]
        bubble_sizes = [item[2] / 2 for item in bubble_data]  # Divide by 2 to convert to molecule count

        # Compute the percentage relative to the initial size
        initial_size = bubble_sizes[0]
        bubble_percentages = [(size / initial_size) * 100 for size in bubble_sizes]

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Upper plot: absolute size
        ax1.plot(times_ns, bubble_sizes, 'b-', linewidth=2, label=r'$\mathrm{N_2}$ molecules')
        ax1.set_xlabel('Time (ns)', fontsize=14)
        ax1.set_ylabel(r'Bubble Size ($\mathrm{N_2}$ molecules)', fontsize=14)
        ax1.set_title('Bubble Size Evolution', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=12)
        
        # Lower plot: percentage
        ax2.plot(times_ns, bubble_percentages, 'r-', linewidth=2, label='Bubble Size (%)')
        ax2.set_xlabel('Time (ns)', fontsize=14)
        ax2.set_ylabel('Bubble Size (%)', fontsize=14)
        ax2.set_title('Bubble Size Evolution (Relative to Initial)', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=12)
        
        plt.tight_layout()
        
        # Save the figure
        plot_file = os.path.join(self.output_dir, "bubble_size_evolution.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Bubble evolution plot saved: {plot_file}")

        # Save evolution data
        data_file = os.path.join(self.output_dir, "data_bubble_evolution.txt")
        with open(data_file, 'w') as f:
            f.write("# Bubble size evolution data\n")
            f.write("# Format: FrameIndex Time(ns) BubbleSize(N2_molecules) Percentage(%) TimePeriod\n")
            f.write("# Note: Time = FrameIndex / 1000.0 (1ps per frame converted to ns)\n")
            for item in bubble_data:
                frame_idx, time_ns, bubble_size_raw, period_idx = item
                bubble_size_molecules = bubble_size_raw / 2  # Divide by 2 to convert to molecule count
                percentage = (bubble_size_molecules / initial_size) * 100
                f.write(f"{frame_idx}\t{time_ns:.3f}\t{bubble_size_molecules:.0f}\t{percentage:.2f}\t{period_idx}\n")

        print(f"Bubble evolution data saved: {data_file}")

        return bubble_data

    def plot_timerange_ion_distributions(self, all_ion_data, time_range_ns, args):
        """Plot ion distance distributions within a specified time range."""
        print(f"Generating ion distribution plots for {time_range_ns[0]}-{time_range_ns[1]} ns...")

        filtered_ion_data = defaultdict(list)

        # Build a mapping from time-range indices to actual ranges
        if not hasattr(self, '_time_dirs'):
            print("Error: time_dirs is not initialized; cannot filter by time range")
            return
            
        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    # Check whether the time range overlaps with the requested window
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))

        if not filtered_ion_data:
            print(f"Warning: No ion data found within {time_range_ns[0]}-{time_range_ns[1]} ns")
            return

        self.setup_nature_style()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        
        bins = args.bins
        
        total_selected = sum(len(data) for data in filtered_ion_data.values())
        print(f"Ion data points within {time_range_ns[0]}-{time_range_ns[1]} ns: {total_selected}")
        
        all_centroids_filtered = []
        for ion_data in filtered_ion_data.values():
            if ion_data:
                all_centroids_filtered.extend([item[0] for item in ion_data])
        avg_bubble_radius = np.mean(all_centroids_filtered) if all_centroids_filtered else 10.0

        # Data storage
        data_centroid_prob, data_interface_prob = {}, {}
        data_centroid_vol, data_interface_vol = {}, {}

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            print(f"  {ion_type}: {len(data_list)} data points")

            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]

            style = self.ion_styles.get(ion_type, {'color': 'black', 'marker': 'o', 'label': ion_type})
            ion_label_safe = style['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')

            # === Probability-normalized distributions ===
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, density=False)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            hist_centroid_norm = hist_centroid / np.sum(hist_centroid) * 100
            
            ax1.plot(bin_centers_centroid, hist_centroid_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, density=False)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            hist_interface_norm = hist_interface / np.sum(hist_interface) * 100
            
            ax2.plot(bin_centers_interface, hist_interface_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # === Volume-normalized distributions ===
            bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
            shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
            shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
            hist_centroid_vol = hist_centroid_norm / shell_volumes_centroid * 1000  # %/nm³
            
            ax3.plot(bin_centers_centroid, hist_centroid_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
            effective_radii = avg_bubble_radius + bin_centers_interface
            shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
            shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
            hist_interface_vol = hist_interface_norm / shell_volumes_interface * 1000  # %/nm³
            
            ax4.plot(bin_centers_interface, hist_interface_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # Store data for saving
            if 'd_centroid(A)' not in data_centroid_prob:
                data_centroid_prob['d_centroid(A)'] = bin_centers_centroid
                data_centroid_vol['d_centroid(A)'] = bin_centers_centroid
            if 'd_interface(A)' not in data_interface_prob:
                data_interface_prob['d_interface(A)'] = bin_centers_interface
                data_interface_vol['d_interface(A)'] = bin_centers_interface

            data_centroid_prob[f'Prob_{ion_label_safe}(%)'] = hist_centroid_norm
            data_interface_prob[f'Prob_{ion_label_safe}(%)'] = hist_interface_norm
            data_centroid_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_centroid_vol
            data_interface_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_interface_vol

        # Configure subplots
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Probability (%)', fontsize=14)
        ax1.set_title(f'Ion Distance to Bubble Centroid\n(Density Normalized, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Probability (%)', fontsize=14)
        ax2.set_title(f'Ion Distance to Bubble Surface\n(Density Normalized, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax3.set_title(f'Ion Distance to Bubble Centroid\n(Volume Normalized)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax4.set_title(f'Ion Distance to Bubble Surface\n(Volume Normalized)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"ion_distributions_normalized_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Normalized ion distribution plots saved for {time_range_ns[0]}-{time_range_ns[1]} ns: {plot_file}")

        # Save data to files
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_prob_{time_str}.txt"), [f"Ion dist to centroid (Density Norm) for {time_str}"], data_centroid_prob)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_prob_{time_str}.txt"), [f"Ion dist to interface (Density Norm) for {time_str}"], data_interface_prob)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_vol_{time_str}.txt"), [f"Ion dist to centroid (Volume Norm) for {time_str}"], data_centroid_vol)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_vol_{time_str}.txt"), [f"Ion dist to interface (Volume Norm) for {time_str}"], data_interface_vol)

    def plot_timerange_ion_distributions_absolute(self, all_ion_data, time_range_ns, args):
        """Plot ion distance distributions for a specific time range (absolute counts, non-normalized)."""
        print(f"Generating ion distribution plots for {time_range_ns[0]}-{time_range_ns[1]} ns (absolute counts)...")
        
        filtered_ion_data = defaultdict(list)
        
        # Build a mapping from time-range indices to actual ranges
        if not hasattr(self, '_time_dirs'):
            print("Error: time_dirs is not initialized; cannot filter by time range")
            return
            
        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    # Check whether the time range overlaps with the requested window
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))

        if not filtered_ion_data:
            print(f"Warning: No ion data found within {time_range_ns[0]}-{time_range_ns[1]} ns")
            return
        
        self.setup_nature_style()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        
        bins = args.bins
        
        total_selected = sum(len(data) for data in filtered_ion_data.values())
        print(f"Ion data points within {time_range_ns[0]}-{time_range_ns[1]} ns: {total_selected}")
        
        # Estimate the total number of frames within the time range (assuming 1 ps per frame)
        total_frames_in_range = 0
        for period_idx in range(len(self._time_dirs)):
            start_time, end_time, _, _ = self._time_dirs[period_idx]
            # Check whether the time range overlaps with the requested window
            if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                # Compute the duration of the overlap
                overlap_start = max(start_time, time_range_ns[0])
                overlap_end = min(end_time, time_range_ns[1])
                overlap_duration_ns = overlap_end - overlap_start
                frames_in_overlap = int(overlap_duration_ns * 1000)  # 1 ps per frame
                total_frames_in_range += frames_in_overlap

        print(f"Estimated frames within the time range: {total_frames_in_range}")
        
        all_centroids_filtered = []
        for ion_data in filtered_ion_data.values():
            if ion_data:
                all_centroids_filtered.extend([item[0] for item in ion_data])
        avg_bubble_radius = np.mean(all_centroids_filtered) if all_centroids_filtered else 10.0

        # Data storage
        data_centroid_num, data_interface_num = {}, {}
        data_centroid_density, data_interface_density = {}, {}

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            print(f"  {ion_type}: {len(data_list)} data points")
            
            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]
            
            style = self.ion_styles.get(ion_type, {'color': 'black', 'marker': 'o', 'label': ion_type})
            ion_label_safe = style['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')
            
            # === Absolute count distributions (per-frame averages) ===
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, density=False)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            # Divide by frame count to obtain the per-frame average
            hist_centroid_per_frame = hist_centroid / max(total_frames_in_range, 1)
            
            ax1.plot(bin_centers_centroid, hist_centroid_per_frame,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, density=False)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            # Divide by frame count to obtain the per-frame average
            hist_interface_per_frame = hist_interface / max(total_frames_in_range, 1)
            
            ax2.plot(bin_centers_interface, hist_interface_per_frame,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # === Density distributions (per-frame average per volume) ===
            bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
            shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
            shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
            hist_centroid_density = hist_centroid_per_frame / shell_volumes_centroid * 1000  # num per frame/nm³
            
            ax3.plot(bin_centers_centroid, hist_centroid_density,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
            effective_radii = avg_bubble_radius + bin_centers_interface
            shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
            shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
            hist_interface_density = hist_interface_per_frame / shell_volumes_interface * 1000  # num per frame/nm³
            
            ax4.plot(bin_centers_interface, hist_interface_density,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # Store data for saving
            if 'd_centroid(A)' not in data_centroid_num:
                data_centroid_num['d_centroid(A)'] = bin_centers_centroid
                data_centroid_density['d_centroid(A)'] = bin_centers_centroid
            if 'd_interface(A)' not in data_interface_num:
                data_interface_num['d_interface(A)'] = bin_centers_interface
                data_interface_density['d_interface(A)'] = bin_centers_interface

            data_centroid_num[f'Count_{ion_label_safe}(num)'] = hist_centroid
            data_interface_num[f'Count_{ion_label_safe}(num)'] = hist_interface
            data_centroid_density[f'Density_{ion_label_safe}(num/nm^3)'] = hist_centroid_density
            data_interface_density[f'Density_{ion_label_safe}(num/nm^3)'] = hist_interface_density

        # Configure subplots
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1.set_title(f'Ion Count vs Distance to Bubble Centroid\n(Absolute Count, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2.set_title(f'Ion Count vs Distance to Bubble Surface\n(Absolute Count, Time: {time_range_ns[0]}-{time_range_ns[1]} ns)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Ion Density (num / nm$^3$)', fontsize=14)
        ax3.set_title(f'Ion Density vs Distance to Bubble Centroid\n(Volume Normalized)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Ion Density (num / nm$^3$)', fontsize=14)
        ax4.set_title(f'Ion Density vs Distance to Bubble Surface\n(Volume Normalized)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"ion_distributions_absolute_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Absolute ion distribution plots saved for {time_range_ns[0]}-{time_range_ns[1]} ns: {plot_file}")

        # Save data to files
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_absolute_{time_str}.txt"), [f"Ion dist to centroid (Absolute Count) for {time_str}"], data_centroid_num)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_absolute_{time_str}.txt"), [f"Ion dist to interface (Absolute Count) for {time_str}"], data_interface_num)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_centroid_density_{time_str}.txt"), [f"Ion dist to centroid (Density) for {time_str}"], data_centroid_density)
        self.save_plot_data(os.path.join(self.output_dir, f"data_ion_dist_interface_density_{time_str}.txt"), [f"Ion dist to interface (Density) for {time_str}"], data_interface_density)

    def plot_comprehensive_distributions(self, all_ion_data, total_time, args):
        """Plot ion distance distributions aggregated across all time ranges."""
        if not all_ion_data:
            print("No ion data available to plot comprehensive distributions")
            return
        
        self.setup_nature_style()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        
        bins = args.bins
        
        all_centroids = []
        for ion_data in all_ion_data.values():
            if ion_data:
                all_centroids.extend([item[0] for item in ion_data])
        global_avg_bubble_radius = np.mean(all_centroids) if all_centroids else 10.0
        
        # Data storage
        data_centroid_prob, data_interface_prob = {}, {}
        data_centroid_vol, data_interface_vol = {}, {}

        for ion_type, data_list in all_ion_data.items():
            if not data_list:
                continue
            
            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]
            
            style = self.ion_styles.get(ion_type, {'color': 'black', 'marker': 'o', 'label': ion_type})
            ion_label_safe = style['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')

            # === Probability normalization ===
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, density=False)
            bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
            hist_centroid_norm = hist_centroid / np.sum(hist_centroid) * 100
            
            ax1.plot(bin_centers_centroid, hist_centroid_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, density=False)
            bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
            hist_interface_norm = hist_interface / np.sum(hist_interface) * 100
            
            ax2.plot(bin_centers_interface, hist_interface_norm,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # === Volume normalization ===
            bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
            shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
            shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
            hist_centroid_vol = hist_centroid_norm / shell_volumes_centroid * 1000
            
            ax3.plot(bin_centers_centroid, hist_centroid_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
            effective_radii = global_avg_bubble_radius + bin_centers_interface
            shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
            shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
            hist_interface_vol = hist_interface_norm / shell_volumes_interface * 1000
            
            ax4.plot(bin_centers_interface, hist_interface_vol,
                     marker=style['marker'], color=style['color'],
                     linewidth=2.0, markersize=4, alpha=0.8, label=style['label'])
            
            # Store data for saving
            if 'd_centroid(A)' not in data_centroid_prob:
                data_centroid_prob['d_centroid(A)'] = bin_centers_centroid
                data_centroid_vol['d_centroid(A)'] = bin_centers_centroid
            if 'd_interface(A)' not in data_interface_prob:
                data_interface_prob['d_interface(A)'] = bin_centers_interface
                data_interface_vol['d_interface(A)'] = bin_centers_interface

            data_centroid_prob[f'Prob_{ion_label_safe}(%)'] = hist_centroid_norm
            data_interface_prob[f'Prob_{ion_label_safe}(%)'] = hist_interface_norm
            data_centroid_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_centroid_vol
            data_interface_vol[f'ProbVol_{ion_label_safe}(%/nm^3)'] = hist_interface_vol
        
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Probability (%)', fontsize=14)
        ax1.set_title(f'Ion Distance to Bubble Centroid\n(Density Normalized, Total time: {total_time:.1f}ns)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Probability (%)', fontsize=14)
        ax2.set_title(f'Ion Distance to Bubble Surface\n(Density Normalized, Total time: {total_time:.1f}ns)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax3.set_title(f'Ion Distance to Bubble Centroid\n(Volume Normalized)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
        ax4.set_title(f'Ion Distance to Bubble Surface\n(Volume Normalized, Equiv. Sphere)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, "comprehensive_ion_distributions_normalized.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Normalized comprehensive distribution plots saved: {plot_file}")

        # Save data to files
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_centroid_prob.txt"), [f"Comprehensive ion dist to centroid (Density Norm)"], data_centroid_prob)
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_interface_prob.txt"), [f"Comprehensive ion dist to interface (Density Norm)"], data_interface_prob)
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_centroid_vol.txt"), [f"Comprehensive ion dist to centroid (Volume Norm)"], data_centroid_vol)
        self.save_plot_data(os.path.join(self.output_dir, "data_comprehensive_dist_interface_vol.txt"), [f"Comprehensive ion dist to interface (Volume Norm)"], data_interface_vol)

    def plot_time_grouped_distributions(self, time_grouped_data, num_time_periods, time_dirs, args):
        """Plot ion distance distributions grouped by time (four segments, colors from light to dark)."""
        if not time_grouped_data:
            print("No ion data available to plot time-grouped distributions")
            return
        
        self.setup_nature_style()
        
        group_size = max(1, num_time_periods // 4)
        groups = [list(range(i, min(i + group_size, num_time_periods))) for i in range(0, num_time_periods, group_size)]
        
        if len(groups) > 4 and len(groups[-1]) < group_size / 2:
            groups[-2].extend(groups.pop())
        
        print(f"Dividing {num_time_periods} time ranges into {len(groups)} groups:")
        for i, group in enumerate(groups):
            time_ranges = [f"{time_dirs[j][2]}" for j in group]
            print(f"  Group {i+1}: {', '.join(time_ranges)}")
        
        all_ion_types = sorted({key for time_data in time_grouped_data.values() for key in time_data.keys()})
        
        if not all_ion_types:
            print("No ion data found")
            return
        
        for ion_type in all_ion_types:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_evolution[0], args.figsize_evolution[1]*1.5))
            
            base_color = self.ion_styles.get(ion_type, {'color': '#1f77b4'})['color']
            import matplotlib.colors as mcolors
            color_alphas = np.linspace(0.4, 1.0, len(groups))
            
            bins = args.bins
            
            # Data storage
            ion_label_safe = self.ion_styles.get(ion_type, {'label': ion_type})['label'].replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '').replace('^', '').replace('_', '').replace('-', '(surf)')
            data_centroid_prob, data_interface_prob = {}, {}
            data_centroid_vol, data_interface_vol = {}, {}

            for group_idx, time_indices in enumerate(groups):
                group_d_centroids = []
                group_d_interfaces = []
                
                for time_idx in time_indices:
                    if time_idx in time_grouped_data and ion_type in time_grouped_data[time_idx]:
                        data_list = time_grouped_data[time_idx][ion_type]
                        group_d_centroids.extend([item[0] for item in data_list])
                        group_d_interfaces.extend([item[1] for item in data_list])
                
                if not group_d_centroids:
                    continue
                
                alpha = color_alphas[group_idx]
                color = mcolors.to_rgba(base_color, alpha)
                
                start_time = time_dirs[time_indices[0]][0]
                end_time = time_dirs[time_indices[-1]][1]
                label = f"Time {group_idx+1}: {start_time:.1f}-{end_time:.1f} ns"
                label_safe = label.replace(' ', '_').replace(':', '').replace('-', '_to_')

                # === Probability normalization ===
                hist_centroid, bin_edges_centroid = np.histogram(group_d_centroids, bins=bins, density=False)
                bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
                hist_centroid_norm = hist_centroid / np.sum(hist_centroid) * 100
                ax1.plot(bin_centers_centroid, hist_centroid_norm, color=color, linewidth=2.5, label=label)
                
                hist_interface, bin_edges_interface = np.histogram(group_d_interfaces, bins=bins, density=False)
                bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
                hist_interface_norm = hist_interface / np.sum(hist_interface) * 100
                ax2.plot(bin_centers_interface, hist_interface_norm, color=color, linewidth=2.5, label=label)
                
                # === Volume normalization ===
                bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
                shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
                shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
                hist_centroid_vol = hist_centroid_norm / shell_volumes_centroid * 1000
                ax3.plot(bin_centers_centroid, hist_centroid_vol, color=color, linewidth=2.5, label=label)
                
                bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
                avg_bubble_radius = np.mean(group_d_centroids)
                effective_radii = avg_bubble_radius + bin_centers_interface
                shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
                shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
                hist_interface_vol = hist_interface_norm / shell_volumes_interface * 1000
                ax4.plot(bin_centers_interface, hist_interface_vol, color=color, linewidth=2.5, label=label)

                # Store data
                if 'd_centroid(A)' not in data_centroid_prob:
                    data_centroid_prob['d_centroid(A)'] = bin_centers_centroid
                    data_centroid_vol['d_centroid(A)'] = bin_centers_centroid
                if 'd_interface(A)' not in data_interface_prob:
                    data_interface_prob['d_interface(A)'] = bin_centers_interface
                    data_interface_vol['d_interface(A)'] = bin_centers_interface
                
                data_centroid_prob[f'Prob_{label_safe}(%)'] = hist_centroid_norm
                data_interface_prob[f'Prob_{label_safe}(%)'] = hist_interface_norm
                data_centroid_vol[f'ProbVol_{label_safe}(%/nm^3)'] = hist_centroid_vol
                data_interface_vol[f'ProbVol_{label_safe}(%/nm^3)'] = hist_interface_vol

            ion_label = self.ion_styles.get(ion_type, {'label': ion_type})['label']
            
            ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
            ax1.set_ylabel('Probability (%)', fontsize=14)
            ax1.set_title(f'{ion_label} Distance to Bubble Centroid\n(Density Normalized, Time Evolution)', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(frameon=False, fontsize=9)
            
            ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
            ax2.set_ylabel('Probability (%)', fontsize=14)
            ax2.set_title(f'{ion_label} Distance to Bubble Surface\n(Density Normalized, Time Evolution)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend(frameon=False, fontsize=9)
            
            ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
            ax3.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
            ax3.set_title(f'{ion_label} Distance to Bubble Centroid\n(Volume Normalized, Time Evolution)', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.legend(frameon=False, fontsize=9)
            
            ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
            ax4.set_ylabel(r'Probability (% / nm$^3$)', fontsize=14)
            ax4.set_title(f'{ion_label} Distance to Bubble Surface\n(Volume Normalized, Time Evolution)', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend(frameon=False, fontsize=9)
            
            plt.tight_layout()
            
            plot_file = os.path.join(self.output_dir, f"time_evolution_normalized_{ion_type}_distributions.png")
            plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Normalized time-evolution distribution saved for {ion_type}: {plot_file}")

            # Save data
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_centroid_prob.txt"), [f"Time evolution for {ion_label_safe} dist to centroid (Density Norm)"], data_centroid_prob)
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_interface_prob.txt"), [f"Time evolution for {ion_label_safe} dist to interface (Density Norm)"], data_interface_prob)
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_centroid_vol.txt"), [f"Time evolution for {ion_label_safe} dist to centroid (Volume Norm)"], data_centroid_vol)
            self.save_plot_data(os.path.join(self.output_dir, f"data_time_evolution_{ion_type}_interface_vol.txt"), [f"Time evolution for {ion_label_safe} dist to interface (Volume Norm)"], data_interface_vol)

    def plot_charge_distributions(self, all_ion_data, time_range_ns, args):
        """Plot cumulative charge distributions within a specified time range."""
        print(f"Generating charge distribution plots for {time_range_ns[0]}-{time_range_ns[1]} ns...")

        filtered_ion_data = defaultdict(list)
        if not hasattr(self, '_time_dirs'):
            print("Error: time_dirs is not initialized; cannot filter by time range")
            return

        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))

        if not filtered_ion_data:
            print(f"Warning: No ion data found within {time_range_ns[0]}-{time_range_ns[1]} ns")
            return

        # Estimate the total number of frames within the time range (assuming 1 ps per frame)
        total_frames_in_range = 0
        for period_idx in range(len(self._time_dirs)):
            start_time, end_time, _, _ = self._time_dirs[period_idx]
            # Check whether the time range overlaps with the requested window
            if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                # Compute the duration of the overlap
                overlap_start = max(start_time, time_range_ns[0])
                overlap_end = min(end_time, time_range_ns[1])
                overlap_duration_ns = overlap_end - overlap_start
                frames_in_overlap = int(overlap_duration_ns * 1000)  # 1 ps per frame
                total_frames_in_range += frames_in_overlap

        print(f"Estimated frames within the time range: {total_frames_in_range}")

        self.setup_nature_style()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        bins = args.bins
        
        # Determine the data range to apply consistent binning
        all_d_centroids = [d[0] for data in filtered_ion_data.values() for d in data]
        all_d_interfaces = [d[1] for data in filtered_ion_data.values() for d in data]
        if not all_d_centroids:
            print("Warning: No data points after filtering; cannot generate charge distribution plot")
            return
            
        range_centroid = (np.min(all_d_centroids), np.max(all_d_centroids))
        range_interface = (np.min(all_d_interfaces), np.max(all_d_interfaces))

        # Initialize charge histograms
        total_charge_centroid = np.zeros(bins)
        total_charge_interface = np.zeros(bins)
        bulk_only_charge_centroid = np.zeros(bins)
        bulk_only_charge_interface = np.zeros(bins)

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            charge = self.ion_charges.get(ion_type, 0)
            if charge == 0:
                continue

            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]

            # Compute ion histograms, multiply by charge, then divide by total frames to obtain per-frame charge
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, range=range_centroid)
            charge_per_frame_centroid = hist_centroid * charge / max(total_frames_in_range, 1)
            total_charge_centroid += charge_per_frame_centroid
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, range=range_interface)
            charge_per_frame_interface = hist_interface * charge / max(total_frames_in_range, 1)
            total_charge_interface += charge_per_frame_interface

            # Compute charges considering bulk ions only
            if ion_type not in ['surface_H', 'surface_OH']:
                bulk_only_charge_centroid += charge_per_frame_centroid
                bulk_only_charge_interface += charge_per_frame_interface

        # Plot non-volume-normalized curves
        bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
        ax1.plot(bin_centers_centroid, total_charge_centroid, color='k', linewidth=2.5, label='All Ions')
        ax1.plot(bin_centers_centroid, bulk_only_charge_centroid, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')

        bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
        ax2.plot(bin_centers_interface, total_charge_interface, color='k', linewidth=2.5, label='All Ions')
        ax2.plot(bin_centers_interface, bulk_only_charge_interface, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')

        # Plot volume-normalized curves
        bin_width_centroid = bin_edges_centroid[1] - bin_edges_centroid[0]
        shell_volumes_centroid = 4 * np.pi * bin_centers_centroid**2 * bin_width_centroid
        shell_volumes_centroid = np.maximum(shell_volumes_centroid, 1e-10)
        charge_density_centroid = total_charge_centroid / shell_volumes_centroid
        charge_density_centroid_bulk = bulk_only_charge_centroid / shell_volumes_centroid
        ax3.plot(bin_centers_centroid, charge_density_centroid, color='k', linewidth=2.5, label='All Ions')
        ax3.plot(bin_centers_centroid, charge_density_centroid_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')

        avg_bubble_radius = np.mean(all_d_centroids)
        bin_width_interface = bin_edges_interface[1] - bin_edges_interface[0]
        effective_radii = avg_bubble_radius + bin_centers_interface
        shell_volumes_interface = 4 * np.pi * effective_radii**2 * bin_width_interface
        shell_volumes_interface = np.maximum(shell_volumes_interface, 1e-10)
        charge_density_interface = total_charge_interface / shell_volumes_interface
        charge_density_interface_bulk = bulk_only_charge_interface / shell_volumes_interface
        ax4.plot(bin_centers_interface, charge_density_interface, color='k', linewidth=2.5, label='All Ions')
        ax4.plot(bin_centers_interface, charge_density_interface_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')

        # Configure subplots
        title_suffix = f"\n(Time: {time_range_ns[0]}-{time_range_ns[1]} ns)"
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Net Charge per Frame (e)', fontsize=14)
        ax1.set_title('Net Charge per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Net Charge per Frame (e)', fontsize=14)
        ax2.set_title('Net Charge per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax3.set_title('Charge Density per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax4.set_title('Charge Density per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"charge_distributions_normalized_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Charge distribution plots saved for {time_range_ns[0]}-{time_range_ns[1]} ns: {plot_file}")

        # Save data
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        data_to_save_centroid = {
            'd_centroid(A)': bin_centers_centroid,
            'NetCharge_All_per_frame(e)': total_charge_centroid,
            'NetCharge_Bulk_per_frame(e)': bulk_only_charge_centroid,
            'ChargeDensity_All_per_frame(e/A^3)': charge_density_centroid,
            'ChargeDensity_Bulk_per_frame(e/A^3)': charge_density_centroid_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_charge_dist_centroid_{time_str}.txt"), [f"Charge dist vs d_centroid for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}"], data_to_save_centroid)

        data_to_save_interface = {
            'd_interface(A)': bin_centers_interface,
            'NetCharge_All_per_frame(e)': total_charge_interface,
            'NetCharge_Bulk_per_frame(e)': bulk_only_charge_interface,
            'ChargeDensity_All_per_frame(e/A^3)': charge_density_interface,
            'ChargeDensity_Bulk_per_frame(e/A^3)': charge_density_interface_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_charge_dist_interface_{time_str}.txt"), [f"Charge dist vs d_interface for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}"], data_to_save_interface)

    def plot_cumulative_charge_distributions(self, all_ion_data, time_range_ns, args):
        """Plot cumulative charge distributions (integrated from the minimum distance to the current position)."""
        print(f"Generating cumulative charge distribution plots for {time_range_ns[0]}-{time_range_ns[1]} ns...")

        filtered_ion_data = defaultdict(list)
        if not hasattr(self, '_time_dirs'):
            print("Error: time_dirs is not initialized; cannot filter by time range")
            return

        for ion_type, data_list in all_ion_data.items():
            for d_centroid, d_interface, period_idx in data_list:
                if period_idx < len(self._time_dirs):
                    start_time, end_time, _, _ = self._time_dirs[period_idx]
                    if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                        filtered_ion_data[ion_type].append((d_centroid, d_interface))

        if not filtered_ion_data:
            print(f"Warning: No ion data found within {time_range_ns[0]}-{time_range_ns[1]} ns")
            return

        # Estimate the total number of frames within the time range (assuming 1 ps per frame)
        total_frames_in_range = 0
        for period_idx in range(len(self._time_dirs)):
            start_time, end_time, _, _ = self._time_dirs[period_idx]
            # Check whether the time range overlaps with the requested window
            if not (end_time < time_range_ns[0] or start_time > time_range_ns[1]):
                # Compute the duration of the overlap
                overlap_start = max(start_time, time_range_ns[0])
                overlap_end = min(end_time, time_range_ns[1])
                overlap_duration_ns = overlap_end - overlap_start
                frames_in_overlap = int(overlap_duration_ns * 1000)  # 1 ps per frame
                total_frames_in_range += frames_in_overlap
        
        print(f"Estimated frames within the time range: {total_frames_in_range}")

        self.setup_nature_style()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(args.figsize_comprehensive[0], args.figsize_comprehensive[1]*1.5))
        bins = args.bins
        
        # Determine the data range to apply consistent binning
        all_d_centroids = [d[0] for data in filtered_ion_data.values() for d in data]
        all_d_interfaces = [d[1] for data in filtered_ion_data.values() for d in data]
        if not all_d_centroids:
            print("Warning: No data points after filtering; cannot generate cumulative charge plot")
            return
            
        range_centroid = (np.min(all_d_centroids), np.max(all_d_centroids))
        range_interface = (np.min(all_d_interfaces), np.max(all_d_interfaces))

        # Initialize charge histograms
        total_charge_centroid = np.zeros(bins)
        total_charge_interface = np.zeros(bins)
        bulk_only_charge_centroid = np.zeros(bins)
        bulk_only_charge_interface = np.zeros(bins)

        for ion_type, data_list in filtered_ion_data.items():
            if not data_list:
                continue
            
            charge = self.ion_charges.get(ion_type, 0)
            if charge == 0:
                continue

            d_centroids = [item[0] for item in data_list]
            d_interfaces = [item[1] for item in data_list]

            # Compute ion histograms, multiply by charge, then divide by total frames to obtain per-frame charge
            hist_centroid, bin_edges_centroid = np.histogram(d_centroids, bins=bins, range=range_centroid)
            charge_per_frame_centroid = hist_centroid * charge / max(total_frames_in_range, 1)
            total_charge_centroid += charge_per_frame_centroid
            
            hist_interface, bin_edges_interface = np.histogram(d_interfaces, bins=bins, range=range_interface)
            charge_per_frame_interface = hist_interface * charge / max(total_frames_in_range, 1)
            total_charge_interface += charge_per_frame_interface

            # Compute charges considering bulk ions only
            if ion_type not in ['surface_H', 'surface_OH']:
                bulk_only_charge_centroid += charge_per_frame_centroid
                bulk_only_charge_interface += charge_per_frame_interface

        # Calculate cumulative charge (integrated from the minimum distance to the current position)
        cumulative_charge_centroid_all = np.cumsum(total_charge_centroid)
        cumulative_charge_centroid_bulk = np.cumsum(bulk_only_charge_centroid)
        cumulative_charge_interface_all = np.cumsum(total_charge_interface)
        cumulative_charge_interface_bulk = np.cumsum(bulk_only_charge_interface)

        # Plot cumulative charge without volume normalization
        bin_centers_centroid = (bin_edges_centroid[:-1] + bin_edges_centroid[1:]) / 2
        ax1.plot(bin_centers_centroid, cumulative_charge_centroid_all, color='k', linewidth=2.5, label='All Ions')
        ax1.plot(bin_centers_centroid, cumulative_charge_centroid_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
        
        bin_centers_interface = (bin_edges_interface[:-1] + bin_edges_interface[1:]) / 2
        ax2.plot(bin_centers_interface, cumulative_charge_interface_all, color='k', linewidth=2.5, label='All Ions')
        ax2.plot(bin_centers_interface, cumulative_charge_interface_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

        # Calculate cumulative spherical volume from the center to the current radius
        # For d_centroid: sphere volume = 4/3 * π * r^3
        cumulative_volumes_centroid = (4.0/3.0) * np.pi * bin_centers_centroid**3
        cumulative_volumes_centroid = np.maximum(cumulative_volumes_centroid, 1e-10)
        cumulative_charge_density_centroid_all = cumulative_charge_centroid_all / cumulative_volumes_centroid
        cumulative_charge_density_centroid_bulk = cumulative_charge_centroid_bulk / cumulative_volumes_centroid
        
        ax3.plot(bin_centers_centroid, cumulative_charge_density_centroid_all, color='k', linewidth=2.5, label='All Ions')
        ax3.plot(bin_centers_centroid, cumulative_charge_density_centroid_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

        # For d_interface: volume from the bubble surface to the current distance (spherical shell from R to R+d)
        avg_bubble_radius = np.mean(all_d_centroids)
        # Cumulative volume = 4/3*π*[(R+d)^3 - R^3]
        outer_radii = avg_bubble_radius + bin_centers_interface
        cumulative_volumes_interface = (4.0/3.0) * np.pi * (outer_radii**3 - avg_bubble_radius**3)
        cumulative_volumes_interface = np.maximum(cumulative_volumes_interface, 1e-10)
        cumulative_charge_density_interface_all = cumulative_charge_interface_all / cumulative_volumes_interface
        cumulative_charge_density_interface_bulk = cumulative_charge_interface_bulk / cumulative_volumes_interface
        
        ax4.plot(bin_centers_interface, cumulative_charge_density_interface_all, color='k', linewidth=2.5, label='All Ions')
        ax4.plot(bin_centers_interface, cumulative_charge_density_interface_bulk, color='r', linestyle='--', linewidth=2.0, label='Bulk Ions Only')
        ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

        # Configure subplots
        title_suffix = f"\n(Time: {time_range_ns[0]}-{time_range_ns[1]} ns)"
        ax1.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax1.set_ylabel('Cumulative Net Charge per Frame (e)', fontsize=14)
        ax1.set_title('Cumulative Net Charge per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10)
        
        ax2.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax2.set_ylabel('Cumulative Net Charge per Frame (e)', fontsize=14)
        ax2.set_title('Cumulative Net Charge per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10)
        
        ax3.set_xlabel(r'$d_{\mathrm{centroid}}$ (Å)', fontsize=14)
        ax3.set_ylabel(r'Cumulative Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax3.set_title('Cumulative Charge Density per Frame vs. Distance to Centroid' + title_suffix, fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, fontsize=10)
        
        ax4.set_xlabel(r'$d_{\mathrm{interface}}$ (Å)', fontsize=14)
        ax4.set_ylabel(r'Cumulative Charge Density per Frame (e / Å$^3$)', fontsize=14)
        ax4.set_title('Cumulative Charge Density per Frame vs. Distance to Surface' + title_suffix, fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"cumulative_charge_distributions_{time_range_ns[0]}-{time_range_ns[1]}ns.png")
        plt.savefig(plot_file, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Cumulative charge distribution plots saved for {time_range_ns[0]}-{time_range_ns[1]} ns: {plot_file}")

        # Save data
        time_str = f"{time_range_ns[0]}-{time_range_ns[1]}ns"
        data_to_save_centroid = {
            'd_centroid(A)': bin_centers_centroid,
            'CumulativeCharge_All_per_frame(e)': cumulative_charge_centroid_all,
            'CumulativeCharge_Bulk_per_frame(e)': cumulative_charge_centroid_bulk,
            'CumulativeChargeDensity_All_per_frame(e/A^3)': cumulative_charge_density_centroid_all,
            'CumulativeChargeDensity_Bulk_per_frame(e/A^3)': cumulative_charge_density_centroid_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_cumulative_charge_dist_centroid_{time_str}.txt"), [f"Cumulative charge dist vs d_centroid for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}", f"Note: Cumulative from minimum distance to current distance"], data_to_save_centroid)

        data_to_save_interface = {
            'd_interface(A)': bin_centers_interface,
            'CumulativeCharge_All_per_frame(e)': cumulative_charge_interface_all,
            'CumulativeCharge_Bulk_per_frame(e)': cumulative_charge_interface_bulk,
            'CumulativeChargeDensity_All_per_frame(e/A^3)': cumulative_charge_density_interface_all,
            'CumulativeChargeDensity_Bulk_per_frame(e/A^3)': cumulative_charge_density_interface_bulk,
        }
        self.save_plot_data(os.path.join(self.output_dir, f"data_cumulative_charge_dist_interface_{time_str}.txt"), [f"Cumulative charge dist vs d_interface for {time_str} (per frame average)", f"Total frames in range: {total_frames_in_range}", f"Average bubble radius: {avg_bubble_radius:.2f} A", f"Note: Cumulative from bubble surface to current distance"], data_to_save_interface)

    def run_analysis(self, args=None):
        """Run the complete batch analysis."""
        if args is None:
            args = type('DefaultArgs', (), {
                'bins': 50, 'dpi': 300,
                'figsize_comprehensive': [20, 8],
                'figsize_evolution': [20, 8],
                'figsize_combined': [24, 10]
            })()
        
        print("Starting batch ion distance distribution analysis...")
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        time_dirs, total_time = self.find_time_directories()
        if not time_dirs:
            print("Error: No time-range directories were found")
            return
        
        self._time_dirs = time_dirs # Store for later use
        
        print("="*60)
        
        print("Reading bubble centroid data...")
        bubble_data = self.read_bubble_centroids(time_dirs, centroids_subpath=args.centroids_subpath)
        
        print("="*60)

        print("Reading ion distance data...")
        all_ion_data, time_grouped_data, num_time_periods = self.read_ion_distances(time_dirs, centroids_subpath=args.centroids_subpath)
        
        if not all_ion_data:
            print("Error: No ion data were read")
            # Even without ion data, bubble data may still need processing
            if bubble_data:
                print("="*60)
                print("Analyzing bubble size evolution...")
                self.plot_bubble_evolution(bubble_data, args)
            return
        
        print("="*60)
        
        print("Data summary:")
        total_points = sum(len(data) for data in all_ion_data.values())
        for ion_type, data_list in all_ion_data.items():
            print(f"  {ion_type}: {len(data_list)} data points")
        print(f"Total data points: {total_points}")
        print("="*60)
        
        print("Generating comprehensive distribution plots...")
        self.plot_comprehensive_distributions(all_ion_data, total_time, args)
        
        print("="*60)
        
        print("Generating time-evolution distribution plots...")
        self.plot_time_grouped_distributions(time_grouped_data, num_time_periods, time_dirs, args)
        
        print("="*60)

        if bubble_data:
            print("Analyzing bubble size evolution...")
            self.plot_bubble_evolution(bubble_data, args)
        
        print("="*60)
        
        # Generate ion and charge distribution plots for the specified time ranges
        time_range_for_specific_plots = (args.time_range_start, args.time_range_end)
        self.plot_timerange_ion_distributions(all_ion_data, time_range_for_specific_plots, args)
        self.plot_timerange_ion_distributions_absolute(all_ion_data, time_range_for_specific_plots, args)
        self.plot_charge_distributions(all_ion_data, time_range_for_specific_plots, args)
        self.plot_cumulative_charge_distributions(all_ion_data, time_range_for_specific_plots, args)

        print("="*60)
        print("Batch analysis complete!")
        print(f"Output files saved in: {self.output_dir}")
        print("\nGenerated file types:")
        print("  Image files (.png):")
        print("  - comprehensive_ion_distributions_normalized.png: Ion distributions across all time ranges")
        print("  - time_evolution_normalized_[ion_type]_distributions.png: Time-evolution distributions for each ion type")
        print(f"  - ion_distributions_normalized_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: Ion distributions for the specified time range (normalized)")
        print(f"  - ion_distributions_absolute_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: Ion distributions for the specified time range (absolute counts)")
        print(f"  - charge_distributions_normalized_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: Charge distributions per bin")
        print(f"  - cumulative_charge_distributions_{time_range_for_specific_plots[0]}-{time_range_for_specific_plots[1]}ns.png: Cumulative charge distributions (integrated from the minimum distance)")
        if bubble_data:
            print("  - bubble_size_evolution.png: Bubble size evolution plot")
        
        print("\n  Data files (.txt):")
        print("  - data_*.txt: Source data for each plot")

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Batch analysis script for ion distance distributions')
    
    parser.add_argument('--base_path', 
                        default='/home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4',
                        help='Base directory containing subdirectories for each time range')
    
    parser.add_argument('--output_dir', 
                        default=None,
                        help='Output directory path. Defaults to base_path/analysis_ion/centroids_density if omitted')
    
    parser.add_argument('--bins', 
                        type=int, 
                        default=50,
                        help='Number of histogram bins (default: 50)')
    
    parser.add_argument('--dpi', 
                        type=int, 
                        default=300,
                        help='Output image DPI (default: 300)')
    
    parser.add_argument('--figsize_comprehensive', 
                        nargs=2, 
                        type=float, 
                        default=[20, 8],
                        help='Figure size for comprehensive plots (width height) (default: 20 8)')
    
    parser.add_argument('--figsize_evolution', 
                        nargs=2, 
                        type=float, 
                        default=[20, 8],
                        help='Figure size for time-evolution plots (width height) (default: 20 8)')
    
    parser.add_argument('--figsize_combined', 
                        nargs=2, 
                        type=float, 
                        default=[24, 10],
                        help='Figure size for combined time-evolution plots (width height) (default: 24 10)')
    
    parser.add_argument('--centroids_subpath',
                        default='centroids_density_2',
                        help='Subpath to the bubble centroid files (default: centroids_density_2)')
    
    parser.add_argument('--time_range_start',
                        type=float,
                        default=2.0,
                        help='Start time (ns) for specific-range plots (default: 2.0)')
    
    parser.add_argument('--time_range_end',
                        type=float,
                        default=5.0,
                        help='End time (ns) for specific-range plots (default: 5.0)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = get_args()
    
    if not os.path.isdir(args.base_path):
        print(f"Error: base directory {args.base_path} is not valid")
        sys.exit(1)
    
    analyzer = BatchIonAnalyzer(base_dir=args.base_path, output_dir=args.output_dir)
    
    print("="*60)
    print("Batch ion distance distribution analysis")
    print("="*60)
    print(f"Input directory: {args.base_path}")
    print(f"Output directory: {analyzer.output_dir}")
    print(f"Image parameters: DPI={args.dpi}, bins={args.bins}")
    print(f"Figure sizes: comprehensive={args.figsize_comprehensive}, evolution={args.figsize_evolution}, combined={args.figsize_combined}")
    print("="*60)
    
    analyzer.run_analysis(args)

if __name__ == "__main__":
    main()