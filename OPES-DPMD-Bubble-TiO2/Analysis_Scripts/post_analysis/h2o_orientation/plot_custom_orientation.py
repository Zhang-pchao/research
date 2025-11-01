import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import argparse
import os
from scipy import stats

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create custom plots from saved H2O orientation data')
    parser.add_argument('--data_file', default='h2o_orientation_results/binned_data.pkl',
                       help='Path to binned data pickle file')
    parser.add_argument('--output_dir', default='custom_plots',
                       help='Output directory for custom plots')
    parser.add_argument('--plot_type', choices=['violin', 'box', 'mean', 'histogram', 'all'],
                       default='all', help='Type of plot to generate')
    parser.add_argument('--metric', choices=['cos_theta', 'theta', 'both'],
                       default='both', help='Which metric to plot')
    parser.add_argument('--n_z_bins', type=int, default=5,
                       help='Number of lowest z bins to plot in cos(theta) distribution (default: 4)')
    
    return parser.parse_args()

def setup_plot_style():
    """Setup Nature-style matplotlib parameters"""
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
    })

def plot_violin(binned_results, output_dir, metric='cos_theta'):
    """Generate violin plot"""
    bins = binned_results['bins']
    bin_centers = binned_results['bin_centers']
    binned_data = binned_results['binned_data']
    
    data_list = []
    positions = []
    
    for i in range(len(bin_centers)):
        if i in binned_data:
            data_list.append(binned_data[i][metric])
            positions.append(bin_centers[i])
    
    if len(data_list) == 0:
        print("No data available for violin plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    parts = ax.violinplot(data_list, positions=positions, widths=bins[1]-bins[0]*0.8,
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.7)
    
    ax.set_xlabel('$\Delta$Z (Å)', fontsize=14)
    ax.set_xlim(right=65)
    
    if metric == 'cos_theta':
        ax.set_ylabel(r'cos($\theta$)', fontsize=14)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=1, color='r', linestyle=':', linewidth=1, alpha=0.3, label='Pointing up')
        ax.axhline(y=-1, color='b', linestyle=':', linewidth=1, alpha=0.3, label='Pointing down')
    else:
        ax.set_ylabel(r'$\theta$ (degrees)', fontsize=14)
        ax.axhline(y=90, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perpendicular')
    
    ax.legend(loc='best', frameon=False)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'violin_{metric}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Violin plot saved: {output_file}")

def plot_box(binned_results, output_dir, metric='cos_theta'):
    """Generate box plot"""
    bins = binned_results['bins']
    bin_centers = binned_results['bin_centers']
    binned_data = binned_results['binned_data']
    
    data_list = []
    positions = []
    
    for i in range(len(bin_centers)):
        if i in binned_data:
            data_list.append(binned_data[i][metric])
            positions.append(bin_centers[i])
    
    if len(data_list) == 0:
        print("No data available for box plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(data_list, positions=positions, widths=bins[1]-bins[0]*0.6,
                    patch_artist=True, showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#ff7f0e')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('$\Delta$Z (Å)', fontsize=14)
    ax.set_xlim(right=65)
    
    if metric == 'cos_theta':
        ax.set_ylabel(r'cos($\theta$)', fontsize=14)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    else:
        ax.set_ylabel(r'$\theta$ (degrees)', fontsize=14)
        ax.axhline(y=90, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'box_{metric}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Box plot saved: {output_file}")

def plot_mean_with_error(binned_results, output_dir, metric='cos_theta'):
    """Generate mean with error bars plot"""
    bins = binned_results['bins']
    bin_centers = binned_results['bin_centers']
    binned_data = binned_results['binned_data']
    
    positions = []
    means = []
    stds = []
    
    for i in range(len(bin_centers)):
        if i in binned_data:
            data = binned_data[i][metric]
            positions.append(bin_centers[i])
            means.append(np.mean(data))
            stds.append(np.std(data))
    
    if len(positions) == 0:
        print("No data available for mean plot")
        return
    
    positions = np.array(positions)
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.errorbar(positions, means, yerr=stds, fmt='o-', capsize=5,
                linewidth=2, markersize=8, label='Mean ± Std')
    
    ax.fill_between(positions, means - stds, means + stds, alpha=0.3)
    
    ax.set_xlabel('$\Delta$Z (Å)', fontsize=14)
    ax.set_xlim(right=65)
    
    if metric == 'cos_theta':
        ax.set_ylabel(r'cos($\theta$)', fontsize=14)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    else:
        ax.set_ylabel(r'$\theta$ (degrees)', fontsize=14)
        ax.axhline(y=90, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.legend(loc='best', frameon=False)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'mean_{metric}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Mean plot saved: {output_file}")

def plot_histogram_2d(binned_results, output_dir, metric='cos_theta'):
    """Generate 2D histogram showing distribution density"""
    binned_data = binned_results['binned_data']
    bin_centers = binned_results['bin_centers']
    
    # Collect all data points
    z_coords = []
    values = []
    
    for i in range(len(bin_centers)):
        if i in binned_data:
            n_points = len(binned_data[i][metric])
            z_coords.extend([bin_centers[i]] * n_points)
            values.extend(binned_data[i][metric])
    
    if len(z_coords) == 0:
        print("No data available for 2D histogram")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create 2D histogram
    if metric == 'cos_theta':
        bins_y = np.linspace(-1, 1, 50)
        ylabel = r'cos($\theta$)'
    else:
        bins_y = np.linspace(0, 180, 50)
        ylabel = r'$\theta$ (degrees)'
    
    bins_x = binned_results['bins']
    
    h, xedges, yedges, img = ax.hist2d(z_coords, values, bins=[bins_x, bins_y],
                                       cmap='viridis', cmin=1)
    
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Count', fontsize=12)
    
    ax.set_xlabel('$\Delta$Z (Å)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(right=65)
    
    if metric == 'cos_theta':
        ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    else:
        ax.axhline(y=90, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'histogram2d_{metric}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"2D histogram saved: {output_file}")

def plot_cos_theta_distribution_by_z(binned_results, output_dir, n_bins=4):
    """
    Plot cos(theta) distribution for the first n_bins (lowest z values)
    Shows how water orientation changes near the surface
    """
    bin_centers = binned_results['bin_centers']
    binned_data = binned_results['binned_data']
    
    # Get sorted bin indices (by z coordinate)
    sorted_bin_indices = sorted([i for i in range(len(bin_centers)) if i in binned_data],
                                key=lambda i: bin_centers[i])
    
    # Take only the first n_bins (lowest z values)
    selected_bins = sorted_bin_indices[:min(n_bins, len(sorted_bin_indices))]
    
    if len(selected_bins) == 0:
        print("No data available for cos(theta) distribution plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot distribution for each selected bin
    for idx, bin_idx in enumerate(selected_bins):
        cos_theta_data = np.array(binned_data[bin_idx]['cos_theta'])
        z_center = bin_centers[bin_idx]
        
        # Create histogram with normalized density
        counts, bin_edges = np.histogram(cos_theta_data, bins=50, range=(-1, 1), density=True)
        bin_centers_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot as line
        color = colors[idx % len(colors)]
        ax.plot(bin_centers_hist, counts, '-', linewidth=2.5, 
                color=color, label=f'$\Delta$Z = {z_center:.1f} Å', alpha=0.8)
    
    ax.set_xlabel(r'cos($\theta$)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_xlim(-1, 1)
    
    # Add reference lines
    #ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3, label='Perpendicular')
    
    # Legend
    ax.legend(loc='best', frameon=False, fontsize=14)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'cos_theta_distribution_by_z.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"cos(theta) distribution plot saved: {output_file}")

def plot_theta_distribution_by_z(binned_results, output_dir, n_bins=4):
    """
    Plot theta (angle in degrees) distribution for the first n_bins (lowest z values)
    Shows how water orientation changes near the surface
    """
    bin_centers = binned_results['bin_centers']
    binned_data = binned_results['binned_data']
    
    # Get sorted bin indices (by z coordinate)
    sorted_bin_indices = sorted([i for i in range(len(bin_centers)) if i in binned_data],
                                key=lambda i: bin_centers[i])
    
    # Take only the first n_bins (lowest z values)
    selected_bins = sorted_bin_indices[:min(n_bins, len(sorted_bin_indices))]
    
    if len(selected_bins) == 0:
        print("No data available for theta distribution plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot distribution for each selected bin
    for idx, bin_idx in enumerate(selected_bins):
        theta_data = np.array(binned_data[bin_idx]['theta'])
        z_center = bin_centers[bin_idx]
        
        # Create histogram with normalized density
        counts, bin_edges = np.histogram(theta_data, bins=50, range=(0, 180), density=True)
        bin_centers_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot as line
        color = colors[idx % len(colors)]
        ax.plot(bin_centers_hist, counts, '-', linewidth=2.5, 
                color=color, label=f'$\Delta$Z = {z_center:.1f} Å', alpha=0.8)
    
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_xlim(0, 180)
    
    # Add reference lines
    #ax.axvline(x=90, color='k', linestyle='--', linewidth=1, alpha=0.3, label='Perpendicular')
    
    # Legend
    ax.legend(loc='best', frameon=False, fontsize=14)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'theta_distribution_by_z.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"theta distribution plot saved: {output_file}")

def main():
    args = get_args()
    
    # Setup plot style
    setup_plot_style()
    
    # Load data
    print(f"Loading data from: {args.data_file}")
    with open(args.data_file, 'rb') as f:
        binned_results = pickle.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which metrics to plot
    metrics = []
    if args.metric in ['cos_theta', 'both']:
        metrics.append('cos_theta')
    if args.metric in ['theta', 'both']:
        metrics.append('theta')
    
    # Generate plots
    for metric in metrics:
        if args.plot_type in ['violin', 'all']:
            plot_violin(binned_results, args.output_dir, metric)
        
        if args.plot_type in ['box', 'all']:
            plot_box(binned_results, args.output_dir, metric)
        
        if args.plot_type in ['mean', 'all']:
            plot_mean_with_error(binned_results, args.output_dir, metric)
        
        if args.plot_type in ['histogram', 'all']:
            plot_histogram_2d(binned_results, args.output_dir, metric)
    
    # Always plot cos(theta) and theta distribution for lowest z bins
    plot_cos_theta_distribution_by_z(binned_results, args.output_dir, n_bins=args.n_z_bins)
    plot_theta_distribution_by_z(binned_results, args.output_dir, n_bins=args.n_z_bins)
    
    print(f"\nAll plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()


