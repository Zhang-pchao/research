import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import MDAnalysis as mda
import argparse
import pickle

# Setup logging
def setup_logging(enable_log_file=False):
    """Setup logging configuration"""
    log_handlers = [logging.StreamHandler()]
    
    if enable_log_file:
        log_file = 'h2o_orientation_analysis.log'
        if os.path.exists(log_file):
            os.remove(log_file)
        log_handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze H2O dipole orientation relative to z-axis')
    parser.add_argument('--input', default='../model_atomic.data', 
                       help='LAMMPS data file name')
    parser.add_argument('--traj', default='../bubble_1k.lammpstrj', 
                       help='LAMMPS trajectory file name')
    parser.add_argument('--atom_style', default='id type x y z', 
                       help='LAMMPS atom_style')
    parser.add_argument('--step_interval', type=int, default=100, 
                       help='Frame interval for analysis')
    parser.add_argument('--start_frame', type=int, default=0, 
                       help='Starting frame for analysis')
    parser.add_argument('--end_frame', type=int, default=-1, 
                       help='Ending frame for analysis, -1 for last frame')
    parser.add_argument('--oh_cutoff', type=float, default=1.4, 
                       help='O-H bond distance cutoff (Å)')
    parser.add_argument('--ti_o_cutoff', type=float, default=3.5, 
                       help='Ti-O distance cutoff for surface detection (Å)')
    parser.add_argument('--z_bin_size', type=float, default=5.0, 
                       help='Z-coordinate bin size (Å)')
    parser.add_argument('--has_tio2', action='store_true', 
                       help='Whether the system contains TiO2 (if yes, only analyze H2O above TiO2 surface)')
    parser.add_argument('--enable_log_file', action='store_true', 
                       help='Enable log file output')
    parser.add_argument('--output_dir', default='h2o_orientation_results', 
                       help='Output directory for results')
    
    return parser.parse_args()

def apply_pbc(positions, box):
    """Apply periodic boundary conditions to generate all image positions"""
    offsets = np.array([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
                       [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                       [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                       [0, -1, -1], [0, -1, 0], [0, -1, 1],
                       [0, 0, -1], [0, 0, 0], [0, 0, 1],
                       [0, 1, -1], [0, 1, 0], [0, 1, 1],
                       [1, -1, -1], [1, -1, 0], [1, -1, 1],
                       [1, 0, -1], [1, 0, 0], [1, 0, 1],
                       [1, 1, -1], [1, 1, 0], [1, 1, 1]])
    
    extended_positions = []
    original_indices = []
    
    for i, pos in enumerate(positions):
        for offset in offsets:
            extended_pos = pos + offset * box
            extended_positions.append(extended_pos)
            original_indices.append(i)
    
    return np.array(extended_positions), np.array(original_indices)

def find_h2o_molecules_kdtree(h_positions, o_positions, box, oh_cutoff=1.4):
    """Find H2O molecules using KDTree for efficient distance calculation"""
    
    # Create extended periodic images for O atoms
    extended_o_positions, o_original_indices = apply_pbc(o_positions, box)
    
    # Build KDTree
    tree = cKDTree(extended_o_positions)
    
    # Build O-H bonding relationship
    o_h_bonds = defaultdict(list)
    h_bonded_to_o = set()
    
    for h_idx, h_pos in enumerate(h_positions):
        # Find all O atoms within cutoff distance
        indices = tree.query_ball_point(h_pos, oh_cutoff)
        
        if indices:
            # Calculate actual distances and find nearest
            distances = []
            for idx in indices:
                extended_o_pos = extended_o_positions[idx]
                dist = np.linalg.norm(h_pos - extended_o_pos)
                distances.append((dist, o_original_indices[idx]))
            
            # Find nearest O atom
            min_dist, nearest_o_idx = min(distances)
            if min_dist <= oh_cutoff:
                o_h_bonds[nearest_o_idx].append(h_idx)
                h_bonded_to_o.add(h_idx)
    
    # Count H atoms for each O atom
    o_h_counts = {o_idx: len(h_list) for o_idx, h_list in o_h_bonds.items()}
    
    return o_h_counts, h_bonded_to_o, o_h_bonds

def calculate_tio2_surface_z(o_positions, ti_positions, box, ti_o_cutoff=3.5):
    """Calculate average z-coordinate of TiO2 surface adsorbed O atoms"""
    
    # Find top layer Ti atoms
    ti_z_coords = ti_positions[:, 2]
    max_ti_z = np.max(ti_z_coords)
    
    # Ti atoms with z-coordinate within 2 Å of max are considered top layer
    top_ti_mask = np.abs(ti_z_coords - max_ti_z) < 2
    top_ti_positions = ti_positions[top_ti_mask]
    
    logging.debug(f"Found {len(top_ti_positions)} top layer Ti atoms at z: {max_ti_z:.3f}")
    
    # Find surface O atoms bonded to top layer Ti
    surface_o_z_coords = []
    
    for o_pos in o_positions:
        for ti_pos in top_ti_positions:
            dist = np.linalg.norm(o_pos - ti_pos)
            if dist <= ti_o_cutoff:
                surface_o_z_coords.append(o_pos[2])
                break
    
    if surface_o_z_coords:
        avg_z = np.mean(surface_o_z_coords)
        logging.info(f"Found {len(surface_o_z_coords)} surface O atoms, average z: {avg_z:.3f} Å")
        return avg_z
    else:
        logging.warning("No surface O atoms found")
        return None

def calculate_h2o_dipole_vector(o_pos, h_positions):
    """
    Calculate H2O dipole moment vector
    Dipole vector points from negative to positive, i.e., from O to the midpoint of two H atoms
    This is a simplified approximation; the actual dipole includes charge distribution
    """
    if len(h_positions) != 2:
        return None
    
    # Midpoint of two H atoms
    h_midpoint = np.mean(h_positions, axis=0)
    
    # Dipole vector: from O to H midpoint
    dipole_vector = h_midpoint - o_pos
    
    # Normalize
    dipole_norm = np.linalg.norm(dipole_vector)
    if dipole_norm < 1e-6:
        return None
    
    dipole_unit = dipole_vector / dipole_norm
    
    return dipole_unit

def read_lammps_frame(u, frame_idx):
    """Read specified frame from LAMMPS trajectory"""
    try:
        u.trajectory[frame_idx]
        
        positions = u.atoms.positions
        atom_types = u.atoms.types
        
        box_info = u.dimensions
        box_dims = box_info[:3]
        
        # Map LAMMPS atom types to chemical elements
        type_to_element = {
            '1': 'H',   
            '2': 'O',     
            '3': 'N',  
            '4': 'Na',   
            '5': 'Cl',
            '6': 'Ti',
        }
        
        symbols = [type_to_element.get(str(t), 'X') for t in atom_types]
        
        class LAMMPSAtoms:
            def __init__(self, positions, symbols, box_dims, box_info):
                self.positions = positions
                self.symbols = symbols
                self.box = box_dims
                self.box_info = box_info
                
            def get_positions(self):
                return self.positions
                
            def get_chemical_symbols(self):
                return self.symbols
        
        atoms = LAMMPSAtoms(positions, symbols, box_dims, box_info)
        
        return atoms, box_dims, box_info
        
    except Exception as e:
        logging.error(f"Error reading frame {frame_idx}: {str(e)}")
        return None, None, None

def analyze_frame_h2o_orientation(atoms, box_dims, frame_idx, args, surface_z=None):
    """
    Analyze H2O dipole orientation for a single frame
    Returns: dictionary with z-binned angle data
    """
    try:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        # Get atom indices
        o_indices = [i for i, s in enumerate(symbols) if s == "O"]
        h_indices = [i for i, s in enumerate(symbols) if s == "H"]
        ti_indices = [i for i, s in enumerate(symbols) if s == "Ti"]
        
        if len(o_indices) == 0 or len(h_indices) == 0:
            logging.warning(f"Frame {frame_idx}: Missing required atom types")
            return None
        
        # Separate TiO2 O from solution O
        if args.has_tio2 and len(ti_indices) > 0:
            min_ti_index = min(ti_indices)
            max_ti_index = max(ti_indices)
            # Solution O atoms are those with index > max Ti index
            solution_o_indices = [i for i in o_indices if i > max_ti_index]
            
            if surface_z is None:
                # Calculate surface z if not provided
                ti_positions = positions[ti_indices]
                o_positions = positions[o_indices]
                surface_z = calculate_tio2_surface_z(o_positions, ti_positions, box_dims, args.ti_o_cutoff)
        else:
            # No TiO2, use all O atoms
            solution_o_indices = o_indices
            surface_z = None
        
        o_positions = positions[o_indices]
        h_positions = positions[h_indices]
        solution_o_positions = positions[solution_o_indices]
        
        # Find H2O molecules
        o_h_counts, h_bonded_to_o, o_h_bonds = find_h2o_molecules_kdtree(
            h_positions, o_positions, box_dims, args.oh_cutoff
        )
        
        # Collect H2O orientation data
        h2o_data = []
        
        for i, o_idx in enumerate(o_indices):
            # Only process solution O atoms
            if o_idx not in solution_o_indices:
                continue
            
            h_count = o_h_counts.get(i, 0)
            
            # Only H2O molecules (2 H atoms)
            if h_count == 2:
                o_pos = positions[o_idx]
                
                # If TiO2 system, only analyze H2O above surface
                if surface_z is not None and o_pos[2] <= surface_z:
                    continue
                
                # Get H positions bonded to this O
                if i in o_h_bonds and len(o_h_bonds[i]) == 2:
                    h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                    
                    # Calculate dipole vector
                    dipole_vector = calculate_h2o_dipole_vector(o_pos, h_coords)
                    
                    if dipole_vector is not None:
                        # Calculate angle with z-axis
                        z_unit = np.array([0, 0, 1])
                        cos_theta = np.dot(dipole_vector, z_unit)
                        
                        # Clamp to [-1, 1] to avoid numerical errors
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        
                        # Calculate angle in degrees
                        theta = np.arccos(cos_theta) * 180.0 / np.pi
                        
                        # Store data
                        z_coord = o_pos[2] - (surface_z if surface_z is not None else 0)
                        h2o_data.append({
                            'z': z_coord,
                            'theta': theta,
                            'cos_theta': cos_theta
                        })
        
        logging.info(f"Frame {frame_idx}: Found {len(h2o_data)} H2O molecules for orientation analysis")
        
        return {
            'frame': frame_idx,
            'surface_z': surface_z,
            'h2o_data': h2o_data
        }
        
    except Exception as e:
        logging.error(f"Error analyzing frame {frame_idx}: {str(e)}")
        return None

def bin_h2o_by_z(all_frame_results, z_bin_size=5.0):
    """
    Bin H2O molecules by z-coordinate and collect orientation data
    """
    # Collect all z-coordinates to determine range
    all_z = []
    for result in all_frame_results:
        for h2o in result['h2o_data']:
            all_z.append(h2o['z'])
    
    if len(all_z) == 0:
        logging.warning("No H2O data found for binning")
        return None
    
    z_min = np.min(all_z)
    z_max = np.max(all_z)
    
    # Create bins
    bins = np.arange(z_min, z_max + z_bin_size, z_bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Initialize bin data as regular dict (not defaultdict with lambda)
    binned_data = {}
    
    # Assign each H2O to a bin
    for result in all_frame_results:
        for h2o in result['h2o_data']:
            z = h2o['z']
            bin_idx = np.digitize(z, bins) - 1
            
            if 0 <= bin_idx < len(bin_centers):
                if bin_idx not in binned_data:
                    binned_data[bin_idx] = {'theta': [], 'cos_theta': []}
                binned_data[bin_idx]['theta'].append(h2o['theta'])
                binned_data[bin_idx]['cos_theta'].append(h2o['cos_theta'])
    
    logging.info(f"Binned H2O data into {len(binned_data)} z-bins")
    
    return {
        'bins': bins,
        'bin_centers': bin_centers,
        'binned_data': binned_data
    }

def plot_orientation_distribution(binned_results, output_dir):
    """
    Plot H2O orientation distribution as violin plots
    """
    bins = binned_results['bins']
    bin_centers = binned_results['bin_centers']
    binned_data = binned_results['binned_data']
    
    # Setup plot style
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
    })
    
    # Create figure with two subplots: one for cos(theta), one for theta
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Prepare data for violin plot
    cos_theta_data = []
    theta_data = []
    positions = []
    
    for i in range(len(bin_centers)):
        if i in binned_data:
            cos_theta_data.append(binned_data[i]['cos_theta'])
            theta_data.append(binned_data[i]['theta'])
            positions.append(bin_centers[i])
    
    if len(cos_theta_data) == 0:
        logging.warning("No data available for plotting")
        return
    
    # Plot 1: cos(theta) distribution
    parts1 = ax1.violinplot(cos_theta_data, positions=positions, widths=bins[1]-bins[0]*0.8,
                            showmeans=True, showmedians=True)
    
    # Customize violin plot colors
    for pc in parts1['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.7)
    
    ax1.set_xlabel('Z Coordinate (Å)', fontsize=14)
    ax1.set_ylabel(r'cos($\theta$)', fontsize=14)
    ax1.set_title(r'H$_2$O Dipole Orientation: cos($\theta$) vs Z', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add reference lines for cos(theta)
    ax1.axhline(y=1, color='r', linestyle=':', linewidth=1, alpha=0.3, label='Pointing up')
    ax1.axhline(y=-1, color='b', linestyle=':', linewidth=1, alpha=0.3, label='Pointing down')
    ax1.legend(loc='best')
    
    # Plot 2: theta distribution
    parts2 = ax2.violinplot(theta_data, positions=positions, widths=bins[1]-bins[0]*0.8,
                            showmeans=True, showmedians=True)
    
    for pc in parts2['bodies']:
        pc.set_facecolor('#ff7f0e')
        pc.set_alpha(0.7)
    
    ax2.set_xlabel('Z Coordinate (Å)', fontsize=14)
    ax2.set_ylabel(r'$\theta$ (degrees)', fontsize=14)
    ax2.set_title(r'H$_2$O Dipole Orientation: Angle $\theta$ vs Z', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=90, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perpendicular')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'h2o_orientation_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Orientation distribution plot saved: {plot_file}")

def save_analysis_data(binned_results, all_frame_results, output_dir):
    """
    Save analysis data to files
    """
    # Save binned statistics
    stats_file = os.path.join(output_dir, 'orientation_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Z_Bin_Center(Å)\tN_H2O\tMean_cos_theta\tStd_cos_theta\tMean_theta(deg)\tStd_theta(deg)\n")
        
        bin_centers = binned_results['bin_centers']
        binned_data = binned_results['binned_data']
        
        for i in range(len(bin_centers)):
            if i in binned_data:
                cos_theta_list = binned_data[i]['cos_theta']
                theta_list = binned_data[i]['theta']
                
                n_h2o = len(cos_theta_list)
                mean_cos = np.mean(cos_theta_list)
                std_cos = np.std(cos_theta_list)
                mean_theta = np.mean(theta_list)
                std_theta = np.std(theta_list)
                
                f.write(f"{bin_centers[i]:.3f}\t{n_h2o}\t{mean_cos:.6f}\t{std_cos:.6f}\t"
                       f"{mean_theta:.3f}\t{std_theta:.3f}\n")
    
    logging.info(f"Statistics saved: {stats_file}")
    
    # Save frame-by-frame summary
    frame_summary_file = os.path.join(output_dir, 'frame_summary.txt')
    with open(frame_summary_file, 'w') as f:
        f.write("Frame\tSurface_Z(Å)\tN_H2O\tMean_cos_theta\tMean_theta(deg)\n")
        
        for result in all_frame_results:
            frame = result['frame']
            surface_z = result['surface_z'] if result['surface_z'] is not None else 0.0
            h2o_data = result['h2o_data']
            
            if len(h2o_data) > 0:
                cos_theta_list = [h2o['cos_theta'] for h2o in h2o_data]
                theta_list = [h2o['theta'] for h2o in h2o_data]
                
                n_h2o = len(h2o_data)
                mean_cos = np.mean(cos_theta_list)
                mean_theta = np.mean(theta_list)
                
                f.write(f"{frame}\t{surface_z:.3f}\t{n_h2o}\t{mean_cos:.6f}\t{mean_theta:.3f}\n")
    
    logging.info(f"Frame summary saved: {frame_summary_file}")
    
    # Save raw binned data as pickle for further analysis
    pickle_file = os.path.join(output_dir, 'binned_data.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(binned_results, f)
    
    logging.info(f"Binned data saved: {pickle_file}")

def main():
    args = get_args()
    
    # Setup logging
    setup_logging(enable_log_file=args.enable_log_file)
    
    start_time = datetime.now()
    logging.info(f"Starting H2O orientation analysis, frame interval: {args.step_interval}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Read LAMMPS files
        logging.info(f"Reading LAMMPS files: {args.input}, {args.traj}")
        
        u = mda.Universe(args.input, args.traj, 
                        atom_style=args.atom_style,
                        format='LAMMPSDUMP')
        
        total_frames = len(u.trajectory)
        logging.info(f"Total frames in trajectory: {total_frames}")
        
        # Determine frame range
        start_frame = args.start_frame
        end_frame = total_frames if args.end_frame == -1 else min(args.end_frame, total_frames)
        
        frames_to_analyze = list(range(start_frame, end_frame, args.step_interval))
        logging.info(f"Will analyze {len(frames_to_analyze)} frames")
        
        # Calculate surface_z from first frame if TiO2 system
        surface_z = None
        if args.has_tio2:
            logging.info("TiO2 system detected, calculating surface O z-coordinate...")
            atoms_first, box_dims_first, _ = read_lammps_frame(u, frames_to_analyze[0])
            if atoms_first is not None:
                symbols = atoms_first.get_chemical_symbols()
                positions = atoms_first.get_positions()
                o_indices = [i for i, s in enumerate(symbols) if s == "O"]
                ti_indices = [i for i, s in enumerate(symbols) if s == "Ti"]
                
                if len(ti_indices) > 0 and len(o_indices) > 0:
                    ti_positions = positions[ti_indices]
                    o_positions = positions[o_indices]
                    surface_z = calculate_tio2_surface_z(o_positions, ti_positions, 
                                                        box_dims_first, args.ti_o_cutoff)
        
        all_frame_results = []
        
        # Analyze each frame
        for frame_idx in frames_to_analyze:
            logging.info(f"Analyzing frame {frame_idx}...")
            
            atoms, box_dims, box_info = read_lammps_frame(u, frame_idx)
            if atoms is None:
                continue
            
            result = analyze_frame_h2o_orientation(atoms, box_dims, frame_idx, args, surface_z)
            
            if result is not None:
                all_frame_results.append(result)
        
        if len(all_frame_results) == 0:
            logging.error("No valid frames analyzed")
            return
        
        logging.info(f"Successfully analyzed {len(all_frame_results)} frames")
        
        # Bin H2O by z-coordinate
        logging.info("Binning H2O molecules by z-coordinate...")
        binned_results = bin_h2o_by_z(all_frame_results, args.z_bin_size)
        
        if binned_results is None:
            logging.error("Failed to bin H2O data")
            return
        
        # Save analysis data
        save_analysis_data(binned_results, all_frame_results, args.output_dir)
        
        # Plot orientation distribution
        plot_orientation_distribution(binned_results, args.output_dir)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\nAnalysis completed! Total time: {duration}")
        logging.info(f"Results saved in directory: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()


