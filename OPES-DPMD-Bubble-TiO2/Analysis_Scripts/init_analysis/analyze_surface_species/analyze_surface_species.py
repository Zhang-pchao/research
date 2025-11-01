import numpy as np
from ase.io import read, write
from scipy.spatial import cKDTree
from collections import Counter, defaultdict
import logging
from datetime import datetime
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import argparse

# Configure logging
log_file = 'analyze_surface_species.log'
if os.path.exists(log_file):
    os.remove(log_file)  # Remove existing log file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze TiO2 surface adsorbed species')
    parser.add_argument('--format', choices=['exyz', 'lammps'], default='exyz',
                       help='Input file format (exyz or lammps)')
    parser.add_argument('--input', default='dump.xyz', help='Input file name (exyz format) or LAMMPS data file')
    parser.add_argument('--traj', default=None, help='LAMMPS trajectory file name (required for lammps format)')
    parser.add_argument('--atom_style', default=None, help='LAMMPS atom_style (e.g., "id type x y z", "atomic", "full")')
    parser.add_argument('--output', default='dump_last_frame.xyz', help='File name for the extracted last frame')

    # Add cutoff parameters
    parser.add_argument('--ti_o_cutoff', type=float, default=2.5, help='Ti-O distance threshold (Å) for identifying surface adsorption')
    parser.add_argument('--oh_cutoff', type=float, default=1.4, help='O-H bond length threshold (Å)')
    parser.add_argument('--h_ti_cutoff', type=float, default=2.1, help='H-Ti bond length threshold (Å)')
    parser.add_argument('--h_h_cutoff', type=float, default=1.1, help='H-H bond length threshold (H2 molecule) (Å)')
    parser.add_argument('--o_o_cutoff', type=float, default=1.5, help='O-O distance threshold (Å) for detecting O-O neighbors')
    parser.add_argument('--n_n_cutoff', type=float, default=1.5, help='N-N distance threshold (Å) for detecting isolated N atoms')
    
    return parser.parse_args()

def parse_lattice_from_xyz(filename):
    """Extract lattice information from the second line of an xyz file"""
    try:
        with open(filename, 'r') as f:
            f.readline()  # Skip the first line (number of atoms)
            header = f.readline().strip()  # Read the second line

        # Use a regular expression to extract lattice information
        lattice_match = re.search(r'Lattice="([^"]+)"', header)
        if lattice_match:
            lattice_str = lattice_match.group(1)
            lattice_values = list(map(float, lattice_str.split()))

            # Rearrange into a 3x3 matrix
            lattice_matrix = np.array(lattice_values).reshape(3, 3)
            # Extract diagonal elements as box dimensions
            box = np.array([lattice_matrix[0,0], lattice_matrix[1,1], lattice_matrix[2,2]])
            logging.info(f"Extracted lattice dimensions from file: {box}")
            return box
        else:
            logging.warning("Lattice information not found, using default values")
            return np.array([100.0, 100.0, 100.0])

    except Exception as e:
        logging.error(f"Failed to parse lattice information: {e}")
        return np.array([100.0, 100.0, 100.0])

def apply_pbc(positions, box):
    """Apply periodic boundary conditions to generate all images"""
    # Create image offsets (-1, 0, 1) in each direction
    offsets = np.array([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
                       [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                       [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                       [0, -1, -1], [0, -1, 0], [0, -1, 1],
                       [0, 0, -1], [0, 0, 0], [0, 0, 1],
                       [0, 1, -1], [0, 1, 0], [0, 1, 1],
                       [1, -1, -1], [1, -1, 0], [1, -1, 1],
                       [1, 0, -1], [1, 0, 0], [1, 0, 1],
                       [1, 1, -1], [1, 1, 0], [1, 1, 1]])

    # Create all images for each atom
    extended_positions = []
    original_indices = []
    
    for i, pos in enumerate(positions):
        for offset in offsets:
            extended_pos = pos + offset * box
            extended_positions.append(extended_pos)
            original_indices.append(i)
    
    return np.array(extended_positions), np.array(original_indices)

def find_nearest_oxygens_kdtree(h_positions, o_positions, box, oh_cutoff=1.35):
    """Use KDTree to accelerate finding the nearest O atoms"""
    logging.info("Computing H-O nearest neighbors with KDTree...")

    # Create extended periodic images for O atoms
    extended_o_positions, o_original_indices = apply_pbc(o_positions, box)

    # Build KDTree
    tree = cKDTree(extended_o_positions)

    # Find the nearest O atom for each H atom
    o_h_counts = defaultdict(int)
    h_bonded_to_o = set()  # Record indices of H atoms bonded to O

    for h_idx, h_pos in enumerate(h_positions):
        # Find all O atoms within the cutoff distance
        indices = tree.query_ball_point(h_pos, oh_cutoff)

        if indices:
            # Compute actual distances and find the nearest one
            distances = []
            for idx in indices:
                extended_o_pos = extended_o_positions[idx]
                dist = np.linalg.norm(h_pos - extended_o_pos)
                distances.append((dist, o_original_indices[idx]))

            # Identify the nearest O atom
            min_dist, nearest_o_idx = min(distances)
            if min_dist <= oh_cutoff:
                o_h_counts[nearest_o_idx] += 1
                h_bonded_to_o.add(h_idx)

    return o_h_counts, h_bonded_to_o

def find_h_ti_bonds_kdtree(h_positions, ti_positions, box, h_ti_cutoff=2.0):
    """Detect H-Ti bonds using KDTree"""
    logging.info("Computing H-Ti bonds with KDTree...")

    # Create extended periodic images for Ti atoms
    extended_ti_positions, ti_original_indices = apply_pbc(ti_positions, box)

    # Build KDTree
    tree = cKDTree(extended_ti_positions)

    h_bonded_to_ti = set()  # Record indices of H atoms bonded to Ti

    for h_idx, h_pos in enumerate(h_positions):
        # Find all Ti atoms within the cutoff distance
        indices = tree.query_ball_point(h_pos, h_ti_cutoff)

        if indices:
            # Check whether any Ti atoms fall within the cutoff
            for idx in indices:
                extended_ti_pos = extended_ti_positions[idx]
                dist = np.linalg.norm(h_pos - extended_ti_pos)
                if dist <= h_ti_cutoff:
                    h_bonded_to_ti.add(h_idx)
                    break

    return h_bonded_to_ti

def find_h2_molecules_kdtree(h_positions, box, h_h_cutoff=1.0):
    """Detect H2 molecules using KDTree"""
    logging.info("Computing H-H bonds (H2 molecules) with KDTree...")

    # Create extended periodic images for H atoms
    extended_h_positions, h_original_indices = apply_pbc(h_positions, box)

    # Build KDTree
    tree = cKDTree(extended_h_positions)

    h_bonded_to_h = set()  # Record indices of H atoms bonded to H
    h2_pairs = []  # Record H2 molecular pairs

    for h_idx, h_pos in enumerate(h_positions):
        if h_idx in h_bonded_to_h:
            continue  # Already marked as part of an H2 molecule

        # Find all H atoms within the cutoff distance
        indices = tree.query_ball_point(h_pos, h_h_cutoff)

        for idx in indices:
            partner_h_idx = h_original_indices[idx]
            # Exclude itself and ensure each H2 molecule is only counted once
            if partner_h_idx != h_idx and partner_h_idx not in h_bonded_to_h:
                extended_h_pos = extended_h_positions[idx]
                dist = np.linalg.norm(h_pos - extended_h_pos)
                if dist <= h_h_cutoff:
                    h_bonded_to_h.add(h_idx)
                    h_bonded_to_h.add(partner_h_idx)
                    h2_pairs.append((h_idx, partner_h_idx))
                    break
    
    return h_bonded_to_h, h2_pairs

def find_surface_oxygens_kdtree(o_positions, ti_positions, box, ti_o_cutoff=2.3):
    """Use KDTree to identify surface-adsorbed O atoms and distinguish between top and bottom surfaces"""
    logging.info("Computing Ti-O surface adsorption with KDTree...")

    # Create extended periodic images for Ti atoms
    extended_ti_positions, ti_original_indices = apply_pbc(ti_positions, box)

    # Build KDTree
    tree = cKDTree(extended_ti_positions)

    # Find the topmost and bottommost Ti atoms
    ti_z_coords = ti_positions[:, 2]
    max_ti_z = np.max(ti_z_coords)
    min_ti_z = np.min(ti_z_coords)

    # Treat Ti atoms with z-coordinate differences smaller than 2 as belonging to the same layer to account for numerical noise
    top_ti_mask = np.abs(ti_z_coords - max_ti_z) < 2
    bottom_ti_mask = np.abs(ti_z_coords - min_ti_z) < 2

    top_ti_positions = ti_positions[top_ti_mask]
    bottom_ti_positions = ti_positions[bottom_ti_mask]

    logging.info(f"Identified {len(top_ti_positions)} top-layer Ti atoms, z-coordinate: {max_ti_z:.3f}")
    logging.info(f"Identified {len(bottom_ti_positions)} bottom-layer Ti atoms, z-coordinate: {min_ti_z:.3f}")

    # Mark surface O atoms
    surface_o_mask = np.zeros(len(o_positions), dtype=bool)
    top_surface_mask = np.zeros(len(o_positions), dtype=bool)
    bottom_surface_mask = np.zeros(len(o_positions), dtype=bool)

    # Examine each O atom
    for o_idx, o_pos in enumerate(o_positions):
        # Check whether it belongs to the top surface
        for ti_pos in top_ti_positions:
            dist = np.linalg.norm(o_pos - ti_pos)
            if dist <= ti_o_cutoff:
                surface_o_mask[o_idx] = True
                top_surface_mask[o_idx] = True
                break

        # If not on the top surface, check whether it belongs to the bottom surface
        if not surface_o_mask[o_idx]:
            for ti_pos in bottom_ti_positions:
                dist = np.linalg.norm(o_pos - ti_pos)
                if dist <= ti_o_cutoff:
                    surface_o_mask[o_idx] = True
                    bottom_surface_mask[o_idx] = True
                    break
    
    return surface_o_mask, top_surface_mask, bottom_surface_mask

def validate_atom_counts(total_species, surface_species, bulk_species, total_o, total_h,
                        h_bonded_to_ti, h2_pairs, isolated_h):
    """Validate that the total number of O and H atoms across species matches the actual counts"""
    logging.info("\nValidating atom counts:")

    # Calculate the total number of O atoms across all species
    calculated_o = 0
    for species in ["O", "OH", "H2O", "H3O"]:
        calculated_o += total_species[species]

    # Calculate the total number of H atoms across all species
    calculated_h = 0
    calculated_h += total_species["OH"] * 1      # One H in OH
    calculated_h += total_species["H2O"] * 2     # Two H in H2O
    calculated_h += total_species["H3O"] * 3     # Three H in H3O
    calculated_h += len(h_bonded_to_ti)          # H atoms bonded to Ti
    calculated_h += len(h2_pairs) * 2            # H atoms in H2 molecules
    calculated_h += len(isolated_h)              # Isolated H atoms

    logging.info(f"  Actual number of O atoms: {total_o}")
    logging.info(f"  Calculated number of O atoms: {calculated_o}")
    logging.info(f"  O atom discrepancy: {abs(total_o - calculated_o)}")

    logging.info(f"  Actual number of H atoms: {total_h}")
    logging.info(f"  Calculated number of H atoms: {calculated_h}")
    logging.info(f"    - H bonded to O: {total_species['OH'] + total_species['H2O']*2 + total_species['H3O']*3}")
    logging.info(f"    - H bonded to Ti: {len(h_bonded_to_ti)}")
    logging.info(f"    - H in H2 molecules: {len(h2_pairs) * 2}")
    logging.info(f"    - Isolated H: {len(isolated_h)}")
    logging.info(f"  H atom discrepancy: {abs(total_h - calculated_h)}")

    if abs(total_o - calculated_o) > 0:
        logging.warning(f"Mismatch in O atom count! Difference: {abs(total_o - calculated_o)}")
    else:
        logging.info("O atom count validation passed!")

    if abs(total_h - calculated_h) > 0:
        logging.warning(f"Mismatch in H atom count! Difference: {abs(total_h - calculated_h)}")
    else:
        logging.info("H atom count validation passed!")

    return calculated_o, calculated_h

def plot_species_distribution(surface_species, bulk_species, output_file='species_distribution.png'):
    """Plot the species distribution"""
    logging.info("Generating species distribution plot...")

    # Configure fonts that support the characters used in labels
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Prepare data
    surface_data = [surface_species['OH'], surface_species['H2O']]
    solution_data = [bulk_species['OH'], bulk_species['H3O']]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot TiO2 surface adsorbed species
    species_surface = ['OH', 'H2O']
    colors_surface = ['orange', 'lightblue']

    bars1 = ax1.bar(species_surface, surface_data, color=colors_surface, alpha=0.8)
    ax1.set_title('TiO2 Surface Adsorbed Species', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Molecules', fontsize=12)
    ax1.set_xlabel('Species Type', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add labels on bars
    for bar, value in zip(bars1, surface_data):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(surface_data)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')

    # Plot species in solution
    species_solution = ['OH-', 'H3O+']
    colors_solution = ['red', 'green']

    bars2 = ax2.bar(species_solution, solution_data, color=colors_solution, alpha=0.8)
    ax2.set_title('Solution Species', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Molecules', fontsize=12)
    ax2.set_xlabel('Species Type', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add labels on bars
    for bar, value in zip(bars2, solution_data):
        if value > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(solution_data)*0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Species distribution plot saved as: {output_file}")

def extract_last_frame(dump_file, output_file):
    """Extract the last frame from dump.xyz and save it"""
    try:
        logging.info("Attempting to read file...")
        atoms = read(dump_file, index='-1')  # Read only the last frame
        if atoms is None:
            raise ValueError("Unable to read file contents")

        # Save the last frame
        write(output_file, atoms)
        logging.info(f"Successfully extracted last frame to {output_file}")

        return atoms

    except Exception as e:
        logging.error(f"Error extracting last frame: {str(e)}")
        raise

def read_lammps_files(data_file, traj_file, atom_style=None):
    """Read LAMMPS data and trajectory files"""
    try:
        logging.info(f"Reading LAMMPS data file: {data_file}")
        logging.info(f"Reading LAMMPS trajectory file: {traj_file}")

        # If the user specifies atom_style, use it with the LAMMPSDUMP format
        if atom_style:
            logging.info(f"Using user-specified atom_style='{atom_style}' with format='LAMMPSDUMP'")
            u = mda.Universe(data_file, traj_file,
                           atom_style=atom_style,
                           format='LAMMPSDUMP')
        else:
            # Try different approaches to read the LAMMPS files
            try:
                # Method 1: Use default parameters with the LAMMPSDUMP format
                logging.info("Attempting method 1: atom_style='id type x y z' with format='LAMMPSDUMP'")
                u = mda.Universe(data_file, traj_file,
                               atom_style='id type x y z',
                               format='LAMMPSDUMP')
            except Exception as e1:
                logging.warning(f"Method 1 failed: {e1}")
                try:
                    # Method 2: Try atomic style
                    logging.info("Attempting method 2: atom_style='atomic' with format='LAMMPSDUMP'")
                    u = mda.Universe(data_file, traj_file,
                                   atom_style='atomic',
                                   format='LAMMPSDUMP')
                except Exception as e2:
                    logging.warning(f"Method 2 failed: {e2}")
                    # Method 3: Try full style
                    logging.info("Attempting method 3: atom_style='full' with format='LAMMPSDUMP'")
                    u = mda.Universe(data_file, traj_file,
                                   atom_style='full',
                                   format='LAMMPSDUMP')
        
        # Access the last frame
        u.trajectory[-1]

        # Retrieve atom information
        positions = u.atoms.positions
        atom_types = u.atoms.types

        # Get box dimensions
        box_dims = u.dimensions[:3]  # Only take the first three dimensions (x, y, z)

        logging.info(f"Read {len(u.atoms)} atoms")
        logging.info(f"Box dimensions: {box_dims}")
        logging.info(f"Atom types: {np.unique(atom_types)}")

        # Create a structure similar to ASE atoms
        class LAMMPSAtoms:
            def __init__(self, positions, symbols, box):
                self.positions = positions
                self.symbols = symbols
                self.box = box
                
            def get_positions(self):
                return self.positions
                
            def get_chemical_symbols(self):
                return self.symbols
        
        # Map LAMMPS atom types to chemical elements
        # Adjust the mapping as needed for your system
        type_to_element = {
            '1': 'H',
            '2': 'O',
            '3': 'N',
            '4': 'Na',
            '5': 'Cl',
            '6': 'Ti',
        }

        # Convert atom types to chemical symbols
        symbols = [type_to_element.get(str(t), 'X') for t in atom_types]

        # Check for unknown atom types
        unknown_types = set(str(t) for t in atom_types if str(t) not in type_to_element)
        if unknown_types:
            logging.warning(f"Unknown atom types found: {unknown_types}")
            logging.warning("Please verify the type_to_element mapping")

        # Count atom types
        symbol_counts = Counter(symbols)
        logging.info(f"Atom counts: {dict(symbol_counts)}")

        # Create atom object
        atoms = LAMMPSAtoms(positions, symbols, box_dims)

        return atoms, box_dims

    except Exception as e:
        logging.error(f"Error while reading LAMMPS files: {str(e)}")
        raise

def extract_last_frame_lammps(data_file, traj_file, output_file, atom_style=None):
    """Extract the last frame from LAMMPS files and save it as xyz"""
    try:
        logging.info("Reading LAMMPS files...")
        atoms, box = read_lammps_files(data_file, traj_file, atom_style)

        # Save as xyz format
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        with open(output_file, 'w') as f:
            f.write(f"{len(positions)}\n")
            f.write(f'Lattice="{box[0]} 0.0 0.0 0.0 {box[1]} 0.0 0.0 0.0 {box[2]}" Properties=species:S:1:pos:R:3\n')
            for symbol, pos in zip(symbols, positions):
                f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        logging.info(f"Successfully extracted the last frame to {output_file}")

        return atoms

    except Exception as e:
        logging.error(f"Error extracting the last LAMMPS frame: {str(e)}")
        raise

def parse_lattice_from_lammps(box_dims):
    """Obtain lattice information from LAMMPS box dimensions"""
    try:
        logging.info(f"LAMMPS box dimensions: {box_dims}")
        return box_dims
    except Exception as e:
        logging.error(f"Failed to parse LAMMPS lattice information: {e}")
        return np.array([100.0, 100.0, 100.0])

def find_close_oxygen_pairs_kdtree(o_positions, box, o_o_cutoff=1.5):
    """Detect O-O neighboring pairs using KDTree"""
    logging.info("Detecting O-O neighbors with KDTree...")

    # Create extended periodic images for O atoms
    extended_o_positions, o_original_indices = apply_pbc(o_positions, box)

    # Build KDTree
    tree = cKDTree(extended_o_positions)

    # Store identified O-O neighbor pairs
    close_o_pairs = set()

    # Examine each O atom
    for o_idx, o_pos in enumerate(o_positions):
        # Find all O atoms within the cutoff distance
        indices = tree.query_ball_point(o_pos, o_o_cutoff)

        for idx in indices:
            partner_o_idx = o_original_indices[idx]
            # Exclude itself and ensure each pair is only counted once
            if partner_o_idx > o_idx:  # Record each pair only once
                extended_o_pos = extended_o_positions[idx]
                dist = np.linalg.norm(o_pos - extended_o_pos)
                if dist <= o_o_cutoff:
                    close_o_pairs.add((o_idx, partner_o_idx, dist))

    return close_o_pairs

def find_isolated_nitrogen_kdtree(n_positions, box, n_n_cutoff=1.5):
    """Detect isolated N atoms using KDTree"""
    logging.info("Detecting isolated N atoms with KDTree...")

    if len(n_positions) == 0:
        return set()

    # Create extended periodic images for N atoms
    extended_n_positions, n_original_indices = apply_pbc(n_positions, box)

    # Build KDTree
    tree = cKDTree(extended_n_positions)

    # Store indices of isolated N atoms
    isolated_n = set()

    # Examine each N atom
    for n_idx, n_pos in enumerate(n_positions):
        # Find all N atoms within the cutoff distance
        indices = tree.query_ball_point(n_pos, n_n_cutoff)

        # Remove its own index
        indices = [idx for idx in indices if n_original_indices[idx] != n_idx]

        # If there are no neighboring N atoms, consider it isolated
        if not indices:
            isolated_n.add(n_idx)

    return isolated_n

def analyze_surface_species(atoms, box_dims=None, output_file=None, ti_o_cutoff=2.3, oh_cutoff=1.35,
                          h_ti_cutoff=2.0, h_h_cutoff=1.0, o_o_cutoff=1.5, n_n_cutoff=1.5):
    """Analyze surface adsorbed species"""
    try:
        # Ensure the atoms object is valid
        if atoms is None:
            raise ValueError("Invalid atoms object")
        
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        # Count total atoms
        atom_counts = Counter(symbols)
        total_o = atom_counts.get('O', 0)
        total_h = atom_counts.get('H', 0)

        # Obtain box dimensions
        if box_dims is not None:
            box = box_dims
        elif output_file is not None:
            box = parse_lattice_from_xyz(output_file)
        else:
            box = np.array([100.0, 100.0, 100.0])
            logging.warning("Box dimensions not provided, using default values")

        # Gather indices for each element
        o_indices = [i for i, s in enumerate(symbols) if s == "O"]
        h_indices = [i for i, s in enumerate(symbols) if s == "H"]
        ti_indices = [i for i, s in enumerate(symbols) if s == "Ti"]
        n_indices = [i for i, s in enumerate(symbols) if s == "N"]  # Include N atom indices

        logging.info(f"Number of atoms found: O={len(o_indices)}, H={len(h_indices)}, Ti={len(ti_indices)}, N={len(n_indices)}")
        
        if len(o_indices) == 0 or len(ti_indices) == 0:
            raise ValueError("No O or Ti atoms found in the structure")
        
        # Find the minimum Ti atom index to differentiate TiO2 and solution O atoms
        min_ti_index = min(ti_indices)
        max_ti_index = max(ti_indices)

        # Separate O atoms belonging to TiO2 and to the solution
        tio2_o_indices = [i for i in o_indices if i < min_ti_index]
        solution_o_indices = [i for i in o_indices if i > max_ti_index]

        logging.info(f"Number of O atoms in TiO2: {len(tio2_o_indices)}")
        logging.info(f"Number of O atoms in solution: {len(solution_o_indices)}")

        # Check for O atoms located between Ti atoms (should not occur)
        middle_o_indices = [i for i in o_indices if min_ti_index <= i <= max_ti_index]
        if middle_o_indices:
            logging.warning(f"Found {len(middle_o_indices)} O atom indices between Ti atoms, which may indicate a sorting issue")

        o_positions = positions[o_indices]
        h_positions = positions[h_indices] if h_indices else np.array([])
        ti_positions = positions[ti_indices]
        n_positions = positions[n_indices] if n_indices else np.array([])  # Include N atom positions

        # Locate the top-layer Ti atoms
        ti_z_coords = ti_positions[:, 2]
        max_ti_z = np.max(ti_z_coords)
        # Treat Ti atoms with z differences smaller than 2 as belonging to the same layer
        top_ti_mask = np.abs(ti_z_coords - max_ti_z) < 2
        top_ti_positions = ti_positions[top_ti_mask]
        
        if len(top_ti_positions) == 0:
            raise ValueError("No top layer Ti atoms found")
        
        logging.info(f"Identified {len(top_ti_positions)} top-layer Ti atoms, z-coordinate: {max_ti_z:.3f}")

        # Use KDTree to distinguish top and bottom surfaces by Ti-O distances
        surface_o_mask, top_surface_mask, bottom_surface_mask = find_surface_oxygens_kdtree(
            o_positions, ti_positions, box, ti_o_cutoff)

        # Initialize surface species counters (tracking origin and surface side)
        surface_species_tio2_top = defaultdict(int)     # TiO2 top surface species count
        surface_species_solution_top = defaultdict(int)  # Solution-derived top surface species count
        surface_species_tio2_bottom = defaultdict(int)   # TiO2 bottom surface species count
        surface_species_solution_bottom = defaultdict(int)  # Solution-derived bottom surface species count
        bulk_species = defaultdict(int)  # Bulk species count

        # Use KDTree to compute H-O nearest neighbors efficiently
        o_h_counts = defaultdict(int)
        h_bonded_to_o = set()
        if len(h_indices) > 0 and len(o_indices) > 0:
            o_h_counts, h_bonded_to_o = find_nearest_oxygens_kdtree(h_positions, o_positions, box, oh_cutoff)

        # Detect H-Ti bonds
        h_bonded_to_ti = set()
        if len(h_indices) > 0 and len(ti_indices) > 0:
            h_bonded_to_ti = find_h_ti_bonds_kdtree(h_positions, ti_positions, box, h_ti_cutoff)

        # Detect H2 molecules
        h_bonded_to_h = set()
        h2_pairs = []
        if len(h_indices) > 1:
            h_bonded_to_h, h2_pairs = find_h2_molecules_kdtree(h_positions, box, h_h_cutoff)

        # Identify isolated H atoms (not bonded to O, Ti, or H)
        all_h_indices = set(range(len(h_indices)))
        isolated_h = all_h_indices - h_bonded_to_o - h_bonded_to_ti - h_bonded_to_h

        # Detect O-O neighbors
        close_o_pairs = find_close_oxygen_pairs_kdtree(o_positions, box, o_o_cutoff)
        if close_o_pairs:
            logging.info("\nO-O neighbor analysis:")
            logging.info(f"Identified {len(close_o_pairs)} O-O neighbor pairs (distance < {o_o_cutoff}Å):")
            for o1_idx, o2_idx, dist in close_o_pairs:
                logging.info(f"  O atom pair: {o_indices[o1_idx]}-{o_indices[o2_idx]}, distance: {dist:.3f}Å")
        else:
            logging.info(f"\nNo O-O neighbors found (distance < {o_o_cutoff}Å)")

        # Detect isolated N atoms
        if len(n_positions) > 0:
            isolated_n = find_isolated_nitrogen_kdtree(n_positions, box, n_n_cutoff)
            if isolated_n:
                logging.info("\nN atom dissociation analysis:")
                logging.info(f"Identified {len(isolated_n)} isolated N atoms (no other N atoms within {n_n_cutoff}Å):")
                for n_idx in isolated_n:
                    logging.info(f"  Isolated N atom index: {n_indices[n_idx]}")
            else:
                logging.info(f"\nNo isolated N atoms detected (all N atoms have neighbors within {n_n_cutoff}Å)")

        # Report additional H atom bonding information
        logging.info(f"\nH atom bonding analysis:")
        logging.info(f"  H atoms bonded to O: {len(h_bonded_to_o)}")
        logging.info(f"  H atoms bonded to Ti: {len(h_bonded_to_ti)}")
        logging.info(f"  Number of H2 molecules: {len(h2_pairs)} (containing {len(h2_pairs)*2} H atoms)")
        logging.info(f"  Isolated H atoms: {len(isolated_h)}")
        
        # Analyze each O atom
        tio2_o_count = 0  # Count of O atoms in TiO2
        tio2_bulk_o_count = 0  # Count of O atoms in bulk TiO2
        solution_o_count = 0  # Count of O atoms in solution
        unaccounted_solution_o = []  # Solution O atoms not assigned to any species

        for i, o_idx in enumerate(o_indices):
            h_count = o_h_counts[i]
            is_surface = surface_o_mask[i]
            is_top = top_surface_mask[i]
            is_bottom = bottom_surface_mask[i]

            # If the O atom belongs to TiO2
            if o_idx < min_ti_index:
                tio2_o_count += 1
                # Determine species type based on number of bonded H atoms
                species_type = "O"
                if h_count == 1:
                    species_type = "OH"
                elif h_count == 2:
                    species_type = "H2O"
                elif h_count == 3:
                    species_type = "H3O"
                
                # Record surface species
                if is_top:
                    surface_species_tio2_top[species_type] += 1
                elif is_bottom:
                    surface_species_tio2_bottom[species_type] += 1
                else:
                    tio2_bulk_o_count += 1

            # For O atoms originating from the solution
            elif o_idx > max_ti_index:
                solution_o_count += 1
                # Species in the solution
                if h_count == 1:
                    species_type = "OH"
                    if is_top:
                        surface_species_solution_top[species_type] += 1
                    elif is_bottom:
                        surface_species_solution_bottom[species_type] += 1
                    else:
                        bulk_species[species_type] += 1
                elif h_count == 2:
                    species_type = "H2O"
                    if is_top:
                        surface_species_solution_top[species_type] += 1
                    elif is_bottom:
                        surface_species_solution_bottom[species_type] += 1
                    else:
                        bulk_species[species_type] += 1
                elif h_count == 3:
                    species_type = "H3O"
                    if is_top:
                        surface_species_solution_top[species_type] += 1
                    elif is_bottom:
                        surface_species_solution_bottom[species_type] += 1
                    else:
                        bulk_species[species_type] += 1
                elif h_count == 0:
                    unaccounted_solution_o.append((o_idx, h_count))
                    logging.warning(f"Detected an isolated O atom in solution, index {o_idx}")

        # Combine surface species statistics for overall counts
        surface_species = defaultdict(int)
        for species in ["O", "OH", "H2O", "H3O"]:
            surface_species[species] = (
                surface_species_tio2_top[species] + surface_species_solution_top[species] +
                surface_species_tio2_bottom[species] + surface_species_solution_bottom[species]
            )

        # Output analysis results
        logging.info("\nSpecies distribution in the entire system:")
        total_species = defaultdict(int)
        for species in ["O", "OH", "H2O", "H3O"]:
            total = surface_species[species] + bulk_species[species]
            total_species[species] = total
            logging.info(f"  {species}: {total} (surface: {surface_species[species]}, bulk: {bulk_species[species]})")

        logging.info("\nTiO2 top surface adsorption analysis:")
        logging.info("Surface species originating from TiO2:")
        for species in ["O", "OH", "H2O", "H3O"]:
            count = surface_species_tio2_top[species]
            if count > 0:
                logging.info(f"  Adsorbed {species}: {count}")

        logging.info("\nTiO2 top surface species originating from solution:")
        for species in ["OH", "H2O", "H3O"]:
            count = surface_species_solution_top[species]
            if count > 0:
                logging.info(f"  Adsorbed {species}: {count}")

        logging.info("\nTiO2 bottom surface adsorption analysis:")
        logging.info("Surface species originating from TiO2:")
        for species in ["O", "OH", "H2O", "H3O"]:
            count = surface_species_tio2_bottom[species]
            if count > 0:
                logging.info(f"  Adsorbed {species}: {count}")

        logging.info("\nTiO2 bottom surface species originating from solution:")
        for species in ["OH", "H2O", "H3O"]:
            count = surface_species_solution_bottom[species]
            if count > 0:
                logging.info(f"  Adsorbed {species}: {count}")

        logging.info("\nTotal surface species:")
        for species in ["O", "OH", "H2O", "H3O"]:
            count = surface_species[species]
            if count > 0:
                total_top = surface_species_tio2_top[species] + surface_species_solution_top[species]
                total_bottom = surface_species_tio2_bottom[species] + surface_species_solution_bottom[species]
                logging.info(f"  Adsorbed {species}: {count} (top surface: {total_top}, bottom surface: {total_bottom})")

        logging.info("\nSpecies distribution in solution:")
        for species in ["OH", "H2O", "H3O"]:
            count = bulk_species[species]
            if count > 0:
                logging.info(f"  {species} in solution: {count}")

        # Provide detailed oxygen statistics
        logging.info("\nDetailed oxygen statistics:")
        logging.info(f"  Total O atoms in TiO2: {tio2_o_count}")
        top_surface_o = sum(surface_species_tio2_top.values())
        bottom_surface_o = sum(surface_species_tio2_bottom.values())
        logging.info(f"    - Top surface O atoms: {top_surface_o}")
        logging.info(f"    - Bottom surface O atoms: {bottom_surface_o}")
        logging.info(f"    - Bulk O atoms: {tio2_bulk_o_count}")
        logging.info(f"  Total O atoms in solution: {solution_o_count}")
        logging.info(f"    - Bonded O atoms: {solution_o_count - len(unaccounted_solution_o)}")
        if unaccounted_solution_o:
            logging.info(f"    - Unbonded O atoms: {len(unaccounted_solution_o)}")
            logging.info("    - Details of unbonded O atoms:")
            for o_idx, h_count in unaccounted_solution_o:
                logging.info(f"      Index: {o_idx}, number of bonded H atoms: {h_count}")

        # Validate atom counts
        validate_atom_counts(total_species, surface_species, bulk_species, total_o, total_h,
                           h_bonded_to_ti, h2_pairs, isolated_h)

        # Generate species distribution plot
        plot_species_distribution(surface_species, bulk_species)
        
        return total_species, surface_species, bulk_species, surface_o_mask, close_o_pairs, isolated_n if len(n_positions) > 0 else set()
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        logging.error("Error details:", exc_info=True)
        raise

if __name__ == "__main__":
    # Retrieve command line arguments
    args = get_args()

    try:
        # Record start time
        start_time = datetime.now()

        if args.format == 'exyz':
            logging.info(f"Starting analysis of {args.input} (EXYZ format)")
            # Extract the last frame
            atoms = extract_last_frame(args.input, args.output)

            # Analyze surface species
            total_species, surface_species, bulk_species, surface_o_mask, close_o_pairs, isolated_n = analyze_surface_species(
                atoms,
                output_file=args.output,
                ti_o_cutoff=args.ti_o_cutoff,
                oh_cutoff=args.oh_cutoff,
                h_ti_cutoff=args.h_ti_cutoff,
                h_h_cutoff=args.h_h_cutoff,
                o_o_cutoff=args.o_o_cutoff,
                n_n_cutoff=args.n_n_cutoff
            )
            
        elif args.format == 'lammps':
            # Set default LAMMPS file paths
            if args.traj is None:
                args.traj = '../bubble_1k.lammpstrj'
            if args.input == 'dump.xyz':  # Replace default with LAMMPS data file
                args.input = '../model_atomic.data'

            logging.info(f"Starting analysis of LAMMPS files: {args.input}, {args.traj}")
            if args.atom_style:
                logging.info(f"Using specified atom_style: {args.atom_style}")

            # Extract the last frame
            atoms = extract_last_frame_lammps(args.input, args.traj, args.output, args.atom_style)

            # Obtain box dimensions
            _, box_dims = read_lammps_files(args.input, args.traj, args.atom_style)

            # Analyze surface species
            total_species, surface_species, bulk_species, surface_o_mask, close_o_pairs, isolated_n = analyze_surface_species(
                atoms,
                box_dims=box_dims,
                ti_o_cutoff=args.ti_o_cutoff,
                oh_cutoff=args.oh_cutoff,
                h_ti_cutoff=args.h_ti_cutoff,
                h_h_cutoff=args.h_h_cutoff,
                o_o_cutoff=args.o_o_cutoff,
                n_n_cutoff=args.n_n_cutoff
            )
        
        # Record end time and total duration
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\nAnalysis complete! Total duration: {duration}")

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise