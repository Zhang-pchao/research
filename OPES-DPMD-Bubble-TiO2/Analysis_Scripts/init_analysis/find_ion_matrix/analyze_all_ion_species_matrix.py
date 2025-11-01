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
# Use a non-interactive backend
matplotlib.use('Agg')
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import argparse

# Configure logging
def setup_logging(enable_log_file=False):
    """Configure logging handlers"""
    log_handlers = [logging.StreamHandler()]
    
    if enable_log_file:
        log_file = 'analyze_ion_species.log'
        if os.path.exists(log_file):
            os.remove(log_file)  # Remove an existing log file
        log_handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze TiO2 surface ion species')
    parser.add_argument('--format', choices=['exyz', 'lammps'], default='lammps',
                       help='Input file format (exyz or lammps)')
    parser.add_argument('--input', default='../model_atomic.data', help='Input filename (exyz format) or LAMMPS data file')
    parser.add_argument('--traj', default='../bubble_1k.lammpstrj', help='LAMMPS trajectory filename (required for lammps format)')
    parser.add_argument('--atom_style', default=None, help='LAMMPS atom_style (e.g. "id type x y z", "atomic", "full")')
    parser.add_argument('--step_interval', type=int, default=100, help='Frame interval for analysis (analyze every n frames)')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame index for analysis')
    parser.add_argument('--end_frame', type=int, default=-1, help='Ending frame index (-1 for the last frame)')

    # Add cutoff parameters
    parser.add_argument('--ti_o_cutoff', type=float, default=3.5, help='Ti-O distance cutoff (Å) for identifying surface adsorption')
    parser.add_argument('--oh_cutoff', type=float, default=1.4, help='O-H bond length cutoff (Å)')
    parser.add_argument('--h_ti_cutoff', type=float, default=2.1, help='H-Ti bond length cutoff (Å)')
    parser.add_argument('--enable_log_file', action='store_true', help='Enable log file output')
    
    return parser.parse_args()

def parse_lattice_from_xyz(filename):
    """Extract lattice information from the second line of an xyz file"""
    try:
        with open(filename, 'r') as f:
            f.readline()  # Skip the first line (number of atoms)
            header = f.readline().strip()  # Read the second line

        # Extract lattice information using regular expressions
        lattice_match = re.search(r'Lattice="([^"]+)"', header)
        if lattice_match:
            lattice_str = lattice_match.group(1)
            lattice_values = list(map(float, lattice_str.split()))

            # Reshape into a 3x3 matrix
            lattice_matrix = np.array(lattice_values).reshape(3, 3)
            # Use the diagonal elements as the box dimensions
            box = np.array([lattice_matrix[0,0], lattice_matrix[1,1], lattice_matrix[2,2]])
            logging.info(f"Extracted lattice dimensions from file: {box}")
            return box
        else:
            logging.warning("Lattice information not found; using default values")
            return np.array([100.0, 100.0, 100.0])

    except Exception as e:
        logging.error(f"Failed to parse lattice information: {e}")
        return np.array([100.0, 100.0, 100.0])

def apply_pbc(positions, box):
    """Apply periodic boundary conditions and generate all images"""
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
    """Use a KDTree to find nearest O atoms and build O-H bonding relationships"""

    # Create periodic images for O atoms
    extended_o_positions, o_original_indices = apply_pbc(o_positions, box)

    # Build the KDTree
    tree = cKDTree(extended_o_positions)

    # Build O-H bonding mappings
    o_h_bonds = defaultdict(list)  # List of H atom indices bonded to each O
    h_bonded_to_o = set()  # Track H atom indices bonded to any O

    for h_idx, h_pos in enumerate(h_positions):
        # Query all O atoms within the cutoff distance
        indices = tree.query_ball_point(h_pos, oh_cutoff)

        if indices:
            # Compute distances and find the nearest
            distances = []
            for idx in indices:
                extended_o_pos = extended_o_positions[idx]
                dist = np.linalg.norm(h_pos - extended_o_pos)
                distances.append((dist, o_original_indices[idx]))

            # Find the nearest O atom
            min_dist, nearest_o_idx = min(distances)
            if min_dist <= oh_cutoff:
                o_h_bonds[nearest_o_idx].append(h_idx)
                h_bonded_to_o.add(h_idx)

    # Count H atoms bonded to each O atom
    o_h_counts = {o_idx: len(h_list) for o_idx, h_list in o_h_bonds.items()}

    return o_h_counts, h_bonded_to_o, o_h_bonds

def find_surface_oxygens_kdtree(o_positions, ti_positions, box, ti_o_cutoff=3.5):
    """Use a KDTree to identify surface-adsorbed O atoms on the top surface"""

    # Locate the top-layer Ti atoms
    ti_z_coords = ti_positions[:, 2]
    max_ti_z = np.max(ti_z_coords)

    # Treat Ti atoms with z differences smaller than 2 Å as part of the same layer
    top_ti_mask = np.abs(ti_z_coords - max_ti_z) < 2
    top_ti_positions = ti_positions[top_ti_mask]

    logging.debug(f"Found {len(top_ti_positions)} top-layer Ti atoms, z coordinate: {max_ti_z:.3f}")

    # Mark O atoms on the top surface
    top_surface_mask = np.zeros(len(o_positions), dtype=bool)

    # Check whether each O atom is on the top surface
    for o_idx, o_pos in enumerate(o_positions):
        for ti_pos in top_ti_positions:
            dist = np.linalg.norm(o_pos - ti_pos)
            if dist <= ti_o_cutoff:
                top_surface_mask[o_idx] = True
                break
    
    return top_surface_mask

def read_lammps_frame(u, frame_idx):
    """Read a specified frame from a LAMMPS trajectory"""
    try:
        u.trajectory[frame_idx]
        
        # Retrieve atomic information
        positions = u.atoms.positions
        atom_types = u.atoms.types

        # Obtain the full box information [a, b, c, alpha, beta, gamma]
        box_info = u.dimensions
        box_dims = box_info[:3]  # Only use the first three dimensions (x, y, z) for distance calculations

        # Map LAMMPS atom types to chemical elements
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

        # Create an ASE-like atom container
        class LAMMPSAtoms:
            def __init__(self, positions, symbols, box_dims, box_info):
                self.positions = positions
                self.symbols = symbols
                self.box = box_dims
                self.box_info = box_info  # Full box information
                
            def get_positions(self):
                return self.positions
                
            def get_chemical_symbols(self):
                return self.symbols
        
        atoms = LAMMPSAtoms(positions, symbols, box_dims, box_info)
        
        return atoms, box_dims, box_info
        
    except Exception as e:
        logging.error(f"Error while reading frame {frame_idx}: {str(e)}")
        return None, None, None

def write_coordinates_to_xyz(positions, symbols, filename, frame_idx, box_info=None, append_mode=False, atom_indices=None):
    """Write coordinates to an extended xyz file, optionally including atom indices"""
    try:
        mode = 'a' if append_mode else 'w'
        with open(filename, mode) as f:
            f.write(f"{len(positions)}\n")
            
            # Build the second line for extended xyz format
            header_parts = []

            # Add frame information
            header_parts.append(f"Frame={frame_idx}")

            # Add PBC information (periodic in all directions by default)
            header_parts.append('pbc="T T T"')

            # Add lattice information when available
            if box_info is not None and len(box_info) >= 3:
                # Build a 3x3 lattice matrix
                a, b, c = box_info[:3]
                if len(box_info) >= 6:
                    alpha, beta, gamma = box_info[3:6]
                    # For non-orthogonal boxes a conversion would be needed
                    # Simplify by assuming an orthogonal box (typical for many LAMMPS simulations)
                    lattice_str = f"{a} 0.0 0.0 0.0 {b} 0.0 0.0 0.0 {c}"
                else:
                    # Orthogonal box
                    lattice_str = f"{a} 0.0 0.0 0.0 {b} 0.0 0.0 0.0 {c}"
                header_parts.append(f'lattice="{lattice_str}"')

            # Add properties information, including atom indices if provided
            if atom_indices is not None:
                header_parts.append('properties=species:S:1:pos:R:3:atom_index:I:1')
            else:
                header_parts.append('properties=species:S:1:pos:R:3')

            # Write the header line
            header = " ".join(header_parts)
            f.write(f"{header}\n")

            # Write atom coordinates
            if atom_indices is not None:
                for symbol, pos, idx in zip(symbols, positions, atom_indices):
                    f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {idx}\n")
            else:
                for symbol, pos in zip(symbols, positions):
                    f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    except Exception as e:
        logging.error(f"Error while writing coordinate file {filename}: {e}")

def append_coordinates_to_xyz(positions, symbols, filename, frame_idx, box_info=None, atom_indices=None):
    """Append coordinates to an extended xyz file"""
    write_coordinates_to_xyz(positions, symbols, filename, frame_idx, box_info, append_mode=True, atom_indices=atom_indices)

def verify_bonding_distances(coords, symbols, oh_cutoff=1.35, max_oh_distance=1.8):
    """Validate whether intramolecular bonding distances are reasonable"""
    if len(coords) == 0:
        return False, "Empty molecule"

    # Locate the positions of O and H atoms
    o_positions = []
    h_positions = []
    
    for i, (coord, symbol) in enumerate(zip(coords, symbols)):
        if symbol == 'O':
            o_positions.append((i, coord))
        elif symbol == 'H':
            h_positions.append((i, coord))
    
    # Ensure there is exactly one O atom
    if len(o_positions) != 1:
        return False, f"Molecule should contain exactly one O atom, found {len(o_positions)}"

    # Validate O-H bond distances
    o_coord = o_positions[0][1]
    for h_idx, h_coord in h_positions:
        distance = np.linalg.norm(np.array(o_coord) - np.array(h_coord))
        if distance > max_oh_distance:
            return False, f"O-H distance too large: {distance:.3f}Å > {max_oh_distance}Å"
        if distance < 0.5:  # Distances that are too short are also unreasonable
            return False, f"O-H distance too small: {distance:.3f}Å < 0.5Å"

    return True, "Bond distance validation passed"

def create_valid_molecule(o_coord, h_coords, h_indices, expected_h_count, oh_cutoff=1.35):
    """Create a validated molecule ensuring all H atoms are bonded to the O atom"""
    if len(h_coords) != expected_h_count:
        logging.warning(f"Mismatch in H atom count: expected {expected_h_count}, found {len(h_coords)}")
        return None

    # Validate the distance between each H atom and the O atom
    valid_h_coords = []
    valid_h_indices = []

    for h_coord, h_idx in zip(h_coords, h_indices):
        distance = np.linalg.norm(np.array(o_coord) - np.array(h_coord))
        if distance <= oh_cutoff:
            valid_h_coords.append(h_coord)
            valid_h_indices.append(h_idx)
        else:
            logging.warning(f"H atom {h_idx} is too far from the O atom ({distance:.3f}Å); skipping")

    # Check whether the number of valid H atoms matches expectations
    if len(valid_h_coords) != expected_h_count:
        logging.warning(f"Mismatch in valid H atom count: expected {expected_h_count}, valid {len(valid_h_coords)}")
        return None

    # Construct the molecule
    molecule = {
        'coords': [o_coord] + valid_h_coords,
        'symbols': ['O'] + ['H'] * len(valid_h_coords)
    }

    # Validate bond distances
    is_valid, message = verify_bonding_distances(molecule['coords'], molecule['symbols'], oh_cutoff)
    if not is_valid:
        logging.warning(f"Molecule bond validation failed: {message}")
        return None

    return molecule

def validate_molecular_integrity(results, frame_idx):
    """Validate molecular integrity and produce a report including symbol order and bond checks"""
    integrity_report = {
        'frame': frame_idx,
        'tio2_surface_h': {'valid': 0, 'invalid': 0},
        'solution_surface_oh': {'valid': 0, 'invalid': 0},
        'solution_surface_h2o': {'valid': 0, 'invalid': 0},
        'solution_bulk_oh': {'valid': 0, 'invalid': 0},
        'solution_bulk_h3o': {'valid': 0, 'invalid': 0}
    }
    
    # Validate TiO2 surface H (expected 1 atom: H)
    for molecule in results['tio2_surface_h']['molecules']:
        if (len(molecule['coords']) == 1 and len(molecule['symbols']) == 1 and 
            molecule['symbols'] == ['H']):
            integrity_report['tio2_surface_h']['valid'] += 1
        else:
            integrity_report['tio2_surface_h']['invalid'] += 1
            logging.warning(f"Frame {frame_idx}: TiO2 surface H molecule incomplete: {len(molecule['coords'])} coordinates, "
                          f"symbols: {molecule['symbols']}")

    # Validate surface OH (expected 2 atoms: O, H)
    for molecule in results['solution_surface_oh']['molecules']:
        is_valid = (len(molecule['coords']) == 2 and len(molecule['symbols']) == 2 and 
                   molecule['symbols'] == ['O', 'H'])
        if is_valid:
            # Additional bond validation
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_surface_oh']['valid'] += 1
            else:
                integrity_report['solution_surface_oh']['invalid'] += 1
                logging.warning(f"Frame {frame_idx}: Surface OH molecule failed bond validation: {message}")
        else:
            integrity_report['solution_surface_oh']['invalid'] += 1
            logging.warning(f"Frame {frame_idx}: Surface OH molecule format error: {len(molecule['coords'])} coordinates, "
                          f"symbols: {molecule['symbols']}")

    # Validate surface H2O (expected 3 atoms: O, H, H)
    for molecule in results['solution_surface_h2o']['molecules']:
        is_valid = (len(molecule['coords']) == 3 and len(molecule['symbols']) == 3 and 
                   molecule['symbols'] == ['O', 'H', 'H'])
        if is_valid:
            # Additional bond validation
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_surface_h2o']['valid'] += 1
            else:
                integrity_report['solution_surface_h2o']['invalid'] += 1
                logging.warning(f"Frame {frame_idx}: Surface H2O molecule failed bond validation: {message}")
        else:
            integrity_report['solution_surface_h2o']['invalid'] += 1
            logging.warning(f"Frame {frame_idx}: Surface H2O molecule format error: {len(molecule['coords'])} coordinates, "
                          f"symbols: {molecule['symbols']}")

    # Validate bulk OH (expected 2 atoms: O, H)
    for molecule in results['solution_bulk_oh']['molecules']:
        is_valid = (len(molecule['coords']) == 2 and len(molecule['symbols']) == 2 and 
                   molecule['symbols'] == ['O', 'H'])
        if is_valid:
            # Additional bond validation
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_bulk_oh']['valid'] += 1
            else:
                integrity_report['solution_bulk_oh']['invalid'] += 1
                logging.warning(f"Frame {frame_idx}: Bulk OH molecule failed bond validation: {message}")
        else:
            integrity_report['solution_bulk_oh']['invalid'] += 1
            logging.warning(f"Frame {frame_idx}: Bulk OH molecule format error: {len(molecule['coords'])} coordinates, "
                          f"symbols: {molecule['symbols']}")

    # Validate H3O (expected 4 atoms: O, H, H, H)
    for molecule in results['solution_bulk_h3o']['molecules']:
        is_valid = (len(molecule['coords']) == 4 and len(molecule['symbols']) == 4 and 
                   molecule['symbols'] == ['O', 'H', 'H', 'H'])
        if is_valid:
            # Additional bond validation
            is_bonded, message = verify_bonding_distances(molecule['coords'], molecule['symbols'])
            if is_bonded:
                integrity_report['solution_bulk_h3o']['valid'] += 1
            else:
                integrity_report['solution_bulk_h3o']['invalid'] += 1
                logging.warning(f"Frame {frame_idx}: H3O molecule failed bond validation: {message}")
        else:
            integrity_report['solution_bulk_h3o']['invalid'] += 1
            logging.warning(f"Frame {frame_idx}: H3O molecule format error: {len(molecule['coords'])} coordinates, "
                          f"symbols: {molecule['symbols']}")
    
    return integrity_report

def analyze_frame_ion_species(atoms, box_dims, frame_idx, ti_o_cutoff=3.5, oh_cutoff=1.35):
    """Analyze ion species for a single frame"""
    try:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        # Gather atom indices by element
        o_indices = [i for i, s in enumerate(symbols) if s == "O"]
        h_indices = [i for i, s in enumerate(symbols) if s == "H"]
        ti_indices = [i for i, s in enumerate(symbols) if s == "Ti"]
        na_indices = [i for i, s in enumerate(symbols) if s == "Na"]
        cl_indices = [i for i, s in enumerate(symbols) if s == "Cl"]
        
        if len(o_indices) == 0 or len(ti_indices) == 0 or len(h_indices) == 0:
            logging.warning(f"Frame {frame_idx}: missing required atom types")
            return None

        # Identify Ti index range to distinguish TiO2 O atoms from solution O atoms
        min_ti_index = min(ti_indices)
        max_ti_index = max(ti_indices)
        
        # Separate TiO2 O atoms from solution O atoms
        tio2_o_indices = [i for i in o_indices if i < min_ti_index]
        solution_o_indices = [i for i in o_indices if i > max_ti_index]
        
        o_positions = positions[o_indices]
        h_positions = positions[h_indices]
        ti_positions = positions[ti_indices]
        
        # Identify top-surface O atoms
        top_surface_mask = find_surface_oxygens_kdtree(o_positions, ti_positions, box_dims, ti_o_cutoff)

        # Determine H-O bonding
        o_h_counts, h_bonded_to_o, o_h_bonds = find_nearest_oxygens_kdtree(h_positions, o_positions, box_dims, oh_cutoff)

        # Initialize results structured at the molecular level, including atom indices
        results = {
            'frame': frame_idx,
            'tio2_surface_h': {'count': 0, 'molecules': []},      # A: TiO2 surface-adsorbed H, each entry is a single H atom
            'solution_surface_oh': {'count': 0, 'molecules': []}, # B: Solution-derived surface OH, each entry is [O, H]
            'solution_surface_h2o': {'count': 0, 'molecules': []},  # C: Solution-derived surface H2O, each entry is [O, H, H]
            'solution_bulk_oh': {'count': 0, 'molecules': []},    # D: Bulk OH in solution, each entry is [O, H]
            'solution_bulk_h3o': {'count': 0, 'molecules': []},   # E: Bulk H3O in solution, each entry is [O, H, H, H]
            'na_ions': {'count': 0, 'coords': [], 'symbols': [], 'indices': []},             # Na ions
            'cl_ions': {'count': 0, 'coords': [], 'symbols': [], 'indices': []},             # Cl ions
            'tio2_surface_o': {'count': 0, 'z_coords': [], 'avg_z': 0.0}      # TiO2 surface-adsorbed O atoms
        }

        # Analyze each O atom
        for i, o_idx in enumerate(o_indices):
            h_count = o_h_counts.get(i, 0)  # Default to zero when not found
            is_top_surface = top_surface_mask[i]

            # Retrieve O atom coordinates
            o_coord = positions[o_idx]

            # For TiO2 O atoms
            if o_idx < min_ti_index:
                # TiO2 surface-adsorbed O atoms (h_count=0)
                if is_top_surface and h_count == 0:
                    results['tio2_surface_o']['count'] += 1
                    results['tio2_surface_o']['z_coords'].append(o_coord[2])

                # A: TiO2 surface-adsorbed H (formerly OH, now treated as H)
                elif is_top_surface and h_count == 1:
                    results['tio2_surface_h']['count'] += 1
                    # Locate the H atom bonded to this O atom
                    if i in o_h_bonds:
                        for h_idx in o_h_bonds[i]:
                            h_pos = h_positions[h_idx]
                            # Store information for the single H atom, including its index
                            h_molecule = {
                                'coords': [h_pos],
                                'symbols': ['H'],
                                'indices': [h_indices[h_idx]]  # Record the original H atom index
                            }
                            results['tio2_surface_h']['molecules'].append(h_molecule)
                            break

            # For O atoms in solution
            elif o_idx > max_ti_index:
                if is_top_surface:
                    if h_count == 1:
                        # B: Solution-derived surface OH
                        if i in o_h_bonds and len(o_h_bonds[i]) == 1:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # Create an OH molecule via the validation helper
                            oh_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 1, oh_cutoff)
                            if oh_molecule is not None:
                                # Attach atom indices
                                oh_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_surface_oh']['count'] += 1
                                results['solution_surface_oh']['molecules'].append(oh_molecule)
                            else:
                                logging.warning(f"Frame {frame_idx}: Failed to create surface OH molecule (O index: {o_idx})")
                    elif h_count == 2:
                        # C: Solution-derived surface H2O
                        if i in o_h_bonds and len(o_h_bonds[i]) == 2:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # Create an H2O molecule
                            h2o_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 2, oh_cutoff)
                            if h2o_molecule is not None:
                                # Attach atom indices
                                h2o_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_surface_h2o']['count'] += 1
                                results['solution_surface_h2o']['molecules'].append(h2o_molecule)
                            else:
                                logging.warning(f"Frame {frame_idx}: Failed to create surface H2O molecule (O index: {o_idx})")
                else:
                    # Bulk species
                    if h_count == 1:
                        # D: Bulk OH in solution
                        if i in o_h_bonds and len(o_h_bonds[i]) == 1:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # Create an OH molecule via the validation helper
                            oh_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 1, oh_cutoff)
                            if oh_molecule is not None:
                                # Attach atom indices
                                oh_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_bulk_oh']['count'] += 1
                                results['solution_bulk_oh']['molecules'].append(oh_molecule)
                            else:
                                logging.warning(f"Frame {frame_idx}: Failed to create bulk OH molecule (O index: {o_idx})")
                    elif h_count == 3:
                        # E: Bulk H3O in solution
                        if i in o_h_bonds and len(o_h_bonds[i]) == 3:
                            h_coords = [h_positions[h_idx] for h_idx in o_h_bonds[i]]
                            bonded_h_indices = o_h_bonds[i]
                            # Create an H3O molecule via the validation helper
                            h3o_molecule = create_valid_molecule(o_coord, h_coords, bonded_h_indices, 3, oh_cutoff)
                            if h3o_molecule is not None:
                                # Attach atom indices
                                h3o_molecule['indices'] = [o_idx] + [h_indices[h_idx] for h_idx in bonded_h_indices]
                                results['solution_bulk_h3o']['count'] += 1
                                results['solution_bulk_h3o']['molecules'].append(h3o_molecule)
                            else:
                                logging.warning(f"Frame {frame_idx}: Failed to create H3O molecule (O index: {o_idx})")

        # Record Na ion coordinates
        if na_indices:
            for na_idx in na_indices:
                na_coord = positions[na_idx]
                results['na_ions']['count'] += 1
                results['na_ions']['coords'].append(na_coord)
                results['na_ions']['symbols'].append('Na')
                results['na_ions']['indices'].append(na_idx)

        # Record Cl ion coordinates
        if cl_indices:
            for cl_idx in cl_indices:
                cl_coord = positions[cl_idx]
                results['cl_ions']['count'] += 1
                results['cl_ions']['coords'].append(cl_coord)
                results['cl_ions']['symbols'].append('Cl')
                results['cl_ions']['indices'].append(cl_idx)

        # Compute the average z coordinate of TiO2 surface-adsorbed O atoms
        if results['tio2_surface_o']['z_coords']:
            results['tio2_surface_o']['avg_z'] = np.mean(results['tio2_surface_o']['z_coords'])

        return results

    except Exception as e:
        logging.error(f"Error analyzing frame {frame_idx}: {str(e)}")
        return None

def save_frame_coordinates(results, frame_idx, output_dir, box_info=None, is_first_frame=False):
    """Save frame coordinates to consolidated xyz files while preserving molecular integrity and atom indices"""

    def extract_molecular_coords_and_symbols(molecules, expected_atoms_per_molecule):
        """Extract coordinates, symbols, and indices while validating integrity and bond distances"""
        all_coords = []
        all_symbols = []
        all_indices = []
        valid_molecules = 0
        invalid_molecules = 0
        
        for molecule in molecules:
            mol_coords = molecule['coords']
            mol_symbols = molecule['symbols']
            mol_indices = molecule.get('indices', None)
            
            # Validate the atom count for each molecule
            if len(mol_coords) != expected_atoms_per_molecule or len(mol_symbols) != expected_atoms_per_molecule:
                logging.warning(f"Molecule atom count mismatch: expected {expected_atoms_per_molecule} atoms, "
                                f"found {len(mol_coords)} coordinates and {len(mol_symbols)} symbols")
                invalid_molecules += 1
                continue

            # Validate the index count
            if mol_indices is not None and len(mol_indices) != expected_atoms_per_molecule:
                logging.warning(f"Molecule index count mismatch: expected {expected_atoms_per_molecule}, found {len(mol_indices)}")
                invalid_molecules += 1
                continue

            # Validate the symbol sequence
            if expected_atoms_per_molecule == 1:  # H atom
                if mol_symbols != ['H']:
                    logging.warning(f"H molecule symbol sequence error: {mol_symbols}")
                    invalid_molecules += 1
                    continue
            elif expected_atoms_per_molecule == 2:  # OH molecule
                if mol_symbols != ['O', 'H']:
                    logging.warning(f"OH molecule symbol sequence error: {mol_symbols}")
                    invalid_molecules += 1
                    continue
            elif expected_atoms_per_molecule == 3:  # H2O molecule
                if mol_symbols != ['O', 'H', 'H']:
                    logging.warning(f"H2O molecule symbol sequence error: {mol_symbols}")
                    invalid_molecules += 1
                    continue
            elif expected_atoms_per_molecule == 4:  # H3O molecule
                if mol_symbols != ['O', 'H', 'H', 'H']:
                    logging.warning(f"H3O molecule symbol sequence error: {mol_symbols}")
                    invalid_molecules += 1
                    continue

            # For multi-atom molecules, validate bond distances
            if expected_atoms_per_molecule > 1:
                is_valid, message = verify_bonding_distances(mol_coords, mol_symbols)
                if not is_valid:
                    logging.warning(f"Molecule bond validation failed: {message}")
                    invalid_molecules += 1
                    continue

            # Molecules that pass all checks
            all_coords.extend(mol_coords)
            all_symbols.extend(mol_symbols)
            if mol_indices is not None:
                all_indices.extend(mol_indices)
            valid_molecules += 1

        # Ensure the total atom count is a multiple of the expected number
        if len(all_coords) % expected_atoms_per_molecule != 0:
            logging.warning(f"Total atom count {len(all_coords)} is not a multiple of {expected_atoms_per_molecule}")

        if invalid_molecules > 0:
            logging.info(f"Molecule validation results: {valid_molecules} valid, {invalid_molecules} invalid")
        
        return all_coords, all_symbols, all_indices
    
    # A: TiO2 surface-adsorbed H
    if results['tio2_surface_h']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['tio2_surface_h']['molecules'], 1)
        if coords:
            filename = os.path.join(output_dir, "tio2_surface_h.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
    
    # B: Solution-derived surface OH
    if results['solution_surface_oh']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_surface_oh']['molecules'], 2)
        if coords and len(coords) % 2 == 0:  # Ensure the atom count is a multiple of 2
            filename = os.path.join(output_dir, "solution_surface_oh.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"Frame {frame_idx}: surface OH total atom count ({len(coords)}) is not a multiple of 2; skipping write")

    # C: Solution-derived surface H2O
    if results['solution_surface_h2o']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_surface_h2o']['molecules'], 3)
        if coords and len(coords) % 3 == 0:  # Ensure the atom count is a multiple of 3
            filename = os.path.join(output_dir, "solution_surface_h2o.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"Frame {frame_idx}: surface H2O total atom count ({len(coords)}) is not a multiple of 3; skipping write")

    # D: Bulk OH in solution
    if results['solution_bulk_oh']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_bulk_oh']['molecules'], 2)
        if coords and len(coords) % 2 == 0:  # Ensure the atom count is a multiple of 2
            filename = os.path.join(output_dir, "solution_bulk_oh.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"Frame {frame_idx}: bulk OH total atom count ({len(coords)}) is not a multiple of 2; skipping write")

    # E: Bulk H3O in solution
    if results['solution_bulk_h3o']['count'] > 0:
        coords, symbols, indices = extract_molecular_coords_and_symbols(results['solution_bulk_h3o']['molecules'], 4)
        if coords and len(coords) % 4 == 0:  # Ensure the atom count is a multiple of 4
            filename = os.path.join(output_dir, "solution_bulk_h3o.xyz")
            if is_first_frame:
                write_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, append_mode=False, atom_indices=indices)
            else:
                append_coordinates_to_xyz(coords, symbols, filename, frame_idx, box_info, atom_indices=indices)
        elif coords:
            logging.error(f"Frame {frame_idx}: H3O total atom count ({len(coords)}) is not a multiple of 4; skipping write")

    # Na ions
    if results['na_ions']['count'] > 0:
        filename = os.path.join(output_dir, "na_ions.xyz")
        if is_first_frame:
            write_coordinates_to_xyz(
                results['na_ions']['coords'],
                results['na_ions']['symbols'],
                filename, frame_idx, box_info, append_mode=False,
                atom_indices=results['na_ions']['indices']
            )
        else:
            append_coordinates_to_xyz(
                results['na_ions']['coords'],
                results['na_ions']['symbols'],
                filename, frame_idx, box_info,
                atom_indices=results['na_ions']['indices']
            )
    
    # Cl ions
    if results['cl_ions']['count'] > 0:
        filename = os.path.join(output_dir, "cl_ions.xyz")
        if is_first_frame:
            write_coordinates_to_xyz(
                results['cl_ions']['coords'],
                results['cl_ions']['symbols'],
                filename, frame_idx, box_info, append_mode=False,
                atom_indices=results['cl_ions']['indices']
            )
        else:
            append_coordinates_to_xyz(
                results['cl_ions']['coords'],
                results['cl_ions']['symbols'],
                filename, frame_idx, box_info,
                atom_indices=results['cl_ions']['indices']
            )

def setup_nature_style():
    """Configure matplotlib parameters inspired by Nature journal styling"""
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
        'text.usetex': False,  # Use matplotlib's built-in math rendering rather than LaTeX
        'mathtext.default': 'regular'  # Ensure math text uses the regular font
    })

def plot_species_evolution(all_results, output_file='ion_species_evolution.png'):
    """Plot the evolution of species counts over time in a Nature-inspired style"""

    # Apply Nature-style settings
    setup_nature_style()
    
    frames = [r['frame'] for r in all_results]
    
    tio2_surface_h = [r['tio2_surface_h']['count'] for r in all_results]
    solution_surface_oh = [r['solution_surface_oh']['count'] for r in all_results]
    solution_bulk_oh = [r['solution_bulk_oh']['count'] for r in all_results]
    solution_bulk_h3o = [r['solution_bulk_h3o']['count'] for r in all_results]
    na_ions = [r['na_ions']['count'] for r in all_results]
    cl_ions = [r['cl_ions']['count'] for r in all_results]
    
    # Nature-inspired color palette
    colors = {
        'tio2_h': '#1f77b4',      # Blue
        'surface_oh': '#ff7f0e',   # Orange
        'bulk_oh': '#2ca02c',      # Green
        'bulk_h3o': '#d62728',     # Red
        'na': '#9467bd',           # Purple
        'cl': '#8c564b'            # Brown
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # First subplot: primary ion species
    ax1 = axes[0]
    ax1.plot(frames, tio2_surface_h, 'o-', label=r'TiO$_2$ Surface Adsorbed H$^+$', 
             color=colors['tio2_h'], linewidth=2, markersize=5)
    ax1.plot(frames, solution_surface_oh, 's-', label=r'Surface Adsorbed OH$^-$', 
             color=colors['surface_oh'], linewidth=2, markersize=5)
    ax1.plot(frames, solution_bulk_oh, '^-', label=r'Bulk OH$^-$', 
             color=colors['bulk_oh'], linewidth=2, markersize=5)
    ax1.plot(frames, solution_bulk_h3o, 'v-', label=r'Bulk H$_3$O$^+$', 
             color=colors['bulk_h3o'], linewidth=2, markersize=5)
    
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Species Count')
    ax1.set_title('Ion Species Evolution Over Time')
    ax1.legend(frameon=False, loc='best')
    
    # Second subplot: Na and Cl ions (if present)
    ax2 = axes[1]
    has_ions = False
    
    if max(na_ions) > 0:
        ax2.plot(frames, na_ions, 'o-', label=r'Na$^+$ ions', 
                color=colors['na'], linewidth=2, markersize=5)
        has_ions = True
    
    if max(cl_ions) > 0:
        ax2.plot(frames, cl_ions, 's-', label=r'Cl$^-$ ions', 
                color=colors['cl'], linewidth=2, markersize=5)
        has_ions = True
    
    if has_ions:
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Ion Count')
        ax2.set_title('Salt Ions Distribution')
        ax2.legend(frameon=False, loc='best')
    else:
        ax2.text(0.5, 0.5, r'No Na$^+$ or Cl$^-$ ions detected', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=14, style='italic')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Ion Count')
        ax2.set_title('Salt Ions Distribution')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Species evolution plot saved as: {output_file}")

def build_species_transition_matrix(all_results):
    """Build a species transition matrix tracking conversions among five species.

    The mapping is primarily based on O atom indices (H can move via the Grotthuss mechanism).
    Species definitions:
    0: solution_bulk_h3o (bulk H3O+)
    1: solution_bulk_oh (bulk OH-)
    2: solution_surface_oh (surface-adsorbed OH-)
    3: solution_surface_h2o (surface-adsorbed H2O)
    4: tio2_surface_h (TiO2 surface-adsorbed H+, tracked via H indices)
    """
    
    species_names = ['solution_bulk_h3o', 'solution_bulk_oh', 'solution_surface_oh', 
                     'solution_surface_h2o', 'tio2_surface_h']
    n_species = len(species_names)
    
    # Initialize the transition count matrix
    transition_counts = np.zeros((n_species, n_species), dtype=int)
    
    # Store detailed transition records
    transition_details = []
    
    for frame_idx in range(len(all_results) - 1):
        current_frame = all_results[frame_idx]
        next_frame = all_results[frame_idx + 1]
        
        # Map O atom indices to species for the current frame
        current_o_to_species = {}
        # Map H atom indices to species for the current frame (for tio2_surface_h)
        current_h_to_species = {}
        
        # Process each species in the current frame
        for species_idx, species_name in enumerate(species_names):
            molecules = current_frame[species_name].get('molecules', [])
            for mol in molecules:
                indices = mol.get('indices', [])
                if species_name == 'tio2_surface_h':
                    # tio2_surface_h contains only H atoms
                    if indices:
                        current_h_to_species[indices[0]] = species_idx
                else:
                    # Other species include O atoms, with O as the first index
                    if indices:
                        o_idx = indices[0]
                        current_o_to_species[o_idx] = species_idx
        
        # Build the O-index-to-species map for the next frame
        next_o_to_species = {}
        next_h_to_species = {}
        
        for species_idx, species_name in enumerate(species_names):
            molecules = next_frame[species_name].get('molecules', [])
            for mol in molecules:
                indices = mol.get('indices', [])
                if species_name == 'tio2_surface_h':
                    if indices:
                        next_h_to_species[indices[0]] = species_idx
                else:
                    if indices:
                        o_idx = indices[0]
                        next_o_to_species[o_idx] = species_idx
        
        # Count O-atom transitions based on O indices
        for o_idx, from_species in current_o_to_species.items():
            if o_idx in next_o_to_species:
                to_species = next_o_to_species[o_idx]
                transition_counts[from_species, to_species] += 1
                
                # Record detailed transition information
                if from_species != to_species:
                    transition_details.append({
                        'frame': current_frame['frame'],
                        'o_index': o_idx,
                        'from': species_names[from_species],
                        'to': species_names[to_species]
                    })
        
        # Count H-atom transitions (only for tio2_surface_h)
        for h_idx, from_species in current_h_to_species.items():
            if h_idx in next_h_to_species:
                to_species = next_h_to_species[h_idx]
                transition_counts[from_species, to_species] += 1
                
                if from_species != to_species:
                    transition_details.append({
                        'frame': current_frame['frame'],
                        'h_index': h_idx,
                        'from': species_names[from_species],
                        'to': species_names[to_species]
                    })
    
    return transition_counts, species_names, transition_details

def plot_transition_matrix(transition_counts, species_names, output_file='species_transition_matrix.png'):
    """Plot heatmaps for species transition counts and probabilities"""

    setup_nature_style()

    # Compute the transition probability matrix
    # Each row sum represents the total transitions from that species
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_prob = transition_counts / row_sums

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Simplified display names
    display_names = [
        r'Bulk H$_3$O$^+$',
        r'Bulk OH$^-$',
        r'Surface OH$^-$',
        r'Surface H$_2$O',
        r'Surface H$^+$'
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
    
    # Annotate each cell with counts
    for i in range(len(species_names)):
        for j in range(len(species_names)):
            text = ax1.text(j, i, int(transition_counts[i, j]),
                          ha="center", va="center", color="black" if transition_prob[i, j] < 0.5 else "white",
                          fontsize=10)
    
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
    
    # Annotate each cell with probability values
    for i in range(len(species_names)):
        for j in range(len(species_names)):
            text = ax2.text(j, i, f'{transition_prob[i, j]:.2f}',
                          ha="center", va="center", color="black" if transition_prob[i, j] < 0.5 else "white",
                          fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Probability', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"Transition matrix plot saved as: {output_file}")

def save_transition_data(transition_counts, species_names, transition_details, output_dir):
    """Save transition matrix data and detailed transition records"""

    # Save the transition count matrix
    counts_file = os.path.join(output_dir, "transition_counts.txt")
    with open(counts_file, 'w') as f:
        f.write("Transition Counts Matrix\n")
        f.write("From \\ To\t" + "\t".join(species_names) + "\n")
        for i, from_species in enumerate(species_names):
            f.write(f"{from_species}\t")
            f.write("\t".join(str(int(transition_counts[i, j])) for j in range(len(species_names))))
            f.write("\n")
    
    logging.info(f"Transition counts saved to: {counts_file}")
    
    # Compute and save the transition probability matrix
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
    
    logging.info(f"Transition probabilities saved to: {prob_file}")
    
    # Save detailed transition entries
    details_file = os.path.join(output_dir, "transition_details.txt")
    with open(details_file, 'w') as f:
        f.write("Frame\tAtom_Index\tFrom_Species\tTo_Species\n")
        for detail in transition_details:
            atom_idx = detail.get('o_index', detail.get('h_index', 'N/A'))
            f.write(f"{detail['frame']}\t{atom_idx}\t{detail['from']}\t{detail['to']}\n")
    
    logging.info(f"Transition details saved to: {details_file}")
    
    # Save a transition summary report
    summary_file = os.path.join(output_dir, "transition_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Species Transition Summary ===\n\n")
        
        # Total transitions excluding self-transitions
        total_transitions = np.sum(transition_counts) - np.trace(transition_counts)
        f.write(f"Total transitions (excluding self-transitions): {int(total_transitions)}\n\n")
        
        # Transition statistics per species
        for i, species in enumerate(species_names):
            f.write(f"\n{species}:\n")
            total_from = np.sum(transition_counts[i, :])
            total_to = np.sum(transition_counts[:, i])
            f.write(f"  Total transitions from this species: {int(total_from)}\n")
            f.write(f"  Total transitions to this species: {int(total_to)}\n")
            
            # Primary transition destinations
            if total_from > 0:
                f.write(f"  Main transitions from {species}:\n")
                for j in range(len(species_names)):
                    if i != j and transition_counts[i, j] > 0:
                        prob = transition_prob[i, j]
                        f.write(f"    -> {species_names[j]}: {int(transition_counts[i, j])} ({prob:.2%})\n")
    
    logging.info(f"Transition summary saved to: {summary_file}")

def main():
    args = get_args()
    
    # Configure logging
    setup_logging(enable_log_file=args.enable_log_file)
    
    start_time = datetime.now()
    logging.info(f"Starting ion species analysis, frame interval: {args.step_interval}")
    
    try:
        # Create the output directory
        output_dir = "ion_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        if args.format == 'lammps':
            # Read LAMMPS files
            logging.info(f"Reading LAMMPS files: {args.input}, {args.traj}")
            
            # Attempt to construct an MDAnalysis Universe
            if args.atom_style:
                u = mda.Universe(args.input, args.traj, 
                               atom_style=args.atom_style,
                               format='LAMMPSDUMP')
            else:
                try:
                    u = mda.Universe(args.input, args.traj, 
                                   atom_style='id type x y z',
                                   format='LAMMPSDUMP')
                except:
                    try:
                        u = mda.Universe(args.input, args.traj, 
                                       atom_style='atomic',
                                       format='LAMMPSDUMP')
                    except:
                        u = mda.Universe(args.input, args.traj, 
                                       atom_style='full',
                                       format='LAMMPSDUMP')
            
            total_frames = len(u.trajectory)
            logging.info(f"Total frames in trajectory: {total_frames}")
            
            # Determine the range of frames to analyze
            start_frame = args.start_frame
            end_frame = total_frames if args.end_frame == -1 else min(args.end_frame, total_frames)
            
            frames_to_analyze = list(range(start_frame, end_frame, args.step_interval))
            logging.info(f"Will analyze {len(frames_to_analyze)} frames")
            
            all_results = []
            
            # Analyze each frame
            for i, frame_idx in enumerate(frames_to_analyze):
                logging.info(f"Analyzing frame {frame_idx}...")
                
                atoms, box_dims, box_info = read_lammps_frame(u, frame_idx)
                if atoms is None:
                    continue
                
                results = analyze_frame_ion_species(
                    atoms, box_dims, frame_idx,
                    ti_o_cutoff=args.ti_o_cutoff,
                    oh_cutoff=args.oh_cutoff
                )
                
                if results is not None:
                    # Validate molecular integrity
                    integrity_report = validate_molecular_integrity(results, frame_idx)
                    
                    all_results.append(results)
                    
                    # Save coordinate files (create for the first frame, append thereafter)
                    is_first_frame = (i == 0)
                    save_frame_coordinates(results, frame_idx, output_dir, box_info, is_first_frame)
                    
                    # Output per-frame statistics including molecular integrity
                    logging.info(f"  Frame {frame_idx} results:")
                    logging.info(f"    TiO2 Surface Adsorbed H: {results['tio2_surface_h']['count']} (valid molecules: {integrity_report['tio2_surface_h']['valid']})")
                    logging.info(f"    Surface Adsorbed OH (from solution): {results['solution_surface_oh']['count']} (valid molecules: {integrity_report['solution_surface_oh']['valid']})")
                    logging.info(f"    Surface Adsorbed H2O (from solution): {results['solution_surface_h2o']['count']} (valid molecules: {integrity_report['solution_surface_h2o']['valid']})")
                    logging.info(f"    Bulk OH in solution: {results['solution_bulk_oh']['count']} (valid molecules: {integrity_report['solution_bulk_oh']['valid']})")
                    logging.info(f"    Bulk H3O in solution: {results['solution_bulk_h3o']['count']} (valid molecules: {integrity_report['solution_bulk_h3o']['valid']})")
                    logging.info(f"    Na+ ions: {results['na_ions']['count']}")
                    logging.info(f"    Cl- ions: {results['cl_ions']['count']}")
                    logging.info(f"    TiO2 Surface Adsorbed O: {results['tio2_surface_o']['count']} (avg z: {results['tio2_surface_o']['avg_z']:.3f})")
                    
                    # Issue a warning if invalid molecules are present
                    total_invalid = (integrity_report['tio2_surface_h']['invalid'] + 
                                   integrity_report['solution_surface_oh']['invalid'] + 
                                   integrity_report['solution_surface_h2o']['invalid'] +
                                   integrity_report['solution_bulk_oh']['invalid'] + 
                                   integrity_report['solution_bulk_h3o']['invalid'])
                    if total_invalid > 0:
                        logging.warning(f"    Frame {frame_idx} contains {total_invalid} incomplete molecules")
            
            # Save summary statistics
            stats_file = os.path.join(output_dir, "species_statistics.txt")
            with open(stats_file, 'w') as f:
                f.write("Frame\tTiO2_Surface_H\tSolution_Surface_OH\tSolution_Surface_H2O\tSolution_Bulk_OH\tSolution_Bulk_H3O\tNa_Ions\tCl_Ions\tTiO2_Surface_O\tTiO2_Surface_O_Avg_Z\n")
                for result in all_results:
                    f.write(f"{result['frame']}\t{result['tio2_surface_h']['count']}\t"
                           f"{result['solution_surface_oh']['count']}\t{result['solution_surface_h2o']['count']}\t"
                           f"{result['solution_bulk_oh']['count']}\t{result['solution_bulk_h3o']['count']}\t"
                           f"{result['na_ions']['count']}\t{result['cl_ions']['count']}\t"
                           f"{result['tio2_surface_o']['count']}\t{result['tio2_surface_o']['avg_z']:.6f}\n")
            
            # Save molecular integrity statistics
            integrity_file = os.path.join(output_dir, "molecular_integrity_statistics.txt")
            with open(integrity_file, 'w') as f:
                f.write("Frame\tTiO2_H_Valid\tTiO2_H_Invalid\tSurface_OH_Valid\tSurface_OH_Invalid\t"
                       f"Surface_H2O_Valid\tSurface_H2O_Invalid\t"
                       f"Bulk_OH_Valid\tBulk_OH_Invalid\tBulk_H3O_Valid\tBulk_H3O_Invalid\n")
                for i, result in enumerate(all_results):
                    integrity_report = validate_molecular_integrity(result, result['frame'])
                    f.write(f"{result['frame']}\t"
                           f"{integrity_report['tio2_surface_h']['valid']}\t{integrity_report['tio2_surface_h']['invalid']}\t"
                           f"{integrity_report['solution_surface_oh']['valid']}\t{integrity_report['solution_surface_oh']['invalid']}\t"
                           f"{integrity_report['solution_surface_h2o']['valid']}\t{integrity_report['solution_surface_h2o']['invalid']}\t"
                           f"{integrity_report['solution_bulk_oh']['valid']}\t{integrity_report['solution_bulk_oh']['invalid']}\t"
                           f"{integrity_report['solution_bulk_h3o']['valid']}\t{integrity_report['solution_bulk_h3o']['invalid']}\n")
            
            logging.info(f"Statistics saved to: {stats_file}")
            logging.info(f"Molecular integrity statistics saved to: {integrity_file}")
            
            # Save TiO2 surface O z-coordinate statistics
            tio2_o_file = os.path.join(output_dir, "tio2_surface_o_z_coordinates.txt")
            with open(tio2_o_file, 'w') as f:
                f.write("Frame\tTiO2_Surface_O_Count\tAverage_Z_Coordinate\n")
                for result in all_results:
                    f.write(f"{result['frame']}\t{result['tio2_surface_o']['count']}\t"
                           f"{result['tio2_surface_o']['avg_z']:.6f}\n")
            
            logging.info(f"TiO2 surface O z-coordinates saved to: {tio2_o_file}")
            
            # Generate evolution plots
            if len(all_results) > 1:
                evolution_plot = os.path.join(output_dir, "ion_species_evolution.png")
                plot_species_evolution(all_results, evolution_plot)
                
                # Compute and visualize the transition matrix
                logging.info("\n=== Computing Species Transition Matrix ===")
                transition_counts, species_names, transition_details = build_species_transition_matrix(all_results)
                
                # Plot the transition matrix
                transition_plot = os.path.join(output_dir, "species_transition_matrix.png")
                plot_transition_matrix(transition_counts, species_names, transition_plot)
                
                # Save transition matrix data
                save_transition_data(transition_counts, species_names, transition_details, output_dir)
            
            # Calculate aggregate molecular integrity statistics
            total_integrity_stats = {
                'tio2_h': {'valid': 0, 'invalid': 0},
                'surface_oh': {'valid': 0, 'invalid': 0},
                'surface_h2o': {'valid': 0, 'invalid': 0},
                'bulk_oh': {'valid': 0, 'invalid': 0},
                'bulk_h3o': {'valid': 0, 'invalid': 0}
            }
            
            for result in all_results:
                integrity_report = validate_molecular_integrity(result, result['frame'])
                total_integrity_stats['tio2_h']['valid'] += integrity_report['tio2_surface_h']['valid']
                total_integrity_stats['tio2_h']['invalid'] += integrity_report['tio2_surface_h']['invalid']
                total_integrity_stats['surface_oh']['valid'] += integrity_report['solution_surface_oh']['valid']
                total_integrity_stats['surface_oh']['invalid'] += integrity_report['solution_surface_oh']['invalid']
                total_integrity_stats['surface_h2o']['valid'] += integrity_report['solution_surface_h2o']['valid']
                total_integrity_stats['surface_h2o']['invalid'] += integrity_report['solution_surface_h2o']['invalid']
                total_integrity_stats['bulk_oh']['valid'] += integrity_report['solution_bulk_oh']['valid']
                total_integrity_stats['bulk_oh']['invalid'] += integrity_report['solution_bulk_oh']['invalid']
                total_integrity_stats['bulk_h3o']['valid'] += integrity_report['solution_bulk_h3o']['valid']
                total_integrity_stats['bulk_h3o']['invalid'] += integrity_report['solution_bulk_h3o']['invalid']
            
            # Report overall statistics
            logging.info("\n=== Overall Statistics ===")
            avg_tio2_h = np.mean([r['tio2_surface_h']['count'] for r in all_results])
            avg_surface_oh = np.mean([r['solution_surface_oh']['count'] for r in all_results])
            avg_surface_h2o = np.mean([r['solution_surface_h2o']['count'] for r in all_results])
            avg_bulk_oh = np.mean([r['solution_bulk_oh']['count'] for r in all_results])
            avg_bulk_h3o = np.mean([r['solution_bulk_h3o']['count'] for r in all_results])
            avg_na_ions = np.mean([r['na_ions']['count'] for r in all_results])
            avg_cl_ions = np.mean([r['cl_ions']['count'] for r in all_results])
            avg_tio2_surface_o = np.mean([r['tio2_surface_o']['count'] for r in all_results])
            avg_tio2_surface_o_z = np.mean([r['tio2_surface_o']['avg_z'] for r in all_results if r['tio2_surface_o']['avg_z'] > 0])
            
            logging.info(f"TiO2 Surface Adsorbed H - Average count: {avg_tio2_h:.2f} (valid molecules total: {total_integrity_stats['tio2_h']['valid']}, invalid: {total_integrity_stats['tio2_h']['invalid']})")
            logging.info(f"Surface Adsorbed OH (from solution) - Average count: {avg_surface_oh:.2f} (valid molecules total: {total_integrity_stats['surface_oh']['valid']}, invalid: {total_integrity_stats['surface_oh']['invalid']})")
            logging.info(f"Surface Adsorbed H2O (from solution) - Average count: {avg_surface_h2o:.2f} (valid molecules total: {total_integrity_stats['surface_h2o']['valid']}, invalid: {total_integrity_stats['surface_h2o']['invalid']})")
            logging.info(f"Bulk OH in solution - Average count: {avg_bulk_oh:.2f} (valid molecules total: {total_integrity_stats['bulk_oh']['valid']}, invalid: {total_integrity_stats['bulk_oh']['invalid']})")
            logging.info(f"Bulk H3O in solution - Average count: {avg_bulk_h3o:.2f} (valid molecules total: {total_integrity_stats['bulk_h3o']['valid']}, invalid: {total_integrity_stats['bulk_h3o']['invalid']})")
            logging.info(f"Na+ ions - Average count: {avg_na_ions:.2f}")
            logging.info(f"Cl- ions - Average count: {avg_cl_ions:.2f}")
            logging.info(f"TiO2 Surface Adsorbed O - Average count: {avg_tio2_surface_o:.2f}")
            if not np.isnan(avg_tio2_surface_o_z):
                logging.info(f"TiO2 Surface Adsorbed O - Average Z coordinate: {avg_tio2_surface_o_z:.3f} Å")
            
            # Compute and report overall data quality
            total_valid = sum(stats['valid'] for stats in total_integrity_stats.values())
            total_invalid = sum(stats['invalid'] for stats in total_integrity_stats.values())
            if total_valid + total_invalid > 0:
                quality_percentage = total_valid / (total_valid + total_invalid) * 100
                logging.info(f"\n=== Data Quality Report ===")
                logging.info(f"Total molecules: {total_valid + total_invalid}, valid: {total_valid}, invalid: {total_invalid}")
                logging.info(f"Data quality: {quality_percentage:.1f}%")
            
        else:
            logging.error("Currently only supports LAMMPS format for multi-frame analysis")
            return
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\nAnalysis completed! Total time: {duration}")
        logging.info(f"Results saved in directory: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 