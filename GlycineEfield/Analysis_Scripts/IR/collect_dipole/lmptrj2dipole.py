import os
import numpy as np
import scipy
from scipy import constants
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib import distances
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.lib.distances import capped_distance

geo_path = '/home/pengchao/glycine/geofile'
trj_path = '../'
save_path = './'
_geo_file = 'ZwiGlycine128H2O_opt_full4mda.data'
_trj_file = 'glycine_1_all.lammpstrj'
geo_file = os.path.join(geo_path,_geo_file)
trj_file = os.path.join(trj_path,_trj_file)

u = mda.Universe(geo_file,trj_file, atom_style='id type x y z',format='LAMMPSDUMP')
traj_num    = len(u.trajectory)
trj_step    = 1
trj_skip    = 2*10000
choose_frames = range(trj_skip,traj_num,trj_step)

def bond_o_wc(o_idx: int,o_wc_bonds: np.ndarray) -> int:
    for bond in o_wc_bonds:
        if o_idx == bond[0].index:
            wc_idx = bond[1].index            
    return wc_idx

# Function to shift selected atom's coordinates considering PBC
def pbc_shift(atom_position, shift_center, dimensions):
    # Center the coordinates around the shift_center
    centered_coords = atom_position - shift_center
    # Apply periodic boundary conditions
    centered_coords = np.where(centered_coords > dimensions / 2, centered_coords - dimensions, centered_coords)
    centered_coords = np.where(centered_coords < -dimensions / 2, centered_coords + dimensions, centered_coords)
    return centered_coords

# Function to calculate the dipole moment based on shifted positions
def calculate_dipole_moment(coord,charge):
    # The contribution of each atom type to the dipole moment is weighted.
    dipole_moment = np.zeros([3,])
    for i in range(len(charge)):
        dipole_moment += coord[i]*charge[i] 
    return dipole_moment

hydrogen_atoms  = u.select_atoms('type 1')[:-5] # H # skip last five H atoms in glycine
oxygen_atoms    = u.select_atoms('type 2')[:-2] # O # skip last two  O atoms in glycine
glyn_atoms      = u.select_atoms('type 3')[0] # N
glynw_atoms     = u.select_atoms('type 6')[0] # Nw

# Prepare a list to store bond information
h2o_bonds = []

# Loop over each oxygen atom to find its two closest hydrogens
for oxygen in oxygen_atoms:
    # Calculate distances to all hydrogen atoms
    distances_to_hydrogens = distances.distance_array(oxygen.position, hydrogen_atoms.positions, box=u.dimensions)

    # Find the indices of the two closest hydrogens
    closest_hydrogens_indices = distances_to_hydrogens.argsort()[0][:2]

    # Get the actual hydrogen atoms using the indices
    closest_hydrogens = hydrogen_atoms[closest_hydrogens_indices]
    
    wc_idx = bond_o_wc(oxygen.index,u.bonds)
    # Save the bond pair information (oxygen index, hydrogen index 1, hydrogen index 2, wc_idx)
    h2o_bonds.append((oxygen.index, closest_hydrogens[0].index, closest_hydrogens[1].index, wc_idx))

total_all_dipole = []
total_gly_dipole = []
total_h2o_dipole = []
L = u.dimensions[0] # note: only for NVT cubic box!!!!!!!!!!!!!!!!!!!!!!
vol = L**3
for j in choose_frames:
#for j in [200,201]:
    u.trajectory[j]
    # Initialize lists to store results
    shifted_positions = []
    dipole_moments = []
    
    # Loop over each bond to shift positions and calculate dipole moment
    for bond in h2o_bonds:
        o_idx, h_idx1, h_idx2, wc_idx = bond  # Unpack the bond indices
        
        # Retrieve the positions of the atoms
        o_pos  = u.atoms[o_idx].position
        h_pos1 = u.atoms[h_idx1].position
        h_pos2 = u.atoms[h_idx2].position
        wc_pos = u.atoms[wc_idx].position  # Assuming wc atom is in `u.atoms`
        
        # Shift the positions of the atoms to the box center considering PBC
        center_pos = o_pos#-u.dimensions[:3]/2
        shifted_o_pos  = pbc_shift(o_pos,  center_pos, L)
        shifted_h_pos1 = pbc_shift(h_pos1, center_pos, L)
        shifted_h_pos2 = pbc_shift(h_pos2, center_pos, L)
        shifted_wc_pos = pbc_shift(wc_pos, center_pos, L)
        
        # Save shifted positions for each atom in the bond
        #shifted_positions.append((shifted_o_pos, shifted_h_pos1, shifted_h_pos2, shifted_wc_pos))
        
        # Calculate the dipole moment for the bond
        shift_coord = [shifted_h_pos1,shifted_h_pos2,shifted_o_pos,shifted_wc_pos]
        dipole_moment = calculate_dipole_moment(shift_coord,[1,1,6,-8])
        dipole_moments.append(dipole_moment)
    
    all_h2o_dipole = np.sum(dipole_moments, axis=0)

    gly_position = []
    for i in range(10):
        n_idx = glyn_atoms.index
        gly_position.append(u.atoms[n_idx+i].position)

    for i in range(5):
        nw_idx = glynw_atoms.index
        gly_position.append(u.atoms[nw_idx+i].position)

    gly_dipole = calculate_dipole_moment(gly_position,[5,1,4,4,1,1,6,1,6,1,-8,-3,-3,-8,-8]) 
    
    gly_h2o_dipole = gly_dipole+all_h2o_dipole
    total_all_dipole.append(gly_h2o_dipole)
    total_gly_dipole.append(gly_dipole)
    total_h2o_dipole.append(all_h2o_dipole)

#unit covert for post code
total_all_dipole = np.array(total_all_dipole)/np.sqrt(vol)*np.sqrt(0.52917721067)
total_gly_dipole = np.array(total_gly_dipole)/np.sqrt(vol)*np.sqrt(0.52917721067)
total_h2o_dipole = np.array(total_h2o_dipole)/np.sqrt(vol)*np.sqrt(0.52917721067)

np.save(os.path.join(save_path,"total_all_dipole.npy"),np.array(total_all_dipole))
np.save(os.path.join(save_path,"total_gly_dipole.npy"),np.array(total_gly_dipole))
np.save(os.path.join(save_path,"total_h2o_dipole.npy"),np.array(total_h2o_dipole))