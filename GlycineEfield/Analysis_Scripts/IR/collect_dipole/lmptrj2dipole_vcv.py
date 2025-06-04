import os
import numpy as np
import scipy
from scipy import constants
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib import distances
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.lib.distances import capped_distance

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
def calculate_dipole_moment_h2o(shifted_h_positions,shifted_o_positions,shifted_wc_positions,charge,count_h=2):
    # The contribution of each atom type to the dipole moment is weighted.
    dipole_moment = np.zeros([3,])
    for i in range(count_h):
        dipole_moment += shifted_h_positions[i]*charge[0]
    dipole_moment += shifted_o_positions[0]*charge[1] 
    dipole_moment += shifted_wc_positions[0]*charge[2] 
    return dipole_moment

def calculate_dipole_moment_gly(h_positions,nco_positions,wc_positions,h_charge_list,nco_charge_list,wc_charge_list):
    dipole_moment = np.zeros([3,])
    for i in range(len(h_positions)):
        dipole_moment += h_positions[i]*h_charge_list[0]
    for i in range(len(nco_positions)):
        dipole_moment += nco_positions[i]*nco_charge_list[i]
    for i in range(len(wc_positions)):
        dipole_moment += wc_positions[i]*wc_charge_list[i]        
    return dipole_moment

def get_distance(u,atom_indices,L=12.028,pbc=True):
    atom_positions = u.atoms.positions[atom_indices]
    u.dimensions = np.array([L, L, L, 90, 90, 90])
    #bond_length = mda.lib.distances.distance_array(atom_positions[0], atom_positions[1], 
    #                                                   box=u.dimensions)[0][0]
    diff = atom_positions[0] - atom_positions[1]
    if pbc:
        diff = diff - np.round(diff / u.dimensions[:3]) * u.dimensions[:3]
    bond = np.sqrt(np.sum(diff ** 2))
    return diff,bond

def get_voronoi(hatoms,oatoms,u,dim=3):
    # lambda parameter of the switching function
    l = 500

    OHcouple = np.zeros([len(oatoms),dim])*np.nan
    for j in range(len(hatoms)):
        sum = 0
        w = np.zeros(len(oatoms))
        for i in range(len(oatoms)):
            d,dmod = get_distance(u,[int(oatoms[i].index),int(hatoms[j].index)],L=u.dimensions[0])
            sum = sum + np.exp(-l*dmod)
        for i in range(len(oatoms)):
            d,dmod = get_distance(u,[int(oatoms[i].index),int(hatoms[j].index)],L=u.dimensions[0])
            w[i] = np.exp(-l*dmod) / sum
        #print(np.where(w>0.8))
        if np.size(np.where(w>0.8)) == 0: # voronoi cv failed
            break
        else:
            ohcouple = np.where(w>0.8)[0][0]
            for i in range(dim):
                if np.isnan(OHcouple[ohcouple][i]):
                    OHcouple[ohcouple][i] = j
                    break
    if np.size(np.where(w>0.8)) == 0: # voronoi cv failed
        return np.zeros([3,3])*np.nan,False
    else:
        return OHcouple,True

def calculate_shifted_positions(atom_indices, atoms, center_pos, box_length, shift_flag=True):
    shifted_positions  = []
    original_positions = []
    if shift_flag:
        for idx in atom_indices:
            original_pos = atoms[idx].position
            shifted_pos = pbc_shift(original_pos, center_pos, box_length)
            shifted_positions.append(shifted_pos)
        return shifted_positions
    else:
        for idx in atom_indices:
            original_pos = atoms[idx].position
            original_positions.append(original_pos)
        return original_positions        

geo_path = '/home/pengchao/glycine/dpmd/glycine/046_dpmd/003_128w_gly_bulk_e/opes_flooding/z2a/140/'
trj_path = '../'
save_path = './'
_geo_file = 'glycine_full_restart4mda.data'
_trj_file = 'glycine_1_all.lammpstrj'
geo_file = os.path.join(geo_path,_geo_file)
trj_file = os.path.join(trj_path,_trj_file)

u = mda.Universe(geo_file,trj_file, atom_style='id type x y z',format='LAMMPSDUMP')
traj_num    = len(u.trajectory)
trj_step    = 1
trj_skip    = 2*100000
choose_frames = range(trj_skip,traj_num,trj_step)

all_h_atoms = u.select_atoms("type 1")
o_n_c_atoms = u.select_atoms("type 2 or type 3 or type 4") # last 5 atoms are in glycine

# Loop over each o_n_c atom to find its two closest hydrogens
# for [A] and [C], Voronoi CVs are needed to find H3O and OH!!!!!!
for o_n_c in o_n_c_atoms:
    wc_idx = bond_o_wc(o_n_c.index,u.bonds)
    #h2o_bonds.append((o_n_c.index, closest_hydrogens[0].index, closest_hydrogens[1].index, wc_idx))

total_all_dipole = []
total_gly_dipole = []
total_h2o_dipole = []
L = u.dimensions[0] # note: only for NVT cubic box!!!!!!!!!!!!!!!!!!!!!!
vol = L**3
for j in choose_frames:
#for j in [99,100]:
    u.trajectory[j]
    # Initialize lists to store results
    dipole_moments = []

    ONCHcouple,HFlag = get_voronoi(all_h_atoms,o_n_c_atoms,u)
    OHcouple = ONCHcouple[:-5] # water
    GHcouple = ONCHcouple[-5:] # glycine
    for x in range(np.shape(OHcouple)[0]):
        nan_mask = np.isnan(OHcouple[x])
        count_h  = np.count_nonzero(~nan_mask) # count_not_nan
        #print(count_h)
        o_idx = o_n_c_atoms[:-5][x].index
        wc_idx = bond_o_wc(o_n_c_atoms[:-5][x].index,u.bonds)
        oxygen_indices = [o_idx]
        wc_indices = [wc_idx]

        hydrogen_indices = []
        for i in range(count_h):
            hydrogen_indices.append(all_h_atoms[int(OHcouple[x][i])].index)
        
        center_pos  = u.atoms[oxygen_indices[0]].position
        shifted_o_positions  = calculate_shifted_positions(oxygen_indices,   u.atoms, center_pos, L)
        shifted_h_positions  = calculate_shifted_positions(hydrogen_indices, u.atoms, center_pos, L)
        shifted_wc_positions = calculate_shifted_positions(wc_indices,       u.atoms, center_pos, L)
        
        charge_list = [1,6,-8] # H,O,Ow
        dipole_moment = calculate_dipole_moment_h2o(shifted_h_positions,
                                                    shifted_o_positions,
                                                    shifted_wc_positions,
                                                    charge_list,count_h)
        dipole_moments.append(dipole_moment)
    
    all_h2o_dipole = np.sum(dipole_moments, axis=0)


    nco_indices = []
    wc_indices  = []
    hydrogen_indices = []
    for x in range(np.shape(GHcouple)[0]): # glycine
        nan_mask = np.isnan(GHcouple[x])
        count_h  = np.count_nonzero(~nan_mask) # count_not_nan

        nco_idx = o_n_c_atoms[-5:][x].index
        wc_idx = bond_o_wc(o_n_c_atoms[-5:][x].index,u.bonds)
        nco_indices.append(nco_idx)
        wc_indices.append(wc_idx)

        if count_h > 0:            
            for i in range(count_h):
                hydrogen_indices.append(all_h_atoms[int(GHcouple[x][i])].index)


    center_pos  = [0,0,0]
    nco_positions= calculate_shifted_positions(nco_indices,      u.atoms, center_pos, L,shift_flag=False)
    h_positions  = calculate_shifted_positions(hydrogen_indices, u.atoms, center_pos, L,shift_flag=False)
    wc_positions = calculate_shifted_positions(wc_indices,       u.atoms, center_pos, L,shift_flag=False)

    nco_charge_list = [5,4,4,6,6] # sort: N C C O O
    h_charge_list   = [1] # H
    wc_charge_list  = [-8,-3,-3,-8,-8] # sort: N C C O O
    gly_dipole = calculate_dipole_moment_gly(h_positions,
                                             nco_positions, # sort: N C C O O
                                             wc_positions,  # sort: N C C O O
                                             h_charge_list,
                                             nco_charge_list,
                                             wc_charge_list,
                                             ) 
    
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