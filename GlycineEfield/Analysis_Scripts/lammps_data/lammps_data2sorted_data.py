# %%
import os
import numpy as np

# %%
def lmp_index(trj_file):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    
    i = 0
    for line in lines:
        i += 1
        if "Atoms # atomic" in line:
            xyz_index = i+1
    
    j = 0
    for line in lines:
        j += 1
        if "LAMMPS data file via write_data" in line:
            atoms = lines[j+1].split()[0]
            break
            
    return xyz_index, atoms

# %%
def write_data(xyz_index, atoms, trj_file, gjf_file):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    file = open(gjf_file, 'w+')
    for i in range(xyz_index):
        file.write(lines[i])

    for j in range(atoms):
        #print(j+1)
        ii=0
        # sorted the output data file
        while True:
            line_split = lines[xyz_index+ii].split()
            atom_idx = int(line_split[0])
            #print(j+1,atom_idx,lines[xyz_index+ii])
            if atom_idx == j+1:
              file.write(lines[xyz_index+ii])
              break
            ii+=1
    
    file.close()    

path        = '/pengchao/glycine/dplr/dpmd/efield_0.01/001/1.2'

geo_data1 = os.path.join(path, '1200000.data')
geo_data2 = os.path.join(path, '1200000_atomic.data')

# %%
xyz_index, atoms = lmp_index(geo_data1)
print(xyz_index, atoms)

# %%
write_data(int(xyz_index), int(atoms), geo_data1, geo_data2)

# %%



