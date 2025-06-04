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
        if "Atoms # full" in line:
            xyz_index = i+1
            break
           
    j = 0
    for line in lines:
        j += 1
        if "LAMMPS data file via write_data" in line:
            atoms = lines[j+1].split()[0]
            bonds = lines[j+3].split()[0]
            break
        
    k = 0
    for line in lines:
        k += 1
        if "Velocities" in line:
            vel_index = k-1
            break
            
    return int(xyz_index), int(atoms), int(vel_index), int(bonds)

# %%
def write_data(xyz_index, atoms, vel_index, bonds, trj_file, gjf_file):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    file = open(gjf_file, 'w+')
    for i in range(xyz_index):
        file.write(lines[i])

    for j in range(atoms-bonds):
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

# %%
def read_real_xyz(xyz_index, atoms, bonds, trj_file):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    xyz =[]
    for i in range(atoms-bonds):
        line_split = lines[xyz_index+i].split()
        atom_idx = int(line_split[2])
        if atom_idx!= 1:
            xyz.append([line_split[4], line_split[5], line_split[6]])
    return xyz


# %%
def write_data_wc(xyz_index, atoms, vel_index, bonds, trj_file, gjf_file,xyz):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    file = open(gjf_file, 'a+')

    for j in range(bonds):
        #print(j+1)
        ii=0
        # sorted the output data file
        while True:
            sp = lines[xyz_index+ii].split()
            atom_idx = int(sp[0])
            #print(j+1,atom_idx,lines[xyz_index+ii])
            if atom_idx == j+atoms-bonds+1:
                file.write('%s %s %s %s %s %s %s %s %s %s\n'%(sp[0],sp[1],sp[2],sp[3],xyz[j][0],xyz[j][1],xyz[j][2],sp[7],sp[8],sp[9]))
                break
            ii+=1
    file.write('\n')
    for j in range(len(lines[vel_index:])):
        file.write(lines[j+vel_index])
    file.close()    

# %%
path        = '/pengchao2/glycine/dplr/dpmd/009_128w_gly_interface/no_efield_b40/104_com/0'
geo_data1 = os.path.join(path, '900000.data')
geo_data2 = os.path.join(path, 'glycine_full_restart.data')

# %%
xyz_index, atoms, vel_index, bonds = lmp_index(geo_data1)
print(xyz_index, atoms, vel_index, bonds)

# %%
write_data(xyz_index, atoms, vel_index, bonds, geo_data1, geo_data2)

# %%
xyz = read_real_xyz(xyz_index, atoms, bonds, geo_data2)

# %%
print(len(xyz))

# %%
write_data_wc(xyz_index, atoms, vel_index, bonds,geo_data1, geo_data2,xyz)

# %%



