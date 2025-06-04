#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys
import glob
import shutil

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 

def get_dirlist(dirct):
    dirList = []
    files=os.listdir(dirct)
    for f in files:
        if os.path.isdir(dirct + '/' + f):
            dirList.append(f)
    return dirList

def read_atom_num(file):
    with open(file) as f: 
        lines = f.readlines()[0]
        sp = lines.split()[0]
    return int(sp)

def read_cell(file):
    cell = []
    with open(file) as f: 
        lines = f.readlines()[:]
        idx = 0
        for line in lines:
            idx += 1
            if "&CELL" in line:
                break
        for i in range(3):
            sp = lines[idx+i].split()
            cell.append([float(sp[1]),float(sp[2]),float(sp[3])])
    return cell

def read_xyz(file,typemap,atom_num):
    total = []
    with open(file) as f: 
        lines = f.readlines()[2:]
        for k in typemap:
            for i in range(atom_num):
                sp = lines[i].split()
                if sp[0] == k:
                    for j in range(3):
                        total.append(float(sp[j+1]))
    return total

def read_force(file,xyz_index,atom_num,typemap):
    total = []
    with open(file) as f: 
        lines = f.readlines()[:]
        for frames in xyz_index:
            for k in typemap:
                for i in range(atom_num):
                    sp = lines[frames+i].split()
                    if sp[0] == k:
                        for j in range(3):
                            total.append(float(sp[j+1]))        
    return total

def read_energy(file,xyz_index):
    total = []
    with open(file) as f: 
        lines = f.readlines()[:]
        for frames in xyz_index:
            sp = lines[frames-1].split()
            total.append(float(sp[1]))
    return total

def get_atom_index(trj_file):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    
    atoms = int(lines[0].split()[0])
    
    xyz_index = []
    i = 0
    for line in lines:
        i += 1
        if "Title" in line:
            xyz_index.append(i)    
            
    return xyz_index,atoms
    
def write_xyz(xyz_index, atoms, trj_file, gjf_file, imq_dict):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    
    for i in xyz_index:
        file = open(gjf_file, 'a+')
        file.write('%d\n'%int(atoms))
        file.write('Title\n')
        
        for j in range(int(atoms)):
            line_split = lines[i+j].split()
            atom_idx = int(line_split[1])
            file.write('%s %13.8f%13.8f%13.8f \n' %(imq_dict[atom_idx][-1],float(line_split[2]), float(line_split[3]), float(line_split[4])))

    file.close() 

def type_raw(file, typemap, type_path, map_path):
    w_map = open(map_path, 'w')
    for i in typemap:
        w_map.write(i + '\n')
    w_map.close()
    
    w_type = open(type_path, 'w')
    with open(file) as f: 
        lines = f.readlines()[2:]
        for k in range(len(typemap)):
            for i in range(atom_num):
                sp = lines[i].split()[0]
                if sp == typemap[k]:
                    w_type.write(str(k) + '\n') 
    w_type.close()

def check_npy(loadData):
    print(type(loadData))
    print(loadData.dtype)
    print(loadData.ndim)
    print(loadData.shape)
    print(loadData.size)
    print(loadData)    

# input data path here, string, this directory should contains
data_path = "/data/HOME_BACKUP/pengchao/glycine/deepks/M062X_Dataset/Glycine54H2O_efield/efield_test/lammpstrj"
xyz_name = 'glycine_10_real_step06.xyz'
out_name = 'glycine_10_real_force_step06.xyz'

typemap   = ["H","O","N","C"]
imq_dict = {1: [1.008, 1,'H'],
            2: [15.999,6,'O'],
            3: [14.007,5,'N'],
            4: [12.011,4,'C'],}

all_dir   = ["010-test-a-efild-0.005"]

for i in all_dir:
    dirct     = os.path.join(data_path,i)
    xyz_file = os.path.join(dirct, xyz_name)
    out_file = os.path.join(dirct, out_name)
    xyz_index,atom_num = get_atom_index(xyz_file)
    
    save_path1 = os.path.join(data_path,  "dp_dataset_lmp",i)
    save_path2 = os.path.join(save_path1, "set.000")
    mkdir(save_path2)

    force_path = os.path.join(save_path2,  "force.npy")
    energy_path= os.path.join(save_path2,  "energy.npy")
    
    total_force   = []
    total_energy  = []

    total_force.append(read_force(out_file,xyz_index,atom_num,typemap))
    total_energy.append(read_energy(out_file,xyz_index))

    total_force_np = np.asarray(total_force,dtype=float).reshape(len(xyz_index),atom_num*3)
    np.save(force_path, total_force_np)
    
    total_energy_np = np.asarray(total_energy,dtype=float).reshape(len(xyz_index),1)
    np.save(energy_path, total_energy_np)   
    
check_npy(loadData = np.load(os.path.join(save_path2,'force.npy')))
check_npy(loadData = np.load(os.path.join(save_path2,'energy.npy')))