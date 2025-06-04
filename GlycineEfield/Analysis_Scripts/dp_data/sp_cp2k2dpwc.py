#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os, sys

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 

def get_dirlist(dirct):
    dirList = []
    files=os.listdir(dirct)
    for f in sorted(files):
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
    
def read_xyz_old(file,atom_num):
    total = []
    with open(file) as f: 
        lines = f.readlines()[2:]
        for i in range(atom_num):
            sp = lines[i].split()
            for j in range(3):
                total.append(float(sp[j+1]))
    return total

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

def read_force_old(file,atom_num):
    total = []
    with open(file) as f: 
        lines = f.readlines()[:]
        idx = 0
        for line in lines:
            idx += 1
            if "ATOMIC FORCES in" in line:
                break
        for i in range(atom_num):
            sp = lines[idx+2+i].split()
            for j in range(3):
                total.append(float(sp[j+3]))
    return total

def read_force(file,typemap,atom_num):
    total = []
    with open(file) as f: 
        lines = f.readlines()[:]
        idx = 0
        for line in lines:
            idx += 1
            if "ATOMIC FORCES in" in line:
                break
        for k in typemap:
            for i in range(atom_num):
                sp = lines[idx+2+i].split()
                if sp[2] == k:
                    for j in range(3):
                        total.append(float(sp[j+3]))        
    return total

def read_energy(file):
    total = []
    with open(file) as f: 
        lines = f.readlines()[:]
        for line in lines:
            if "Total energy:" in line:
                sp = line.split()
                total.append(float(sp[2]))
    return total 

def type_raw_old(file, typemap, type_path, map_path):
    w_map = open(map_path, 'w')
    for i in typemap:
        w_map.write(i + '\n')
    w_map.close()
    
    w_type = open(type_path, 'w')
    with open(file) as f: 
        lines = f.readlines()[2:]
        for i in range(atom_num):
            sp = lines[i].split()[0]
            for k in range(len(typemap)):
                if sp == typemap[k]:
                    w_type.write(str(k) + '\n') 
    w_type.close()

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

def get_relate_wannier_xyz(ion_xyz,wac_xyz,cell,wcs=4):
    cell = np.array([cell[0][0],cell[1][1],cell[2][2]])
    relate_xyz = np.zeros((ion_xyz.shape[0],ion_xyz.shape[1]))
    for i in range(ion_xyz.shape[0]):
        wc = {key: np.array([2,2,2]) for key in range(wcs)}
        for j in range(wac_xyz.shape[0]):
            #coords_diff = ion_xyz[i]-wac_xyz[j]
            coords_diff = wac_xyz[j]-ion_xyz[i]
            scaled_coords = coords_diff / cell
            frac_coords = scaled_coords - np.floor(scaled_coords + 0.5)
            pbc_coords = frac_coords * cell
            dist = np.linalg.norm(pbc_coords)
            if dist < np.linalg.norm(wc[0]):
                wc[3] = wc[2]
                wc[2] = wc[1]
                wc[1] = wc[0]
                wc[0] = pbc_coords
            elif dist < np.linalg.norm(wc[1]):
                wc[3] = wc[2]
                wc[2] = wc[1]
                wc[1] = pbc_coords
            elif dist < np.linalg.norm(wc[2]):
                wc[3] = wc[2]
                wc[2] = pbc_coords
            elif dist < np.linalg.norm(wc[3]):
                wc[3] = pbc_coords
        
        if np.linalg.norm(wc[3]) > 0.7:
            print('Warning: wannier center is too far from ion %d'%(i))
        relate_xyz[i] = (wc[3]+wc[2]+wc[1]+wc[0])/4 
    return relate_xyz

def read_relate_wannier_xyz(file,typemap,atom_num,cell):
    total_xyz = []
    typemap_num = {key: 0 for key in typemap}
    with open(file) as f: 
        lines = f.readlines()[2:]
        for k in typemap:
            for i in range(atom_num):
                sp = lines[i].split()
                if sp[0] == k:
                    typemap_num[k] += 1
                    for j in range(3):
                        total_xyz.append(float(sp[j+1]))

    ion_xyz = np.asarray(total_xyz[:-3*typemap_num['X']],dtype=float).reshape(-1,3)
    wac_xyz = np.asarray(total_xyz[-3*typemap_num['X']:],dtype=float).reshape(-1,3)
    relate_xyz = get_relate_wannier_xyz(ion_xyz,wac_xyz,cell)
    return relate_xyz,ion_xyz.shape[0]
    

# conversion unit here, modify if you need
au2eV = 27.211386245988
au2A = 0.529177249
# input data path here, string, this directory should contains
data_path = "./"
typemap   = ["H","O","N","C"]
typemap_wc   = ["O","N","C","X"]
all_dir   = [
"001",
"002",
"003",
"004",
"005",
"006",
"007",
"008",
"009",
"010",
"011",
"012",
]
for i in all_dir:
    dirct     = os.path.join(data_path,i)
    dir_list  = get_dirlist(dirct)
    
    save_path1 = os.path.join(data_path,  "dp_dataset_dpwc_rev_loc",i)
    save_path2 = os.path.join(save_path1, "set.000")
    mkdir(save_path2)
    type_path  = os.path.join(save_path1,  "type.raw")
    map_path   = os.path.join(save_path1,  "type_map.raw")
    box_path   = os.path.join(save_path2,  "box.npy")
    coord_path = os.path.join(save_path2,  "coord.npy")
    force_path = os.path.join(save_path2,  "force.npy")
    energy_path= os.path.join(save_path2,  "energy.npy")
    atom_dipole_path= os.path.join(save_path2,  "atomic_dipole.npy")
    
    choose_frames = dir_list[:]
    total_xyz     = []
    total_box     = []
    total_force   = []
    total_energy  = []
    total_atom_dipole  = []

    for j in choose_frames:   
        print(j)
        cp2k_path = os.path.join(dirct,j)
        cp2k_xyz  = os.path.join(cp2k_path,"POSCAR1.xyz")
        cp2k_inp  = os.path.join(cp2k_path,"m062x/m062x.inp")
        cp2k_out  = os.path.join(cp2k_path,"m062x/m062x.out")
        wannier_center_out  = os.path.join(cp2k_path,"m062x/M062X-HOMO_centers_s1-1_0.xyz")
        atom_num = read_atom_num(cp2k_xyz)
        atom_wc_num = read_atom_num(wannier_center_out)
        cell = read_cell(cp2k_inp)
        total_xyz.append(read_xyz(cp2k_xyz,typemap,atom_num))
        _atom_dipole, _ion_num = read_relate_wannier_xyz(wannier_center_out,typemap_wc,atom_wc_num,np.asarray(cell))
        total_atom_dipole.append(_atom_dipole)
        total_force.append(read_force(cp2k_out,typemap,atom_num))
        total_energy.append(read_energy(cp2k_out))
        total_box.append(cell)
    
    type_raw(cp2k_xyz, typemap, type_path, map_path)
    
    total_xyz_np = np.asarray(total_xyz,dtype=float).reshape(len(choose_frames),atom_num*3)
    np.save(coord_path, total_xyz_np)

    total_atom_dipole_np = np.asarray(total_atom_dipole,dtype=float).reshape(len(choose_frames),_ion_num*3)
    np.save(atom_dipole_path, total_atom_dipole_np)    

    total_force_np = np.asarray(total_force,dtype=float).reshape(len(choose_frames),atom_num*3)
    np.save(force_path, total_force_np*au2eV/au2A)
    
    total_energy_np = np.asarray(total_energy,dtype=float).reshape(len(choose_frames),1)
    np.save(energy_path, total_energy_np*au2eV)   
    
    total_box_np = np.asarray(total_box,dtype=float).reshape(len(choose_frames),9)
    np.save(box_path, total_box_np)  

check_npy(loadData = np.load(os.path.join(save_path2,'atomic_dipole.npy')))
#check_npy(loadData = np.load(os.path.join(save_path2,'coord.npy')))
#check_npy(loadData = np.load(os.path.join(save_path2,'box.npy')))
#check_npy(loadData = np.load(os.path.join(save_path2,'force.npy')))
#check_npy(loadData = np.load(os.path.join(save_path2,'energy.npy')))