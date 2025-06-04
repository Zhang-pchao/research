import os
import numpy as np

def lmp_index(trj_file):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    
    xyz_index = []
    i = 0
    for line in lines:
        i += 1
        if "ITEM: ATOMS id type x y z" in line:
            xyz_index.append(i) 
    
    j = 0
    for line in lines:
        j += 1
        if "ITEM: NUMBER OF ATOMS" in line:
            atoms = lines[j].split()[0]
            break
            
    return xyz_index, atoms

def write_gjf(xyz_index, atoms, trj_file, gjf_file):
    searchfile = open(trj_file, "r")
    lines = searchfile.readlines()
    searchfile.close()
    
    for i in xyz_index:
        file = open(gjf_file, 'a+')
        file.write('%nprocshared=20\n')
        file.write('%mem=30GB\n')
        file.write('# M062X gen int=ultrafine\n')
        file.write('\n')
        file.write('Template file\n')
        file.write('\n')
        file.write('0 1\n')
        
        for j in range(int(atoms)):
            line_split = lines[i+j].split()
            if line_split[1] == "1": # O
                file.write('O %13.8f%13.8f%13.8f \n' %(float(line_split[2]), float(line_split[3]), float(line_split[4])))
            if line_split[1] == "2": # 2
                file.write('H %13.8f%13.8f%13.8f \n' %(float(line_split[2]), float(line_split[3]), float(line_split[4])))
        file.write('\n')
        file.write('@/data/HOME_BACKUP/pengchao/deepmd2/ipi_dp/ipi_water_dp/train03_seed1_d500w/10000/345H2O_gaussian_sp/ma-TZVP.txt\n')
        file.write('\n')
        
        if i != xyz_index[-1]:
            file.write('--link1--\n')
        else:
            file.write('\n')
    file.close()     

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

#trj_file = "4H2O.lammpstrj"
#gjf_file = "4H2O.gjf"
#xyz_index, atoms = lmp_index(trj_file)
#print(xyz_index, atoms)
#write_gjf(xyz_index, atoms, trj_file, gjf_file)

imq_dict = {1: [1.008, 1,'H'],
            2: [15.999,6,'O'],
            3: [14.007,5,'N'],
            4: [12.011,4,'C'],}

path        = '/data/HOME_BACKUP/pengchao/glycine/deepks/M062X_Dataset/Glycine54H2O_efield/lammpstrj/012_r011'
trj_file = os.path.join(path, 'glycine_1b_real.lammpstrj')
xyz_file = os.path.join(path, 'glycine_1b_real.xyz')

xyz_index, atoms = lmp_index(trj_file)
#print(xyz_index, atoms)

write_xyz(xyz_index[:4000:10], atoms, trj_file, xyz_file, imq_dict)