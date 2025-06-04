#!/usr/bin/env python
# coding: utf-8

import os

# Read the number of atoms and the atomic data from the .xyz file
def read_xyz(xyzfilename):
    with open(xyzfilename) as f:
        lines = f.readlines()
        num_atoms = int(lines[0].strip())
        atom_data = [line.strip().split() for line in lines[2:2 + num_atoms]]
    return num_atoms, atom_data

# Output LAMMPS .data file
def output(path, filename, num_atoms, atom_dict, atom_style, atom_data, boundary):
    file = open(path+ '/'  + filename + '_' + atom_style + '.data', 'w')
    
    file.write('LAMMPS data file for ')
    file.write(filename)
    file.write('\n\n')
    file.write('%10d atoms\n\n' %num_atoms)
    file.write('%10d atom types\n\n' %len(atom_dict))
    file.write('%10.4f%10.4f xlo xhi\n' %(boundary[0], boundary[1]))
    file.write('%10.4f%10.4f ylo yhi\n' %(boundary[2], boundary[3]))
    file.write('%10.4f%10.4f zlo zhi\n\n' %(boundary[4], boundary[5]))
    
    file.write('Masses\n\n')
    for element, (atom_type, mass) in atom_dict.items():
        file.write('%10d%10.4f  # %4s\n' % (atom_type, mass, element))
    
    # atomic: atom-ID atom-type x y z
    # full: atom-ID molecule-ID atom-type q x y z
    file.write('\nAtoms')
    file.write('  # %8s\n\n' %atom_style)
    for i, atom in enumerate(atom_data, start=1):
        element, x, y, z = atom
        file.write(' %6i %3i %15.9f %15.9f %15.9f #%2s\n' % (i, atom_dict[element][0], float(x), float(y), float(z), element))
    
    file.close()

if __name__ == '__main__':
    path = '/pengchao2/glycine/dplr/dpmd/009_128w_gly_interface/efield_b40/002/2.1'
    files = os.listdir(path)
    xyzfile = None
    for f in files:
        if os.path.isfile(path + '/' + f):
            if f.endswith(".xyz"):
                xyzfile = path + '/' + f
                print("xyzfile: ", xyzfile)    
   
                filename = f.split(".")[0]
                atom_dict = {'H': [1, 1.008],'O': [2, 15.999],'N': [3, 14.007], 'C': [4, 12.011],}
                
                # atomic: atom-ID atom-type x y z
                atom_style = 'atomic'
                # full: atom-ID molecule-ID atom-type q x y z
                #atom_style = 'full'
                
                num_atoms, atom_data = read_xyz(xyzfile)
                # Define the boundary for the simulation box
                boundary = [0.0, 12.1, 0.0, 12.1, 0.0, 80.5]
                
                output(path, filename, num_atoms, atom_dict, atom_style, atom_data, boundary)

    if not xyzfile:
        raise FileNotFoundError("No .xyz file found in the directory.")
    