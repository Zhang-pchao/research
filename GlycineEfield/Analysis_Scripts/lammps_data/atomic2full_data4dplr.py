import os
import numpy as np
from ase.io import read,write
from ase.data import atomic_numbers

path        = '/pengchao2/glycine/dplr/geo/Anion_Glycine128H2O_full_slab'
atomic_data = os.path.join(path, 'glycine_a_1_atomic.data')
full_data   = os.path.join(path, 'glycine_a_1_full.data')

atoms=read(atomic_data,format="lammps-data",style="atomic",)  

imq_dict = {1: [1.008, 1,'H'],
            2: [15.999,6,'O'],
            3: [14.007,5,'N'],
            4: [12.011,4,'C'],
            5: [15.999,-8,'O',1],
            6: [14.007,-8,'N',2],
            7: [12.011,-3,'C',3],            
            } # index mass charge bond_type
atom_wac = [4,3] # real_atom wannier_centroid

totalatomnum = len(atoms)
atomtypenum  = len(set(atoms.get_atomic_numbers()))

atomtypenum_list = []
for j in range(atomtypenum):
    atomtypenum_list.append(len([i for i in atoms.get_atomic_numbers() if i==j+1]))

bondtypenum = sum(atomtypenum_list[atom_wac[0]-atom_wac[1]:])
atomtype_list = atoms.get_atomic_numbers()

file = open(full_data, 'w')
file.write('LAMMPS data file for ')
file.write(full_data)
file.write('\n\n')
file.write('%10d atoms\n' %(totalatomnum+bondtypenum))   
file.write('%10d bonds\n' %bondtypenum) 
if atomtypenum != atom_wac[0]:
    print('Warning: atom type number not equal to atom_wac[0]')

file.write('%10d atom types\n' %(atomtypenum+atom_wac[1]))
file.write('%10d bond types\n' %atom_wac[1])

file.write('\n%10.4f%10.4f xlo xhi\n' %(0., atoms.cell[0][0]))
file.write('%10.4f%10.4f ylo yhi\n'   %(0., atoms.cell[1][1]))
file.write('%10.4f%10.4f zlo zhi\n\n' %(0., atoms.cell[2][2]))

file.write('Masses\n\n')
for i in range(sum(atom_wac)):
    file.write('%10d' %(i+1))
    file.write('%10.4f' %float(imq_dict[i+1][0]))
    if i < atom_wac[0]:
        file.write('  # %4s\n' %imq_dict[i+1][2])
    else:
        file.write('  # %4s wannier centroid\n' %imq_dict[i+1][2])

# atomic: atom-ID atom-type x y z
# full: atom-ID molecule-ID atom-type q x y z
file.write('\nAtoms')
file.write('  # full # atom-ID molecule-ID atom-type q x y z\n\n')
k = 0
# for read atoms
for j in range(0, totalatomnum):
    file.write(' %6i ' % (k + 1)) # atom-ID
    file.write(' %3i ' % (k + 1)) # molecule-ID    
    file.write(' %3i '% atomtype_list[j]) # atom-type   
    file.write(' %3.4f ' % float(imq_dict[atomtype_list[j]][1])) # q

    file.write(' %15.9f ' % float(atoms.positions[j][0])) #x
    file.write(' %15.9f ' % float(atoms.positions[j][1])) #y
    file.write(' %15.9f ' % float(atoms.positions[j][2])) #z  
    file.write(' #%2s\n' % imq_dict[atomtype_list[j]][2])    
    k = k + 1

bond_pair = []
# only fot wannier centroids
for j in range(0, totalatomnum):
    if atomtype_list[j] != 1: # skip H
        bond_pair.append([j+1,k+1,imq_dict[atomtype_list[j]+atom_wac[1]][-1]])
        file.write(' %6i ' % (k + 1)) # atom-ID
        file.write(' %3i ' % (k + 1)) # molecule-ID    
        file.write(' %3d '% (int(atomtype_list[j])+atom_wac[1])) # atom-type   
        file.write(' %3.4f ' % float(imq_dict[atomtype_list[j]+atom_wac[1]][1])) # q

        file.write(' %15.9f ' % float(atoms.positions[j][0])) #x
        file.write(' %15.9f ' % float(atoms.positions[j][1])) #y
        file.write(' %15.9f ' % float(atoms.positions[j][2])) #z  
        file.write(' #%2s wc\n' % imq_dict[atomtype_list[j]][2])    
        k = k + 1

file.write('\nBonds\n\n')
for ii in range(len(bond_pair)):
    file.write('%8d%8d%8d%8d\n' %(ii+1,bond_pair[ii][-1],bond_pair[ii][0],bond_pair[ii][1]))
file.close()