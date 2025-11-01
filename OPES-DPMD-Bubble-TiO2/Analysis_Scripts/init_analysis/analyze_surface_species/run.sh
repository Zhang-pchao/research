#!/bin/bash
conda activate ase

python analyze_surface_species.py --format lammps --input model_atomic.data --traj ../bubble_1k.lammpstrj --atom_style "id type x y z"
