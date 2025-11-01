#!/bin/bash
conda activate ase

python analyze_all_ion_species_matrix.py \
    --atom_style "id type x y z" \
    --input model_atomic.data \
    --traj ../bubble_1k.lammpstrj \
    --step_interval 1 \
    --start_frame 1 \
    --end_frame -1 \
