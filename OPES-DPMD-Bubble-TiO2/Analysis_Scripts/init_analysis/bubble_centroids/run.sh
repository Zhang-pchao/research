#!/bin/bash
conda activate ase

python calculate_bubble_centroids.py \
    --data model_atomic.data \
    --traj_file ../bubble_1k.lammpstrj \
    --step_interval 1 \
    --start_frame 1 \
    --end_frame -1 \
