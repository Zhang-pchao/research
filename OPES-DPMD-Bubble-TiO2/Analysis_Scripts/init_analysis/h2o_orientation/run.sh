#!/bin/bash
conda activate ase

python analyze_h2o_orientation.py \
    --atom_style "id type x y z" \
    --input model_atomic.data \
    --traj ../bubble_1k.lammpstrj \
    --step_interval 1 \
    --start_frame 1 \
    --end_frame -1 \
    --oh_cutoff 1.4 \
    --ti_o_cutoff 3.5 \
    --z_bin_size 2.0 \
    --has_tio2 \
    --enable_log_file \
    --output_dir h2o_orientation_results


