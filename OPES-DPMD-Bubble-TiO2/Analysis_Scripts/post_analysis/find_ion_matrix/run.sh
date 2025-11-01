#!/bin/bash
conda activate ase

python transition_matrix_analysis.py \
    --base_path /home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4 \
    --output_dir /home/pengchao/bubble_ion/TiO/dpmd/102n2_7401h2o_tio2_water_layer/4/analysis_ion/find_ion_matrix/results

