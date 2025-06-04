#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 500:00:00
#SBATCH --gpus 3090:1
#SBATCH --job-name=4602z

cd $SLURM_SUBMIT_DIR

module load conda
conda activate dpmdkit_v2.2.6_vcvs

# Run the application - the line below is just a random example.
lmp -i in.glycine > glycine.out
