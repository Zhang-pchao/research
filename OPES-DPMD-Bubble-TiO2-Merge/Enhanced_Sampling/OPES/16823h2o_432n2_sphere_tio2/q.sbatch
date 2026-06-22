#!/bin/bash
#SBATCH --job-name=pure_o_100
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1         
#SBATCH --gpus-per-node=1   
#SBATCH --qos=improper-gpu   

source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash

export OMP_NUM_THREADS=1

# Below are executing commands
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
module load lammps/29aug2024-dpv3
mpirun -np $SLURM_NTASKS -map-by $MAP_OPT lmp -sf omp -pk omp $OMP_NUM_THREADS -in in.lammps

# Must explicitly exit
exit
