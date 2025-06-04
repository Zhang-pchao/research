#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 500:00:00
#SBATCH --gpus 3090:1
#SBATCH --job-name=041trn

cd $SLURM_SUBMIT_DIR

module load conda
conda activate dpmdkit_v2.2.6_vcvs

# run
if [ ! -f tag_0_finished ] ;then
    { if [ ! -f model.ckpt.index ]; then
        CUDA_VISIBLE_DEVICES=0 horovodrun -np 1
        dp train --mpi-log=master run.json;
    else
        CUDA_VISIBLE_DEVICES=0 horovodrun -np 1
        dp train --mpi-log=master run.json --restart model.ckpt; fi }
    1>> train.log 2>> train.log
    { if test $? -ne 0; then touch tag_0_failure; 
	else
	  dp freeze && touch tag_0_finished; fi }
fi

#dp train --init-frz-model frozen_model_old.pb run.json
#dp freeze && dp compress
