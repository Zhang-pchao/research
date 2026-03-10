#!/bin/bash
source activate /home/pengchao/app/anaconda3/envs/dpmdkit_v2.2.10

dp train run.json --skip-neighbor-stat # --restart model.ckpt
dp freeze
dp compress #--training-script run4compress.json
