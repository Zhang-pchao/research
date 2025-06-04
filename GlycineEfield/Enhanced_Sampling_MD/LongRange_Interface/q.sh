#!/bin/bash -l

module purge
source activate deepmd-kit_v2.2.9_vcv

# Run the application - the line below is just a random example.
lmp -i in.lmp > lmp.out
