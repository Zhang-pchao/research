#!/bin/bash

for i in $(seq 1 3); do
    cp pack.inp.org pack.inp
    sed -i "s/H2O_1/H2O_$i/g" pack.inp
    packmol < pack.inp > pack.out
done
