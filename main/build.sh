#!/bin/bash

# Source the main bashrc file to initialize the environment
source /u/sw/etc/bash.bashrc

# Check if the source was successful
if [ $? -ne 0 ]; then
    echo "Failed to source /u/sw/etc/bash.bashrc"
    exit 1
fi

# Load the required modules
module load gcc-glibc/11.2.0
module load eigen
module load tbb
module load lis
make
./filter einstein.jpg
mpirun -n 4 ./test1 A2.mtx w.mtx x.mtx hist.txt -tol 1.0e-9 -i bicgstab -p none
./convertX x.mtx