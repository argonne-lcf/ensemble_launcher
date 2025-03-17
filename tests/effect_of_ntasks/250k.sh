#!/bin/bash
#PBS -l select=32
#PBS -l walltime=01:30:00
#PBS -q prod
#PBS -l filesystems=home:eagle:grand
#PBS -A datascience
#PBS -o logs/250k_log.txt
#PBS -e logs/250k_err.txt

module use /soft/modulefiles
module load conda
conda activate 
module list

cd ${PBS_O_WORKDIR}

rm logs/250k*

python3 test_ensemble_launcher.py -i "250k"
