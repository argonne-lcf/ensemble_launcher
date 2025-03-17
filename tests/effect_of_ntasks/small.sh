#!/bin/bash
#PBS -l select=2
#PBS -l walltime=00:05:00
#PBS -q debug
#PBS -l filesystems=home:eagle:grand
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/

module use /soft/modulefiles
module load conda
conda activate 
module list

cd ${PBS_O_WORKDIR}

python3 test_ensemble_launcher.py -i "small"
