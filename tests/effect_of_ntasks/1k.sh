#!/bin/bash
#PBS -l select=32
#PBS -l walltime=00:05:00
#PBS -q prod
#PBS -l filesystems=home:eagle:grand
#PBS -A datascience
#PBS -o logs/1k_log.txt
#PBS -e logs/1k_err.txt


module use /soft/modulefiles
module load conda
conda activate 
module list

cd ${PBS_O_WORKDIR}
python3 test_ensemble_launcher.py -i "1k"
