#!/bin/bash -l
#PBS -l select=500
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q lustre_scaling
#PBS -A Aurora_deployment
#PBS -k doe
#PBS -N python_launch

cd ${PBS_O_WORKDIR}

python ensemble_launcher.py
