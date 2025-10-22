#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -q <queue>
#PBS -A <project>
#PBS -l filesystems=home:flare



if [ -n "${PBS_O_WORKDIR}" ]; then
    cd ${PBS_O_WORKDIR}
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source /path/to/env

dirname=$(pwd)
echo working_dir: $dirname

python3 start_mcp_http.py