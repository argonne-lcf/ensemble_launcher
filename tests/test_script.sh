#!/bin/bash

# Iterate over command-line arguments
args=($@)

even_numbers=()

for ((i=0; i<${#args[@]}; i++)); do
    if [[ $((i % 2)) -eq 1 ]]; then
        even_numbers+=("${args[i]}")
    fi
done

if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_RANKID=$MPI_LOCALRANKID
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_RANKID=$PALS_LOCAL_RANKID
fi

mpi_rank=${_MPI_RANKID:-"not set"}
if command -v taskset &>/dev/null; then
  cpu_affinity=$(taskset -p $$)
else
  cpu_affinity="taskset not available"
fi
gpu_affinity=${ZE_AFFINITY_MASK:-"not set"}
echo "MPI RANK: ${mpi_rank}, GPU Affinity: ${gpu_affinity}"

echo "Even numbers from arguments: ${even_numbers[*]}"
echo "started sleep"
sleep 10
