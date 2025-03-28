
"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use the same GPU
"""
def gen_affinity_bash_script_aurora_1(ngpus_per_process) -> str:
    bash_script = [
                      "#!/bin/bash",
                      "##get the free gpus from the environment variable",
                      r'my_free_gpus=$(IFS=","; echo ${AVAILABLE_GPUS})',
                      "# Get the RankID from different launcher",
                      "if [[ -v MPI_LOCALRANKID ]]; then",
                      "   _MPI_RANKID=$MPI_LOCALRANKID ",
                      "elif [[ -v PALS_LOCAL_RANKID ]]; then",
                      "   _MPI_RANKID=$PALS_LOCAL_RANKID",
                      "fi",
                      "unset EnableWalkerPartition",
                      "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1",
                      "# Calculate the GPUs assigned to this rank",
                      f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                      f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                      r"export ZE_AFFINITY_MASK=${rank_gpus}",
                      r"ulimit -c 0 # Until Aurora filesystem problems are fixed",
                      '"$@"'
                 ]
    return "\n".join(bash_script)


"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use different GPUs
"""
def gen_affinity_bash_script_aurora_2(ngpus_per_process) -> str:
   """
   the below bash script is adapted from gpu_tile_compact.sh script from aurora
   """
   bash_script = [
                    "#!/bin/bash",
                    "##get the hostname",
                    "hname=$(hostname)",
                    "##get the free gpus from the environment variable",
                    r'my_free_gpus=$(IFS=","; echo ${AVAILABLE_GPUS_${hname}})',
                    "# Get the RankID from different launcher",
                    "if [[ -v MPI_LOCALRANKID ]]; then",
                    "   _MPI_RANKID=$MPI_LOCALRANKID ",
                    "elif [[ -v PALS_LOCAL_RANKID ]]; then",
                    "   _MPI_RANKID=$PALS_LOCAL_RANKID",
                    "fi",
                    "unset EnableWalkerPartition",
                    "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1",
                    "# Calculate the GPUs assigned to this rank",
                    f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                    f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                    r"export ZE_AFFINITY_MASK=${rank_gpus}",
                    r"ulimit -c 0 # Until Aurora filesystem problems are fixed",
                    '"$@"'
                ]
   return "\n".join(bash_script)