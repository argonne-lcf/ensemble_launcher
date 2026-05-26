#!/bin/bash
# Usage: ./start_vllm_server.sh HOST PORT NGPUS [VLLM_MODEL] [CACHE_DIR] [TEMPDIR]
#
# Required Arguments:
#   PORT         - vLLM server port
#   NGPUS        - Number of GPUs to use
#
# Optional Arguments:
#   VLLM_MODEL   - Model name (default: meta-llama/Llama-3.1-8B-Instruct)
#   CACHE_DIR    - HuggingFace cache directory (default: $(pwd)/.cache)
#   TEMPDIR      - Temporary directory (default: /tmp)
#
# Examples:
#   ./start_vllm_server.sh $(hostname) 8000 8
#   ./start_vllm_server.sh $(hostname) 8001 4 meta-llama/Llama-3.1-70B-Instruct

# Check required arguments
if [ $# -lt 3 ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 HOST PORT NGPUS [VLLM_MODEL] [CACHE_DIR] [TEMPDIR]"
    echo ""
    echo "Required arguments:"
    echo "  HOST         - vLLM server host"
    echo "  PORT         - vLLM server port"
    echo "  NGPUS        - Number of GPUs to use"
    echo ""
    echo "Example: $0 8000 8"
    exit 1
fi

# Command line arguments
HOST=${1}
PORT=${2}
NGPUS=${3}
VLLM_MODEL=${4:-"meta-llama/Llama-3.1-8B-Instruct"}
CACHE_DIR=${5:-"$(pwd)/.cache"}
TEMPDIR=${6:-"/tmp"}

echo "PORT: $PORT"
echo "NGPUS: $NGPUS"
echo "VLLM_MODEL: $VLLM_MODEL"
echo "CACHE_DIR: $CACHE_DIR"
echo "TEMPDIR: $TEMPDIR"

# Create cache directory if needed
if [ ! -d "${CACHE_DIR}" ]; then
    mkdir -p "${CACHE_DIR}"
fi

# HuggingFace environment variables
export HF_HOME=${CACHE_DIR}
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN=${HUGGINGFACE_HUB_TOKEN}
export HF_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Temporary directory setup (hash based on model name)
MODEL_HASH=$(echo "${VLLM_MODEL}" | md5sum | cut -d' ' -f1 | cut -c1-8)
export TMPDIR="${TEMPDIR}/tmp-${MODEL_HASH}"

if [ ! -d "${TMPDIR}" ]; then
    mkdir -p "${TMPDIR}"
fi

# Intel OneAPI and CCL settings
unset CCL_PROCESS_LAUNCHER
export CCL_PROCESS_LAUNCHER=None
unset ONEAPI_DEVICE_SELECTOR
export OCL_ICD_FILENAMES="libintelocl.so"
export FI_MR_CACHE_MONITOR=userfaultfd

# vLLM specific settings
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TOKENIZERS_PARALLELISM=false

# Build ZE_AFFINITY_MASK from NGPUS if not already set
if [ -z "$ZE_AFFINITY_MASK" ]; then
    export ZE_AFFINITY_MASK=$(seq -s, 0 $(($NGPUS - 1)))
fi

echo "ZE_AFFINITY_MASK: $ZE_AFFINITY_MASK"

# Set up log directory
LOG_DIR="$(pwd)/logs/vllm_server"
mkdir -p "$LOG_DIR"

# Launch vLLM server
if [ $NGPUS -eq 1 ]; then
    echo "$(date) Starting vllm with 1 GPU on port $PORT"
    vllm serve ${VLLM_MODEL} --host ${HOST} --port $PORT --trust-remote-code  --enable-auto-tool-choice --tool-call-parser hermes \
        1> "$LOG_DIR/vllm.server.log" \
        2> "$LOG_DIR/vllm.server.err"
else
    echo "$(date) Starting vllm with ${NGPUS} GPUs on port $PORT"
    vllm serve ${VLLM_MODEL} --distributed-executor-backend mp --host ${HOST} --port $PORT  --enable-auto-tool-choice --tool-call-parser hermes \
        --tensor-parallel-size ${NGPUS} --trust-remote-code \
        1> "$LOG_DIR/vllm.server.log" \
        2> "$LOG_DIR/vllm.server.err"
fi
