#!/bin/bash

PROJ=$PWD
export PYTHONPATH=$PROJ:$PYTHONPATH

source "$PROJ/statics/modules.sh"
source "$1/bin/activate"

if [ "$2" == "offline" ]; then
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script is not running under Slurm."
    export TMPDIR="$SCRATCH/TMPDIR"
else
    echo "This script is running under Slurm."
    export TMPDIR="$SLURM_TMPDIR"
fi