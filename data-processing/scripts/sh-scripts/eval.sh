#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=$SCRATCH/models/slurm-logs/eval-%N-%j.out
#SBATCH --time=04:30:00
#SBATCH --exclude=cdr2656,cdr2658
#SBATCH --account=def-dmouheb
#SBATCH --mail-user=ahmed.cherif.1@ulaval.ca
#SBATCH --mail-type=ALL
module purge
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
source ./statics/environment.sh "$HOME/training_env" offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# 1) What GPUs does Slurm give me?
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# Should be something like "0" or "0,1", not empty.

# 2) Does the node have visible GPUs and a working driver?
date
nvidia-smi || true

# 3) Does my PyTorch build have CUDA and can it init it?
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))
# Fail fast if CUDA can't initialize
raise SystemExit(0 if torch.cuda.is_available() else 42)
PY
if [ $? -ne 0 ]; then
  echo "CUDA failed to initialize on $(hostname). Requeuing this job…"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
# --- CHOOSE YOUR INPUTS ---
MODEL_RUN=llama-3.1-8B-two-eval # <-- or final-model after merging
MODEL_DIR=$SCRATCH/models/llama-gen-logs-model   # <-- For merged, maybe $SCRATCH/models/final-model
OUTPUT_DIR=$SCRATCH/eval/$MODEL_RUN

# For test set with references: 
DATASET=$SCRATCH/datasets/shortest-prompts     # HuggingFace disk format with test split

# For generation-only on raw prompts (jsonl file):
# DATASET=$SCRATCH/datasets/synthetic_prompts.jsonl

mkdir -p $OUTPUT_DIR

python /project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/eval-two-logs.py \
    --model_dir $MODEL_DIR \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens 512 \
