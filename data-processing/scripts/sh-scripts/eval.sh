#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=$SCRATCH/models/slurm-logs/eval-%N-%j.out
#SBATCH --time=06:00:00
#SBATCH --account=def-dmouheb
#SBATCH --mail-user=ahmed.cherif.1@ulaval.ca
#SBATCH --mail-type=ALL

set -euo pipefail

module purge
# Load site-recommended CUDA/PyTorch (ask your cluster docs; examples:)
# module load cuda/12.1
# module load python/3.12  # if needed
# Or a site pytorch module:
# module load pytorch/2.3.1

# Activate your venv AFTER modules
source "$HOME/training_env/bin/activate"

# If your environment.sh only sets Python packages, OK.
# If it exports CUDA paths, consider skipping it or ensure it doesn't override CUDA libs.
# source ./statics/environment.sh "$HOME/training_env" offline

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

MODEL_RUN=llama-3.1-8B-log-generator-eval4
MODEL_DIR=$SCRATCH/models/llama-gen-logs-model
OUTPUT_DIR=$SCRATCH/eval/$MODEL_RUN
DATASET=$SCRATCH/datasets/shortest-prompts

mkdir -p "$OUTPUT_DIR"

python /project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/new-eval.py \
  --model_dir "$MODEL_DIR" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --max_new_tokens 1024
