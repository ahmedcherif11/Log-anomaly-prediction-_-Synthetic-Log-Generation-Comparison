#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=$SCRATCH/models/slurm-logs/eval-%N-%j.out
#SBATCH --time=06:00:00
#SBATCH --account=def-dmouheb
#SBATCH --mail-user=ahmed.cherif.1@ulaval.ca
#SBATCH --mail-type=ALL

source ./statics/environment.sh "$HOME/training_env" offline
export CUDA_VISIBLE_DEVICES=0

# --- CHOOSE YOUR INPUTS ---
MODEL_RUN=llama-3.1-8B-log-generator   # <-- or final-model after merging
MODEL_DIR=$SCRATCH/models/llama-gen-logs-model    # <-- For merged, maybe $SCRATCH/models/final-model
OUTPUT_DIR=$SCRATCH/eval/$MODEL_RUN

# For test set with references:
DATASET=$SCRATCH/datasets/shortest-prompts/test     # HuggingFace disk format with test split

# For generation-only on raw prompts (jsonl file):
# DATASET=$SCRATCH/datasets/synthetic_prompts.jsonl

mkdir -p $OUTPUT_DIR

python /project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/eval.py \
    --model_dir $MODEL_DIR \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens 1024 \
