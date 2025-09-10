#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=$SCRATCH/models/slurm-logs/gen-%N-%j.out
#SBATCH --time=02:00:00
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

MODEL_RUN=llama-3.1-8B-TWO-log-generator-eval3
MODEL_DIR=$SCRATCH/models/llama-gen-logs-model
DATASET=$SCRATCH/datasets/shortest-prompts     # HuggingFace disk format
OUTPUT_JSONL=$SCRATCH/eval/$MODEL_RUN/generated.jsonl

mkdir -p $(dirname $OUTPUT_JSONL)

python /project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/gen.py \
    --model_dir $MODEL_DIR \
    --dataset $DATASET \
    --output_jsonl $OUTPUT_JSONL \
    --max_new_tokens 384