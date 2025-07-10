#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4       # Request 4 V100 GPUs
#SBATCH --mem=180G                   # More RAM for big batches/context
#SBATCH --cpus-per-task=32           # Use more CPUs for dataloader
#SBATCH --output=./data/%N-%j.out
#SBATCH --time=1-00:00:00            # 1 day max
#SBATCH --account=def-dmouheb      

# ---- Environment Setup ----

source ./statics/environment.sh "$HOME/training_env" offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- Variables ----

# Path to your processed JSONL dataset (pre-merged, cleaned, shuffled)
DATASET="$SCRATCH/dataset/data.jsonl"
# Path where you want to store results
OUTPUT_DIR="$SCRATCH/models/windowslog-pretrain"
# Model name (local cache or HuggingFace)
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
RUN_NAME="llama3-winevt-pretrain-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUTPUT_DIR/$RUN_NAME"

# ---- (Optional) Data Preparation Step (if needed) ----
# If you already have your .jsonl ready, skip this.
# If you need to shuffle, filter, or convert, do it here.
# Example: python scripts/shuffle_jsonl.py --input ... --output ...

# ---- Training ----

echo "Starting LLM pretraining on Windows Event Logs..."
cd "$OUTPUT_DIR" || exit

# If using Accelerate
time accelerate launch --num_processes=4 \
  --main_process_port=29500 \
  "$HOME/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data processing/scripts/llm-pretrain\
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --run-name "$RUN_NAME" \
  --batch 1 \
  --grad 8 \
  --context 1024 \
  --root "$OUTPUT_DIR" \
  --checkpoint False \
  --save_checkpoint False

# ---- Save and Archive ----

echo "Training complete. Archiving run..."
tar czf "$OUTPUT_DIR/${RUN_NAME}.tar.gz" -C "$OUTPUT_DIR/$RUN_NAME" .

echo "All done! Logs and models saved to: $OUTPUT_DIR/$RUN_NAME"
