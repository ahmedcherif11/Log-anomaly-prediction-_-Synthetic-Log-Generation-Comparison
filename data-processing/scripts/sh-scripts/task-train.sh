#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4	           # Request 4 V100 GPUs
#SBATCH --mem=100000                  # More RAM for big batches/context
#SBATCH --cpus-per-task=16           # Use more CPUs for dataloader
#SBATCH --output=$SCRATCH/models/slurm-logs/%N-%j.out
#SBATCH --time=32:00:00
#SBATCH --account=def-dmouheb  
#SBATCH --mail-user=ahmed.cherif.1@ulaval.ca
#SBATCH --mail-type=ALL    


### launch exemple : sbatch task-train.sh llama-3.1-8B-log-generator-adapter accelerate

############### Setting up environments & variables ###############

source ./statics/environment.sh "$HOME/training_env" offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN

mkdir -p "$SCRATCH/models/run"

RUN=${1:-"llama-3.1-8B-fttlogs-adapter"}

if [ "$2" != "checkpoint" ] && [ "$2" != "save" ]; then
  BASE_RUN=$RUN
  base_folder="$SCRATCH/models/run/$RUN"
  count=1

  if [ -d "${base_folder}" ]; then
    while true; do
        if [ -d "${base_folder}${count}" ]; then
            count=$((count + 1))
        else
            break
        fi
    done
    RUN="${RUN}${count}"
  fi
  mkdir -p "$SCRATCH/models/run/$RUN"
else
  # Check if RUN ends with a number (count)
  if [[ "$RUN" =~ [0-9]+$ ]]; then
    # If it ends with a number (count), remove the number and leave the base name
    BASE_RUN="${RUN%[0-9]*}"
  else
    BASE_RUN="$RUN"
  fi
fi

PROCESSED="$SCRATCH/datasets/new-new-prompts"

echo "Base run : $BASE_RUN"
echo "Run name : $RUN"
echo "Run saved in $SCRATCH/models/run/$RUN"



cd "$SCRATCH/models/" || exit


echo "Starting training..."

############### Launching task-specific training scripts ###############

if [ "$2" == "accelerate" ]; then
  time accelerate launch --config_file="$PROJ/config/train_config.yaml" "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/train.py" --dataset "$PROCESSED" --model "/scratch/cherif/models/llama-raw-logs-model" --run-name "$RUN" --context 4096
elif [ "$2" == "save" ]; then
  time accelerate launch "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/train.py" --dataset "$PROCESSED" --model "/scratch/cherif/models/llama-raw-logs-model" --run-name "$RUN" --save_checkpoint
elif [ "$2" == "checkpoint" ]; then
  time accelerate launch --config_file="$PROJ/config/train_config.yaml" "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/train.py" --dataset "$PROCESSED" --model "/scratch/cherif/models/llama-raw-logs-model" --run-name "$RUN" --checkpoint --context 1024
else
  time python "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/train.py" --dataset "$PROCESSED" --model "/scratch/cherif/models/llama-raw-logs-model" --run-name "$RUN" --context 2048
fi

echo "End models run : $RUN"
tar czf "$SCRATCH/saves/$RUN/$RUN.tar.gz" -C "$SCRATCH/models/run/$RUN" .
echo "Training complete! Results saved to $SCRATCH/models/run/$RUN"
