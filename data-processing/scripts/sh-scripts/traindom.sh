#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4	           # Request 4 V100 GPUs
#SBATCH --mem=80000                  # More RAM for big batches/context
#SBATCH --cpus-per-task=16           # Use more CPUs for dataloader
#SBATCH --output=$SCRATCH/models/slurm-logs/%N-%j.out
#SBATCH --time=01:30:00            # 1 day max
#SBATCH --account=def-dmouheb  
#SBATCH --mail-user=ahmed.cherif.1@ulaval.ca
#SBATCH --mail-type=ALL    

### launch exemple : sbatch ftd_train.sh llama-3.1-8B-ftdlogs-adapter accelerate

############### Setting up environments & variables ###############

source ./statics/environment.sh "$HOME/training_env" offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29505
export MASTER_ADDR=localhost
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "MASTER_PORT=$MASTER_PORT"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

############### Launching data preparation script ###############

#echo "Preparing data for ftd..."
#time python "$PROJ/py_scripts/ftd_prep.py" --dataset "$SCRATCH/datasets/loghub" --out "$SLURM_TMPDIR/datasets/ftd"


echo "Starting training..."
mkdir -p "$SCRATCH/models/run" && cd "$SCRATCH/models/" || exit

RUN=${1:-"llama-3.1-8B-ftdlogs-adapter"}

base_folder="run/$RUN"
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

echo "Run name : $RUN"
echo "Run saved in $SCRATCH/models/run/$RUN"

mkdir -p "$SCRATCH/models/run/$RUN"
############### Launching domain-specific training script ###############

time accelerate launch --config_file="$PROJ/config/train_config.yaml" "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/train.py" --dataset "$SCRATCH/dataset/dta_train" --model "meta-llama/Meta-Llama-3.1-8B" --run-name "$RUN" 



tar czf "$SCRATCH/save/$RUN.tar.gz" -C "$SCRATCH/models/run/$RUN" .