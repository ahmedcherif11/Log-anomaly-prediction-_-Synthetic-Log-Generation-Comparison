#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4	           # Request 4 V100 GPUs
#SBATCH --mem=80000                  # More RAM for big batches/context
#SBATCH --cpus-per-task=16           # Use more CPUs for dataloader
#SBATCH --output=$SCRATCH/models/slurm-logs/%N-%j.out
#SBATCH --time=02:30:00            # 1 day max
#SBATCH --account=def-dmouheb  
#SBATCH --mail-user=ahmed.cherif.1@ulaval.ca
#SBATCH --mail-type=ALL    

### launch exemple : sbatch merge.sh hubaval/llama-3.1-8B-fttlogs-adapter "hubaval/llama-3.1-8B-fttlogs"

############### Setting up environments & variables ###############

source ./statics/environment.sh "$HOME/training_env" offline
export CUDA_VISIBLE_DEVICES=0

MODEL=${1:-"/home/cherif/scratch/models/run/my-llama3.1-finetune/model"}
mkdir -p "$SCRATCH/models/final-model"

OUT=${2:-"/home/cherif/scratch/models/final-model"}

cd "$SCRATCH/models/" || exit

############### Launch merging script ###############

time srun python "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/merge.py" --model "$MODEL" --out "$OUT"