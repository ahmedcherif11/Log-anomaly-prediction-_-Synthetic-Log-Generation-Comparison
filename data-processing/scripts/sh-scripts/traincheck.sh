#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4	           # Request 4 V100 GPUs
#SBATCH --mem=80000                  # More RAM for big batches/context
#SBATCH --cpus-per-task=16           # Use more CPUs for dataloader
#SBATCH --output=$SCRATCH/models/slurm-logs/%N-%j.out
#SBATCH --time=04:30:00            # 1 day max
#SBATCH --account=def-dmouheb  
#SBATCH --mail-user=ahmed.cherif.1@ulaval.ca
#SBATCH --mail-type=ALL    



### launch exemple : sbatch ftd_train.sh llama-3.1-8B-ftdlogs-adapter accelerate

############### Setting up environments & variables ###############

source ./statics/environment.sh "$HOME/training_env" offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

############### Launching data preparation script ###############

echo "Preparing data for ftd..."
#time python "$PROJ/py_scripts/generate_ds.py" --dataset "$SCRATCH/datasets/WINDOWSLOG.jsonl" --out "$SLURM_TMPDIR/datasets/ftd"

#mkdir -p "$SCRATCH/datasets/ftd"
#tar xzf "$SCRATCH/datasets/WINDOWSLOG.tar.gz" -C "$SCRATCH/datasets/ftd"
#echo "Data $SCRATCH/datasets/WINDOWSLOG.tar.gz extracted to $SCRATCH/datasets/ftd"

echo "Starting training..."
mkdir -p "$SCRATCH/models/run" && cd "$SCRATCH/models/" || exit

RUN=${1:-"llama-3.1-8B-ftdlogs-adapter"}

echo "Run name : $RUN"
echo "Run saved in $SCRATCH/models/run/$RUN"

mkdir -p "$SCRATCH/models/run/$RUN"

############### Launching domain-specific training script ###############

if [ "$2" == "accelerate" ]; then
    time accelerate launch --config_file="$PROJ/config/train_config.yaml" "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/train.py"  --dataset "$SCRATCH/datasets/ftd" --model "meta-llama/Meta-Llama-3.1-8B" --run-name "$RUN" --context 4096 --checkpoint
else
    time python "/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/textdata/train.py"  --dataset "$SCRATCH/datasets/ftd" --model "meta-llama/Meta-Llama-3.1-8B" --run-name "$RUN"
fi
mkdir -p "$SCRATCH/saves"
mkdir -p "$SCRATCH/saves/$RUN"

tar czf "$SCRATCH/saves/$RUN/$RUN.tar.gz" -C "$SCRATCH/models/run/$RUN" .

echo "Training complete! Results saved to $SCRATCH/models/run/$RUN"