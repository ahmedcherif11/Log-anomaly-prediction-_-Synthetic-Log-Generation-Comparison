#!/bin/bash

# -------------------[ SETUP SCRIPT: init.sh ]--------------------

# Usage:
#   sh ./init.sh <huggingface-model-id> <git-dataset-url>
#
# Example:
#   sh ./init.sh meta-llama/Meta-Llama-3.1-8B https://github.com/logpai/loghub.git

#############################
# 1. Load modules 
#############################

source ./statics/modules.sh
#############################
# 2. Setup virtualenv
#############################
ENV="$HOME/training_env"

echo "Creating virtualenv at $ENV"
virtualenv --no-download "$ENV"
source "$ENV/bin/activate"

#############################
# 3. Install Python packages (offline, from local wheels)
#############################

echo "Installing Python dependencies..."
pip install --no-index --upgrade pip
pip install torch transformers datasets bitsandbytes peft deepspeed \
            accelerate==1.2.1 trl==0.13.0 wandb==0.18.6 \
            packaging ninja pyyaml matplotlib seaborn pandas

#############################
# 4. Download HF Model (if provided)
#############################

if [ "$1" ]; then
  MODEL_ID="$1"
  MODEL_DIR="$SCRATCH/models/$MODEL_ID"
  mkdir -p "$MODEL_DIR"
  source /scratch/cherif/dataset/token.sh
  echo "Downloading HF model $MODEL_ID to $MODEL_DIR"
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR" --token "$HF_TOKEN"
fi

#############################
# 5. Clone Dataset Repo 
#############################

if [ "$2" ]; then
  GIT_DATA="$2"
  DATA_DIR="$SCRATCH/dataset/$(basename "$GIT_DATA" .git)"
  echo "Cloning dataset $GIT_DATA to $DATA_DIR"
  git clone "$GIT_DATA" "$DATA_DIR"
fi

#############################
# 6. Deactivate env (done)
#############################
deactivate

echo "Setup complete! Your environment is at $ENV"
echo "Model (if downloaded): $MODEL_DIR"
echo "Dataset (if cloned): $DATA_DIR"

# ---------------------------------------------------------------------
