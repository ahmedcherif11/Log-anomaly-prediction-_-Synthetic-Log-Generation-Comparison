
#!/bin/bash

############### Setting up environments & variables ###############

source ./statics/environment.sh "$HOME/training_env"

if [ $# -eq 0 ]; then
    echo "No parameters provided. Please specify 'wandb' and/or 'huggingface' to sync."
    exit 1
fi

if [ -z "$MODEL" ]; then
  MODEL="hubaval/llama-3.1-8B-fttlogs"
  echo "Model not specified, using default: $MODEL"
fi

if [ -z "$MODELPATH" ]; then
  MODELPATH="$SCRATCH/models/run/llama-3.1-8B-ftdwindows4K-adapter2"
  echo "Model path not specified, using default: $MODELPATH"
fi

echo "Model used: $MODEL, path to the model: $MODELPATH"

param=$1

############### Syncing ###############

for param in "$@"; do
    if [ "$param" == "wandb" ]; then
        echo "Syncing with Weights & Biases (wandb)..."
        export WANDB_LOG_MODEL="checkpoint"
        WANDB_API_KEY=$(python -c "import py_scripts.credentials as cr; print(cr.wandb_key)")
        export WANDB_API_KEY
        wandb sync "$MODELPATH/wandb/latest-run"

    elif [ "$param" == "huggingface" ]; then
        echo "Syncing with Hugging Face..."
        HF_TOKEN=$(python -c "import py_scripts.credentials as cr; print(cr.hf_bis)")
        export HF_TOKEN
        huggingface-cli upload "$MODEL" "$MODELPATH"
        #huggingface-cli upload my-cool-dataset ./data . --repo-type dataset daatset

    else
        echo "Unknown parameter '$param'. Please use 'wandb' and/or 'huggingface'."
    fi
done
