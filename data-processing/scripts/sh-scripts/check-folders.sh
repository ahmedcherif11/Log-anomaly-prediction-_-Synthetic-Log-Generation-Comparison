# List of important folders to check
folders_to_check=(
    "$SCRATCH/dataset"
    "$SCRATCH/models/windowslog-pretrain"
    "$HOME/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts"
    "./statics"
    "./data"
)

echo "Checking required folders..."
for folder in "${folders_to_check[@]}"; do
    if [ ! -d "$folder" ]; then
        echo "WARNING: Directory does not exist: $folder"
        # Optionally, create it:
        # mkdir -p "$folder"
    else
        echo "Exists: $folder"
    fi
done
echo "Folder check complete."
