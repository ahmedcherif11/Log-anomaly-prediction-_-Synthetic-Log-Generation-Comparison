import json
from datasets import Dataset, DatasetDict

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Load your SFT pairs
data = load_jsonl("$SCRATCH/datasets/prompts.jsonl")  # Your file with {"prompt":..., "response":...}

# Optional: train/test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.15, random_state=42)

train_dataset = Dataset.from_list(train)
test_dataset = Dataset.from_list(test)

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
})

dataset.save_to_disk("$SCRATCH/datasets/dta_prompt" )
print("✅ Saved as HuggingFace Dataset with columns: ", train_dataset.column_names)
