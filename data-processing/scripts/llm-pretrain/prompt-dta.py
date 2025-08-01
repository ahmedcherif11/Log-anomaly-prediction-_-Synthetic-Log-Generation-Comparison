import argparse
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main(args):
    data = load_jsonl(args.infile)
    train, test = train_test_split(data, test_size=0.15, random_state=42)
    train_dataset = Dataset.from_list(train)
    test_dataset = Dataset.from_list(test)
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
    dataset.save_to_disk(args.outfile)
    print("✅ Saved as HuggingFace Dataset with columns: ", train_dataset.column_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to HuggingFace Dataset")
    parser.add_argument("-in", "--infile", required=True, help="Input JSONL file")
    parser.add_argument("-out", "--outfile", required=True, help="Output directory for HuggingFace dataset")
    args = parser.parse_args()
    main(args)
