from datasets import load_dataset

# Load a small sample of your dataset
dataset = load_dataset("json", data_files="/home/cherif/scratch/dataset/data_fixed_allfields_null.jsonl", split="train")

# Print the column names
print("Column names:", dataset.column_names)
# Get the first line to inspect
first_row = dataset[0]

# If you have a column 'text', check its type
if "text" in dataset.column_names:
    print("Type of first 'text' value:", type(first_row["text"]))
    print("Value:", first_row["text"])
else:
    print("First row type:", type(first_row))
    print("First row:", first_row)
