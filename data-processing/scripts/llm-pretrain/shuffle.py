import random

input_path = "data_fixed.jsonl"
output_path = "data_fixed_shuffled.jsonl"

with open(input_path, "r") as infile:
    lines = infile.readlines()

random.shuffle(lines)

with open(output_path, "w") as outfile:
    outfile.writelines(lines)

print("Shuffling complete!")
