import random



input_path ="/scratch/cherif/dataset/data_fixed_normalized.jsonl"
output_path = "/scratch/cherif/dataset/data_fixed_shuffled.jsonl"

with open(input_path, "r") as infile:
    lines = infile.readlines()

random.shuffle(lines)

with open(output_path, "w") as outfile:
    outfile.writelines(lines)

print("Shuffling complete!")
