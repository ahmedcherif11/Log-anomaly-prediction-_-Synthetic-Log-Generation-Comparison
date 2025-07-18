input_file = r'C:\Users\AHMED\Desktop\new-approch\dataset\data-for-train-llm\data.jsonl'      # Replace with your original file name
output_file = 'output_1000.jsonl'  # The new file

import random
num_lines = 100

# Read all lines
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Randomly select 1000 lines (without replacement)
sampled_lines = random.sample(lines, min(num_lines, len(lines)))

# Write selected lines to output
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(sampled_lines)