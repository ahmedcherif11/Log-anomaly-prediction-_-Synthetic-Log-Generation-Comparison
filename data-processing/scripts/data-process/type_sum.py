import re
from datasets import Features, Value, Sequence

# Paste your type stats file path here
stats_file = r'C:\Users\AHMED\Desktop\new-approch\typesummury.txt'

features = {}

with open(stats_file, 'r') as f:
    content = f.read()

blocks = content.strip().split('--------------------')
for block in blocks:
    if not block.strip():
        continue
    lines = block.strip().split('\n')
    field = lines[0].split(':', 1)[1].strip()
    types = [l.split(':')[0].strip() for l in lines[1:]]
    if 'int' in types and 'str' not in types and 'list' not in types:
        features[field] = Value("int64")
    elif 'str' in types and 'int' not in types and 'list' not in types:
        features[field] = Value("string")
    elif 'list' in types:
        features[field] = Sequence(Value("string"))  # Or Value("int64") if you know
    elif 'str' in types and 'int' in types:
        features[field] = Value("string")  # Safer to keep as string if mixed
    else:
        features[field] = Value("string")  # Fallback

print("Features dictionary for HuggingFace datasets:\n")
print(features)
