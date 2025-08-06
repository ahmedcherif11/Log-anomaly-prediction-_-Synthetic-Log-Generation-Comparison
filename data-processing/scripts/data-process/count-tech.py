import json
import re
from collections import Counter

technique_counter = Counter()
no_match_count = 0
no_match_lines = []
input_path = r"C:\Users\AHMED\Desktop\new-approch\dataset\prompts-copy.jsonl"

with open(input_path, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            prompt_obj = json.loads(line)
            text = prompt_obj.get("prompt", "")
        except json.JSONDecodeError:
            text = line
        match = re.search(r'ID:\s*(t\d{4}(?:\.\d{3})?)', text, re.IGNORECASE)
        if match:
            technique_counter[match.group(1).lower()] += 1
        else:
            no_match_count += 1
            if len(no_match_lines) < 5:
                no_match_lines.append(text[:200])  # Show first 200 chars

# ...existing code...

print(f"Number of unique techniques: {len(technique_counter)}")
print("Techniques found and their counts:")
for tech, count in technique_counter.most_common():
    print(f"{tech}: {count}")

print(f"\nLines with no technique match: {no_match_count}")
print(f"Total lines processed: {i}")
print(f"Sum of all technique counts: {sum(technique_counter.values())}")
print(f"Check: {sum(technique_counter.values()) + no_match_count} == {i}")

if no_match_lines:
    print("\nSample lines with no technique match:")
    for l in no_match_lines:
        print("---")
        print(l)