import json
import math

infile = "/scratch/cherif/dataset/data_fixed_shuffled.jsonl"
outfile = "/scratch/cherif/dataset/data_fixed_shuffled_normalized.jsonl"

# All fields that might have int, float, or str and should be str (except 'null')
fields_to_str = [
    "ProcessId", "ProcessID", "UserId", "MapDescription", "UserName", "RemoteHost",
    "PayloadData1", "PayloadData2", "PayloadData3", "PayloadData4",
    "PayloadData5", "PayloadData6", "ExecutableInfo", "Keywords", "Task"
]

with open(infile, "r", encoding="utf-8") as f_in, open(outfile, "w", encoding="utf-8") as f_out:
    for idx, line in enumerate(f_in, 1):
        obj = json.loads(line)
        for k in fields_to_str:
            if k in obj and obj[k] is not None:
                # If it's a float nan, convert to empty string or null if you want
                if isinstance(obj[k], float) and math.isnan(obj[k]):
                    obj[k] = ""
                else:
                    obj[k] = str(obj[k])
        f_out.write(json.dumps(obj) + "\n")
        # Optionally print progress
        if idx % 100000 == 0:
            print(f"Processed {idx} lines...")

print("Normalization complete!")
