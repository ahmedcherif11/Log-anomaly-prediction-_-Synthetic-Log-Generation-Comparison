import json

infile = "/scratch/cherif/dataset/data.jsonl"
outfile = "/scratch/cherif/dataset/data_fixed.jsonl"

with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            # Replace unquoted NaN (from pandas) with null before parsing
            fixed_line = line.replace(":NaN", ":null")
            obj = json.loads(fixed_line)
            # If you want to also fix quoted "NaN" as empty string:
            obj = {k: ("" if v == "NaN" else v) for k, v in obj.items()}
            fout.write(json.dumps(obj) + "\n")
        except Exception as e:
            print("Error parsing:", line)
            print(e)
