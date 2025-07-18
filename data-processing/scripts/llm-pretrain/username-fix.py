import json

infile = "/scratch/cherif/dataset/data.jsonl"
outfile = "/scratch/cherif/dataset/data_fixed.jsonl"

# with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
#     for line in fin:
#         try:
#             # Replace unquoted NaN (from pandas) with null before parsing
#             fixed_line = line.replace(":NaN", ":null")
#             obj = json.loads(fixed_line)
#             # If you want to also fix quoted "NaN" as empty string:
#             obj = {k: ("" if v == "NaN" else v) for k, v in obj.items()}
#             fout.write(json.dumps(obj) + "\n")
#         except Exception as e:
#             print("Error parsing:", line)
#             print(e)


with open(infile, "r") as fin, open(outfile, "w") as fout:
    for line in fin:
        obj = json.loads(line)
        kw = obj.get("Keywords", None)
        # Convert integer Keywords to string, or null if it's this value
        if isinstance(kw, int):
            # Option 1: convert ALL int keywords to string
            obj["Keywords"] = str(kw)
            # Option 2: or set just the weird value to null
            # if kw == -9223372036854775808:
            #     obj["Keywords"] = None
            # else:
            #     obj["Keywords"] = str(kw)
        fout.write(json.dumps(obj) + "\n")
