parentuser_contains_list = [
    "AUTHORI",
    "AUTORI"
]
parentuser_endswith_list = [
    "\\NETWORK SERVICE",
    "\\LOCAL SERVICE"
]
user_contains_list = [
    "AUTHORI",
    "AUTORI"
]
user_endswith_list = [
    "\\SYSTEM",
    "\\Système",
    "\\СИСТЕМА"
]
integritylevel_list = [
    "System",
    "S-1-16-16384"
]

input_file = r"C:\Users\AHMED\Desktop\new-approch\example.jsonl"
output_file = "user_parentuser_integrity_aug.jsonl"

total_lines = 0

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.rstrip('\n\r')
        found_parent_contains = "AUTHORI" in line 
        found_parent_endswith = "\\LOCAL SERVICE" in line 

        found_user_endswith = "\\SYSTEM" in line

        found_integrity = "S-1-16-16384" in line

        # If none found, just write the line as is
        if not (found_parent_contains or found_parent_endswith or found_user_contains or found_user_endswith or found_integrity):
            fout.write(line + "\n")
            total_lines += 1
            continue

        for parent_c in parentuser_contains_list:
            for parent_e in parentuser_endswith_list:
                for user_c in user_contains_list:
                    for user_e in user_endswith_list:
                        for integrity in integritylevel_list:
                            new_line = line
                            if found_parent_contains:
                                for p in parentuser_contains_list:
                                    if p in new_line:
                                        new_line = new_line.replace(p, parent_c)
                            if found_parent_endswith:
                                for p in parentuser_endswith_list:
                                    if p in new_line:
                                        new_line = new_line.replace(p, parent_e)
                           
                            if found_user_endswith:
                                for u in user_endswith_list:
                                    if u in new_line:
                                        new_line = new_line.replace(u, user_e)
                            if found_integrity:
                                for i in integritylevel_list:
                                    if i in new_line:
                                        new_line = new_line.replace(i, integrity)
                            fout.write(new_line + "\n")
                            total_lines += 1

print(f"Processed {input_file} and saved all combinations to {output_file}.")
print(f"{total_lines} lines written to output file.")
print(f"Total combinations generated: {len(parentuser_contains_list) * len(parentuser_endswith_list) * len(user_contains_list) * len(user_endswith_list) * len(integritylevel_list)}")
print(len(output_file), "lines written to output file.")