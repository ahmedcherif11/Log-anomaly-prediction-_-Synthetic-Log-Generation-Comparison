import os
import yaml
import re

def extract_info_from_yaml(file_path):
    mitre_techniques = []
    data_phrase_count = 0
    data_phrase_count_2 = 0  
    data_phrase_count_3 = 0  # For 'startswith' phrases


    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Count how many times the phrase appears
            data_phrase_count = content.count("contains")
            data_phrase_count_2 = content.count("endswith") 
            data_phrase_count_3 = content.count("startswith")

            # Try loading YAML to extract tags
            yaml_content = yaml.safe_load(content)
            tags = yaml_content.get('tags', []) if yaml_content else []

            for tag in tags:
                match = re.search(r'(t\d{4}(?:\.\d{3})?)', tag)
                if match:
                    mitre_techniques.append(match.group(1))

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

    return mitre_techniques, data_phrase_count, data_phrase_count_2, data_phrase_count_3


def analyze_yaml_folder(directory):
    unique_techniques = set()
    total_data_phrase_mentions = 0
    count_endd = 0
    count_startt = 0  # For 'startswith' phrases

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                file_path = os.path.join(root, file)
                techniques, phrase_count ,count_end ,count_start= extract_info_from_yaml(file_path)

                unique_techniques.update(techniques)
                total_data_phrase_mentions += phrase_count
                count_endd += count_end
                count_startt += count_start    
                

    return unique_techniques, total_data_phrase_mentions , count_endd, count_startt
def extract_fields_before_contains(file_path):
    fields = set()
    fields2 = set() 
    fields3 = set()  # For fields before '|endswith' and '|startswith'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            matches = re.findall(r'(\w+)\|contains', content)
            matches2 =  re.findall(r'(\w+)\|endswith', content)
            matches3 = re.findall(r'(\w+)\|startswith', content)
            fields.update(matches)
            fields2.update(matches2)
            fields3.update(matches3)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
    return fields , fields2 , fields3

def scan_folder_for_contains_fields(directory):
    all_fields = set()
    all_fields2 = set()  # For fields before '|endswith' 
    all_fields3 = set()  # For fields before '|startswith'
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                file_path = os.path.join(root, file)
                fields ,fields2 ,fields3= extract_fields_before_contains(file_path)
                all_fields.update(fields)
                all_fields2.update(fields2)
                all_fields3.update(fields3)
    return all_fields , all_fields2 , all_fields3
def extract_full_compositions(file_path):
    compositions = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Match patterns like field|func|modifier (greedy for multiple pipes)
            matches = re.findall(r'(\w+(?:\|\w+)+)', content)
            for match in matches:
                compositions[match] = compositions.get(match, 0) + 1
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
    return compositions

def scan_folder_for_full_compositions(directory):
    all_compositions = {}
    exclude_suffixes = (
        '|re' , '|endswith', '|startswith'
    )
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                file_path = os.path.join(root, file)
                comps = extract_full_compositions(file_path)
                for comp, count in comps.items():
                    if not any(comp.endswith(suffix) for suffix in exclude_suffixes):
                        all_compositions[comp] = all_compositions.get(comp, 0) + count
    return all_compositions



def find_first_file_with_two_data_contains_all(directory):
    """
    Finds all files where 'CommandLine|contains|all' appears 2 or more times in the same block.
    A 'block' is defined as a YAML dictionary (e.g., under the same key).
    Saves all such file paths to 'files_with_two_or_more_CommandLine_contains_all.txt'.
    """

    files_with_matches = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            doc = yaml.safe_load(f)
                        except Exception as e:
                            print(f"❌ YAML parse error in {file_path}: {e}")
                            continue
                        if not isinstance(doc, dict):
                            continue
                        # Recursively search for blocks with 2+ 'CommandLine|contains|all'
                        def count_in_block(block):
                            count = 0
                            if isinstance(block, dict):
                                for v in block.values():
                                    count += count_in_block(v)
                            elif isinstance(block, list):
                                for item in block:
                                    count += count_in_block(item)
                            elif isinstance(block, str):
                                if block == 'CommandLine|contains|all':
                                    count += 1
                            return count

                        def has_block_with_two_or_more(block):
                            if isinstance(block, dict):
                                for v in block.values():
                                    if has_block_with_two_or_more(v):
                                        return True
                                # Count in this block
                                block_count = 0
                                for v in block.values():
                                    block_count += count_in_block(v)
                                if block_count >= 2:
                                    return True
                            elif isinstance(block, list):
                                for item in block:
                                    if has_block_with_two_or_more(item):
                                        return True
                            return False

                        if has_block_with_two_or_more(doc):
                            files_with_matches.append(file_path)
                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")

    # Save results
    with open("files_with_two_or_more_CommandLine_contains_all.txt", "w", encoding="utf-8") as out_file:
        for path in files_with_matches:
            out_file.write(path + "\n")
    return files_with_matches
print("------------------------------------------------------------------")
files= find_first_file_with_two_data_contains_all(r"C:\Users\AHMED\Desktop\new-approch\sigma\rules\windows")  # Update path if needed

def find_condition_one_of_pattern(file_path):
    """
    Checks if the YAML file contains a condition in the pattern:
    '1 of ... and 1 of ...'
    Returns True if found, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Look for a line like: condition: 1 of ... and 1 of ...
            pattern = r'condition:\s* [^:]+and ([^\n]+or[^\n]+)'
            if re.search(pattern, content):
                return True
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
    return False
# --- USAGE ---
folder_path = r"C:\Users\AHMED\Desktop\new-approch\sigma\rules\windows"  # Update path if needed
techniques, phrase_count ,count_end,count_start= analyze_yaml_folder(folder_path)
fields_before_contains,fields_before_endwith, fields_before_startswith= scan_folder_for_contains_fields(folder_path)


# --- OUTPUT ---
# print(f"\n✅ Total unique MITRE techniques found: {len(techniques)}")
# for t in sorted(techniques):
#     print(f" - {t}")

# print(f"\n🔍 Total times 'contains:' was mentioned: {phrase_count}")
# print(f"\n🔍 Total times 'endswith:' was mentioned: {count_end}")
# print(f"\n🔍 Total times 'startswith:' was mentioned: {count_start}")
# print("------------------------------------------------------------------")
# print(f"\n🧠 Unique fields mentioned before '|contains': {len(fields_before_contains)}")
# for field in sorted(fields_before_contains):
#     print(f" - {field}")
# print("------------------------------------------------------------------")
# print(f"\n🧠 Unique fields mentioned before '|endswith': {len(fields_before_endwith)}")
# for field in sorted(fields_before_endwith):
#     print(f" - {field}")
# print("------------------------------------------------------------------")
# print(f"\n🧠 Unique fields mentioned before '|startswith': {len(fields_before_startswith)}")
# for field in sorted(fields_before_startswith):
#     print(f" - {field}")

# print("------------------------------------------------------------------")
# print(f"\n🧩 Full compositions found:")
# compositions = scan_folder_for_full_compositions(folder_path)
# for comp, count in sorted(compositions.items(), key=lambda x: x[1], reverse=True):
#     print(f" - {comp}: {count} times")

# print("------------------------------------------------------------------")
# print("\n🔍 Searching for first file with two 'CommandLine|contains|all:' mentions..." )
# first_file = find_first_file_with_two_data_contains_all(folder_path)
# if first_file:
#     print(f"✅ Found: {first_file}")    
# else:
#     print("❌ No file found  with two 'CommandLine|contains|all:' mentions.")
i = 0
print("------------------------------------------------------------------")
print("\n🔍 Searching for '1 of ... and 1 of ...' condition patterns..."    )
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.yaml') or file.endswith('.yml'):
            file_path = os.path.join(root, file)
            if find_condition_one_of_pattern(file_path):
                print(f"✅ Found condition pattern in: {file_path}")
                i= i + 1

