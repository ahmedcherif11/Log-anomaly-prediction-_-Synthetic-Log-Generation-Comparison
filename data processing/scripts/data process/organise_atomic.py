import os
import zipfile
import yaml
import shutil

# Paths
root_dir = r"C:\Users\AHMED\Desktop\new-approch\Security-Datasets\datasets\atomic"
output_root = r"C:\Users\AHMED\Desktop\new-approch\Security-Datasets\datasets\atomic\organized_datasets"
root_dir_yaml = r"C:\Users\AHMED\Desktop\new-approch\Security-Datasets\datasets\atomic\_metadata"

import os
import zipfile

def unzip_all_files(directory):
    """Unzip all .zip files in the given directory and subdirectories,
    but skip if the destination folder already exists and is not empty."""
    
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith(".zip"):
                zip_path = os.path.join(dirpath, file)

                # Define output folder (same as zip name without .zip)
                extract_to = zip_path.replace(".zip", "")

                # ✅ Skip if already extracted
                if os.path.exists(extract_to) and os.listdir(extract_to):
                    print(f"[SKIP] Already extracted: {zip_path}")
                    continue

                # Try to unzip
                try:
                    os.makedirs(extract_to, exist_ok=True)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                    print(f"[OK] Unzipped: {zip_path}")
                except zipfile.BadZipFile:
                    print(f"[ERROR] Bad zip file: {zip_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to unzip {zip_path}: {e}")


def load_yaml_from_dir(directory):
    """Find and load YAML files."""
    yamls = []
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith(".yaml") or file.endswith(".yml"):
                with open(os.path.join(dirpath, file), 'r', encoding='utf-8') as f:
                    try:
                        data = yaml.safe_load(f)
                        yamls.append((data, os.path.join(dirpath, file)))
                    except yaml.YAMLError as e:
                        print(f"Error loading {file}: {e}")
      
    return yamls

def move_windows_technique_files(yaml_list):
    """Move files based on Windows attack mappings."""
    # i and j are counters for files processed and skipped
    i=0
    j=0
    for data, yaml_path in yaml_list:
        if "attack_mappings" not in data or "files" not in data:
            j=j+1
            continue
        for mapping in data["attack_mappings"]:
            # Only process if platform is Windows
         
            if "Windows" not in data["platform"]:
                
                i=i+1
                continue

            technique = mapping.get("technique")
            subtech = mapping.get("sub-technique")
            tactics = mapping.get("tactics", [])
            
            for tactic in tactics:
                tactic_folder = os.path.join(output_root, tactic)
                technique_folder = os.path.join(tactic_folder, technique)

                final_folder = technique_folder
                if subtech:
                    final_folder = os.path.join(technique_folder, f"{technique}_{subtech}")

                os.makedirs(final_folder, exist_ok=True)

                for file_entry in data.get("files", []):
                    zip_path = file_entry.get("link")
                    if not zip_path or not zip_path.endswith(".zip"):
                        continue

                    extracted_folder = zip_path.replace(".zip", "")
                    if os.path.exists(extracted_folder):
                        dest = os.path.join(final_folder, os.path.basename(extracted_folder))
                        if not os.path.exists(dest):
                            shutil.move(extracted_folder, dest)
                            print(f"Moved {extracted_folder} → {dest}")
    return i, j
def count_unmoved_folders(root_dir, organized_root):
    """
    Count and print the number of extracted folders that were not moved
    from each parent directory in root_dir.
    """
    unmoved_counts = {}
    for dirpath, dirnames, _ in os.walk(root_dir):
        # Skip the organized_datasets directory
        if organized_root in dirpath:
            continue
        count = 0
        for dirname in dirnames:
            folder_path = os.path.join(dirpath, dirname)
            # Check if this folder is an extracted folder (was a .zip)
            # and if it still exists in the original location
            if os.path.isdir(folder_path):
                # Check if this folder was moved (does not exist in organized_root)
                found_in_organized = False
                for org_dirpath, org_dirnames, _ in os.walk(organized_root):
                    if dirname in org_dirnames:
                        found_in_organized = True
                        break
                if not found_in_organized:
                    count += 1
        if count > 0:
            unmoved_counts[dirpath] = count

    for folder, cnt in unmoved_counts.items():
        print(f"{cnt} unmoved folders in: {folder}")
    return unmoved_counts


def is_name_in_yamls(name, yaml_data):
    """
    Check if the given name is mentioned in any YAML file's 'files' section.
    Returns a list of YAML file paths where the name is found.
    """
    found_in = []
    for data, yaml_path in yaml_data:
        files = data.get("files", [])
        for file_entry in files:
            link = file_entry.get("link", "")
            # Check if the name matches the extracted folder or zip file
            if name in link or name in os.path.basename(link).replace(".zip", ""):
                found_in.append(yaml_path)
                break
            if found_in:
                print(f"Found '{name}' in {yaml_path}")
    if not found_in:
        print(f"'{name}' not found in any YAML files.")
    return found_in

# Example usage after your main execution:
count_unmoved_folders(root_dir, output_root)
# === MAIN EXECUTION ===
i = 0
j = 0
unzip_all_files(root_dir)
yaml_data = load_yaml_from_dir(root_dir_yaml)
i,j =move_windows_technique_files(yaml_data)
is_name_in_yamls("cmd_process_herpaderping_mimiexplorer", yaml_data)  # Example usage of the function
print(i ,"fichiers non windows")
print(j, "fichiers sans attack_mappings ou files")
print("Processing complete.")

# -----------------------------------------


