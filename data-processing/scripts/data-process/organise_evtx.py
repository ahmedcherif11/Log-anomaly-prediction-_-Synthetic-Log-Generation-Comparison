import os
import shutil
import csv
import time

# Paths
csv_file = r"C:\Users\AHMED\Desktop\new-approch\ConvertedCSVs\EVTX-ATT&CK-Content-Summary.csv"
output_dir = r"C:\Users\AHMED\Desktop\new-approch\dataset\organized"

os.makedirs(output_dir, exist_ok=True)

def create_tactic_folders(csv_file, output_dir):
    # Read the CSV and create a folder for each unique tactic
    tactics_set = set()
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            tactics = [t.strip() for t in row['Tactic'].split(',')]
            tactics_set.update(tactics)

    for tactic in tactics_set:
        tactic_folder = os.path.join(output_dir, tactic)
        os.makedirs(tactic_folder, exist_ok=True)

    print("Tactic folders created.")

# Function to create technique folders based on CSV content
def create_technique_folders(csv_file, output_dir):
    # Read the CSV and create a folder for each unique technique ID under its tactic folder
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            tactics = [t.strip() for t in row['Tactic'].split(',')]
            technique_id = row['ID'].strip()
            if '.' in technique_id:
                parent_id = technique_id.split('.')[0]
            else:
                parent_id = technique_id
            for tactic in tactics:
                tactic_folder = os.path.join(output_dir, tactic)
                parent_folder = os.path.join(tactic_folder, parent_id)
                technique_folder = os.path.join(parent_folder, technique_id)
                os.makedirs(technique_folder, exist_ok=True)
    print("Technique folders created.")

# Function to extract filenames and replace .evtx with .csv
def extract_and_replace_filenames(csv_file):
    results = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            tactic = row['Tactic'].strip()
            technique_id = row['ID'].strip()
            filename = row['files'].strip() if 'files' in row else row.get('files', '').strip()
            if filename.lower().endswith('.evtx'):
                filename = filename[:-5] + '.csv'
            results.append((tactic, technique_id, filename))
    return results

# Function to copy .evtx files to the exact folder generated in create_technique_folders
source_dir = r"C:\Users\AHMED\Desktop\new-approch\ConvertedCSVs"
def copy_files_to_organized(results, source_dir, output_dir):
  
    for tactic, technique_id, filename in results:
        if '.' in technique_id:
            parent_id = technique_id.split('.')[0]
        else:
            parent_id = technique_id
        # Search for the file in the source_dir recursively
        src_path = None
        for root, dirs, files in os.walk(source_dir):
            print(f"Searching in: {root}")
            
            for f in files:
                if f.strip().lower() == filename.strip().lower():
                    src_path = os.path.join(root, f)
                    break
            if src_path:
                print(f"Found {filename} in {root}")
                break
        if src_path is None:
            print(f"File not found in source_dir: {filename}")
            continue
        dest_dir = os.path.join(output_dir, tactic, parent_id, technique_id)
        #print(f"Source path: {src_path}")
        #print(f"Destination directory: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        if os.path.isfile(src_path):
            if os.path.abspath(src_path) == os.path.abspath(dest_path):
                #print(f"Source and destination are the same file, skipping: {src_path}")
                continue
            shutil.copy2(src_path, dest_path)
            #print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"File not found: {src_path}")
# Function to organize files based on the CSV content
def report_copied_files(results, source_dir, output_dir, not_copied_log="not_copied_files.txt"):
    copied = 0
    not_found = 0
    not_found_files = []
    not_found_ids = []
    for tactic, technique_id, filename in results:
        if '.' in technique_id:
            parent_id = technique_id.split('.')[0]
        else:
            parent_id = technique_id
        src_path = None
        for root, dirs, files in os.walk(source_dir):
            print(f"Searching in: {root}")
            
            for f in files:
                if f.strip().lower() == filename.strip().lower():
                    src_path = os.path.join(root, f)
                    break
            if src_path:
                print(f"Found {filename} in {root}")
                break
        if src_path is None:
            print(f"File not found in source_dir: {filename}")
            continue
        dest_dir = os.path.join(output_dir, tactic, parent_id, technique_id)
        dest_path = os.path.join(dest_dir, filename)
        if os.path.isfile(src_path):
            copied += 1
        else:
            not_found += 1
            not_found_files.append(src_path)
            not_found_ids.append(technique_id)
    print(f"Files copied: {copied}")
    print(f"Files not found: {not_found}")
    # Write not found files to a txt file
    with open(not_copied_log, "w", encoding="utf-8") as f:
        for file_path, technique_id in zip(not_found_files, not_found_ids):
            f.write(file_path + "      " + technique_id +"\n")
    print(f"List of not copied files written to {not_copied_log}")

def find_specific_file(filename, search_dir):
    """
    Search for a specific file by name in the given directory (recursively).
    Returns the full path if found, else None.
    """
    for root, dirs, files in os.walk(search_dir):
        print(f"Searching in: {root}")
        print(f"Files in {root}: {files}")
        if filename in files:
            print(f"Found {filename} in {root}")
            return os.path.join(root, filename)
            
    return None
# Call the function

create_tactic_folders(csv_file, output_dir)
time.sleep(1)  # Optional: wait for 2 seconds before creating technique folders
create_technique_folders(csv_file, output_dir)
time.sleep(1)  # Optional: wait for 2 seconds before copying files
copy_files_to_organized(extract_and_replace_filenames(csv_file),source_dir, output_dir)
file_path = r"C:\Users\AHMED\Desktop\new-approch\ConvertedCSVs\not_copied_files.txt"
report_copied_files(extract_and_replace_filenames(csv_file), source_dir, output_dir , not_copied_log="not_copied_files.txt")
 





# Call the function
#create_tactic_folders(csv_file, output_dir)
#create_technique_folders(csv_file, output_dir)