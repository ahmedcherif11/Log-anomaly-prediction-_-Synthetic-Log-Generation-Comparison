import os
import json

# --- SETTINGS ---
input_root = r"C:\Users\AHMED\Desktop\new-approch\dataset\organized_datasets_atomic"  # CHANGE THIS
total_logs = 0
error_files_count = 0
error_files = []
# Walk through all subfolders
total_files_count = 0
for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".json") or file.endswith(".jsonl"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                    total_logs += line_count
                    print(f"{file_path}: {line_count} logs")
            except Exception as e:
                 # Handle any errors that occur while reading the file
                error_files_count= error_files_count + 1
                error_files.append(file_path)
                print(f"❌ Error reading {file_path}: {e}")
print(f"\n✅ Total JSON log entries across all files: {total_logs}")
for error_file in error_files:
    print(f"Error in file: {error_file}")
print(f"Total files processed: {len (files)}")
print(f"Total files with errors: {error_files_count}")


import os

def count_yaml_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                count += 1
    return count

# Example usage
folder_path = r"C:\Users\AHMED\Desktop\new-approch\sigma\rules\windows"  # Change this to your folder path
yaml_count = count_yaml_files(folder_path)
print(f"Total YAML files found: {yaml_count}")
