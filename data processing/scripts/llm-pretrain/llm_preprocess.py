import os
import matplotlib.pyplot as plt

def count_empty_folders(root_folder):
    empty_count = 0
    empty_folders = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # A folder is empty if it has no files and no subfolders
        if not dirnames and not filenames:
            empty_count += 1
            empty_folders.append(os.path.basename(dirpath))
    return empty_count, empty_folders

def count_total_folders(root_folder):
    total_count = 0
    for dirpath, dirnames, filenames in os.walk(root_folder):
        total_count += len(dirnames)
    return total_count

def delete_empty_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        if not dirnames and not filenames:  # Check if the folder is empty
            try:
                os.rmdir(dirpath)  # Remove the empty folder
                print(f"Deleted empty folder: {dirpath}")
            except OSError as e:
                print(f"Error deleting {dirpath}: {e}")

def heat_map_folders(root_folder):
    folder_line_counts = []
    for dirpath, _, filenames in os.walk(root_folder):
        total_lines = 0
        for f in filenames:
            if f.endswith('.jsonl') or f.endswith('.json'):
                file_path = os.path.join(dirpath, f)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        total_lines += sum(1 for _ in file)
                except OSError as e:
                    print(f"Error opening {file_path}: {e}")
        folder_line_counts.append(total_lines)

def count_json_logs(input_root):
    total_logs = 0
    error_files_count = 0
    error_files = []
    total_files_count = 0
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith(".json") or file.endswith(".jsonl"):
                total_files_count += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        line_count = sum(1 for _ in f)
                        total_logs += line_count
                except Exception as e:
                    error_files_count += 1
                    error_files.append(file_path)
    print(f"\n✅ Total JSON log entries across all files: {total_logs}")
   
    print(f"Total files processed: {total_files_count}")
    print(f"Total files with errors: {error_files_count}")

def count_json_files(root_folder):
    json_file_count = 0
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith('.json') or f.endswith('.jsonl'):
                json_file_count += 1
    print(f"Total JSON files: {json_file_count}")
    return json_file_count

def count_all_files(root_folder):
    file_count = 0
    for dirpath, _, filenames in os.walk(root_folder):
        file_count += len(filenames)
    print(f"Total files (all types): {file_count}")
    return file_count

def delete_cap_files(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith('.cap'):
                file_path = os.path.join(dirpath, f)
                try:
                    os.remove(file_path)
                    print(f"Deleted .cap file: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

def count_folders_starting_with_T1(root_folder):
    t1_count = 0
    t1_folders = []
    TA_count = 0
    TA_folders = []
    other_folders = []
    other_count = 0
    for dirpath, dirnames, _ in os.walk(root_folder):
        for dirname in dirnames:
            if dirname.startswith('T') and  not dirname.startswith('TA'):
                t1_count += 1
                t1_folders.append(os.path.join(dirpath, dirname))
            elif dirname.startswith('TA'):
                TA_count += 1
                TA_folders.append(os.path.join(dirpath, dirname))
            else:
                other_count += 1
                other_folders.append(os.path.join(dirpath, dirname))
    print(f"Total folders starting with 'T1': {t1_count}")
    print(f"Total folders starting with 'TA': {TA_count}")
    print(f"Total other folders: {other_count}")
    return t1_count, t1_folders, TA_count, TA_folders , other_count, other_folders

# Example usage:
root = r"C:\Users\AHMED\Desktop\new-approch\dataset\organized_datasets_atomic"
count, empty_folders = count_empty_folders(root)
print(f"Empty folders count: {count}")
count_total = count_total_folders(root)
print(f"Total folders count: {count_total}")
delete_empty_folders(root)
print(f"Empty folders: {empty_folders}")
count_json_logs(root)
count_json_files(root)
count_all_files(root)
delete_cap_files(root)

t1_count, t1_folders, TA_count, TA_folders , other_count, other_folders =count_folders_starting_with_T1(root)
with open("other_folders.txt", "w", encoding="utf-8") as f:
    for folder in other_folders:
        f.write(folder + "\n")
print("Other folders saved to other_folders.txt")