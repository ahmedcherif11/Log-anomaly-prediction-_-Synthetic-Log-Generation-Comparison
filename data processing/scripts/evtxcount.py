import os
import csv

root_dir = r'C:\Users\AHMED\Desktop\new-approch\dataset\organized'
# Ensure the root_dir is correctly set to your dataset directory
total_log_count = 0

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                row_count = sum(1 for row in reader) - 1  # subtract header
                total_log_count += row_count
                print(f"{file_path}: {row_count} logs")

print(f"\nTotal log entries (excluding headers): {total_log_count}")
