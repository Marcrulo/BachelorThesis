
import os
import re
import csv

'''
1) /raw
2) /processed
3) quality_scores.csv
'''

exp_no = input("## Experiment number: ")
split = "train"

dataset_path = os.path.abspath(os.path.join(
        os.getcwd(),
        os.pardir,
        'datasets'
    ))

# Paths
raw_path = os.path.join(dataset_path,'raw',split) 
processed_path = os.path.join(dataset_path,'processed',split)
csv_path = os.path.join(dataset_path,'quality_scores.csv')

# Delete files
raw_files =  os.listdir(raw_path)
for file in raw_files:
    regex=r'\d*\.?\d+'
    matches = re.findall(regex, file)
    if matches[0] == exp_no:
        del_file = os.path.join(raw_path,file)
        os.remove(del_file)
        print("Succesfully deleted:",del_file)

proc_files = os.listdir(processed_path)
for file in proc_files:
    regex=r'\d*\.?\d+'
    matches = re.findall(regex, file)
    if matches[0] == exp_no:
        del_file = os.path.join(processed_path,file)
        os.remove(del_file)
        print("Succesfully deleted:",del_file)

# Delete csv entries
with open(csv_path, 'r') as infile:
    reader = csv.DictReader(infile)
    data_before = [row for row in reader]

with open(csv_path, 'r') as infile:
    reader = csv.DictReader(infile)
    data = [row for row in reader if row['Experiment'] != exp_no]

with open(csv_path, 'w', newline='') as outfile:
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    diff = len(data_before) - len(data)
    entry_str = 'entry' if diff == 1 else 'entries'
    print(f"Succesfully deleted {diff} {entry_str} in CSV-file")