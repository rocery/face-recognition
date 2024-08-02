import os
import csv

# Path to your CSV file and the folder containing the directories to rename
csv_file_path = 'pegawai.csv'
folders_path = 'image__'

# Read the CSV file into a dictionary
nip_dict = {}
with open(csv_file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        nip_dict[row['nama'].title()] = row['nip']

# Iterate through the folders and rename them
for folder_name in os.listdir(folders_path):
    formatted_name = folder_name.title()
    nip = nip_dict.get(formatted_name, 'nip')
    new_folder_name = f"{formatted_name}_{nip}"
    os.rename(os.path.join(folders_path, folder_name), os.path.join(folders_path, new_folder_name))

print("Folders have been renamed successfully.")
