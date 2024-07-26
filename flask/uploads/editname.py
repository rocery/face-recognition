import csv
import os

# Ganti dengan path folder gambarmu
folder_path = 'image_copy'

# Nama file CSV yang akan dibuat
csv_file_path = 'image_names.csv'

# Mengambil nama gambar dari folder tanpa ekstensi
image_names = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Menulis nama gambar ke file CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Menulis header (jika perlu)
    writer.writerow(['Image Name'])
    # Menulis nama gambar tanpa ekstensi
    for name in image_names:
        writer.writerow([name])

print(f'Nama gambar (tanpa ekstensi) telah disimpan ke {csv_file_path}')
