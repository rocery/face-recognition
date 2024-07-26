import os
import shutil

# Path ke folder yang berisi gambar
folder_path = 'image_peg'

# Daftar nama dan folder tujuan
name_folder_map = {
    'Nunik Gia H': 'Nunik Gia H',
    
    
}


# Buat folder tujuan jika belum ada
for folder in name_folder_map.values():
    destination_folder = os.path.join(folder_path, folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

# Dapatkan daftar semua file di folder
file_names = os.listdir(folder_path)

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    # Pastikan hanya memproses file, bukan direktori
    if os.path.isfile(file_path):
        for name, folder in name_folder_map.items():
            if name in file_name:
                # Bangun path lama dan path baru
                old_file_path = os.path.join(folder_path, file_name)
                new_file_path = os.path.join(folder_path, folder, file_name)
                
                # Pindahkan file
                shutil.move(old_file_path, new_file_path)
                print(f"File {file_name} dipindahkan ke folder {folder}.")
                break  # Setelah memindahkan file, tidak perlu cek nama lain untuk file ini

print("File yang sesuai berhasil dipindahkan.")
