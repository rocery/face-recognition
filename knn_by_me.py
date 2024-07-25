import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
from sklearn import neighbors
import pickle
import numpy as np
import shutil
import csv

train_folder = "uploads/train"
trained_folder = "uploads/trained"
model_save_path = "static/clf/trained_knn_model.clf"
n_neighbors_value = 3
csv_success = 'uploads/success_trained.csv'
csv_fail = 'uploads/fail_trained.csv'

def train():
    # Array untuk menyimpan hasil encodings
    X = []
    # Array untuk menyimpan label/nama
    y = []

    # Total folder yang sudah diproses
    folder_counter = 0
    # Total gambar yang sudah diproses
    image_counter = 0
    # Total gambar yang gagal diproses
    failed_images_counter = 0
    # Total waktu Encoding
    total_time_encoding = 0

    # Looping setiap folder
    for folder_name in os.listdir(train_folder):
        folder_counter += 1

        # Looping setiap gambar pada folder_name
        for image_path in image_files_in_folder(os.path.join(train_folder, folder_name)):
            image_counter += 1

            # Waktu Mulai Encoding
            start_time_encoding = time.time()

            # Load metadata gambar
            image_file = face_recognition.load_image_file(image_path)
            # Deteksi lokasi wajah
            face_detected = face_recognition.face_locations(image_file)

            # Jika wajah != 1
            if len(face_detected) != 1:
                print("File {} tidak bisa diproses karena wajah {}".format(image_path, "tidak terdeteksi" if len(face_detected) < 1 else "terdeteksi lebih dari 1"))
                failed_images_counter += 1
                
                with open (csv_fail, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([folder_counter, folder_name, image_path, "Wajah tidak terdeteksi" if len(face_detected) < 1 else "Wajah terdeteksi lebih dari 1"])

            # Jika wajah = 1
            else:
                # Mengambil encoding dari gambar
                face_encodings = face_recognition.face_encodings(image_file, face_detected)[0]

                # Waktu Akhir Encoding
                end_time_encoding = time.time()
                # Waktu Encoding per file
                time_encoding = end_time_encoding - start_time_encoding
                # Total Waktu Encoding
                total_time_encoding = total_time_encoding + time_encoding

                # Menyimpan encoding
                X.append(face_encodings)
                # Menyimpan label/nama
                y.append(folder_name)

                print(f"{folder_counter}. {folder_name}:", 
                    f"File {image_counter}. {image_path} diproses. Waktu Encoding: {time_encoding:.2f} detik")
                
                with open (csv_success, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([folder_counter, folder_name, image_path])

        # Pindahkan folder yang sudah diproses ke folder trained
        shutil.move(os.path.join(train_folder, folder_name), os.path.join(trained_folder, folder_name))
        print(f"Folder {folder_name} telah dipindahkan ke {trained_folder}")

    # Jika model sudah ada, muat model yang ada dan tambahkan data baru
    if os.path.isfile(model_save_path):
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)
        knn_clf._fit_X = np.concatenate((knn_clf._fit_X, X), axis=0)
        knn_clf._y = np.concatenate((knn_clf._y, y), axis=0)
        knn_clf.classes_ = np.concatenate((knn_clf.classes_, np.array(y)))
        print("Model sudah ada, data baru ditambahkan")
    else:
        # Train KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors_value, algorithm='ball_tree', weights='distance')
        knn_clf.fit(X, y)
        print("Model baru dibuat dan dilatih")

    # Save trained KNN classifier
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    print(f"Folder diproses: {folder_counter}")
    print(f"Gambar diproses: {image_counter}")
    print(f"Total Waktu Encoding: {total_time_encoding:.2f} detik")
    print(f"Total Waktu Encoding: {total_time_encoding // 60:.0f} menit {total_time_encoding % 60:.0f} detik")
    print(f"Rata-Rata Waktu Encoding per Gambar: {total_time_encoding / image_counter:.2f} detik")
    print(f"Gambar yang gagal diproses: {failed_images_counter}")

    return knn_clf

if __name__ == "__main__":
    # Membuat folder trained jika belum ada
    if not os.path.exists(trained_folder):
        os.makedirs(trained_folder)
    
    # Membuat CSV jika belum ada
    if not os.path.exists(csv_success):
        with open(csv_success, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['No.', 'Folder', 'File'])
    # Membuat CSV jika belum ada
    if not os.path.exists(csv_fail):
        with open(csv_fail, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['No.', 'Folder', 'File', 'Information'])
            
    train()
