import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import math
from sklearn import neighbors
import pickle

train_folder = "encoding/train"
model_save_path = "trained_knn_model.clf"
n_neighbors_value = 3

def train():
    # Hapus model jika ada
    if os.path.isfile(model_save_path):
        os.remove(model_save_path)
        print('Model sudah ada, dihapus')

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
            # print(os.path.join(train_folder, folder_name, image_path))

            # Load metadata gambar
            image_file = face_recognition.load_image_file(image_path)
            # Deteksi lokasi wajah
            face_detected = face_recognition.face_locations(image_file)

            # Jika wajah != 1
            if len(face_detected) != 1:
                print("File {} tidak bisa diproses karena wajah {}".format(image_path, "tidak terdeteksi" if len(face_detected) < 1 else "terdeteksi lebih dari 1"))
                failed_images_counter += 1

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

    # nilai n_neighbor ditentukan dari banyaknya gambar di tiap folder
    # Jika tiap folder memiliki jumlah ganjil, nilai n_neighbor adalah genap
    # n_neighbors = 3
    # n_neighbors = int(round(math.sqrt(len(X))))
    # if n_neighbors_value is None:
    #     n_neighbors_value = int(round(math.sqrt(len(X))))

    # Train KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors_value, algorithm = 'ball_tree', weights = 'distance')
    knn_clf.fit(X, y)

    # Save trained KNN classifier
    if model_save_path is not None:
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
    train()