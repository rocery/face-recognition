import cv2
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np
import time

def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("File Encoding Tidak Ditemukan")

    # Muat model yang sudah ditrain
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Cari lokasi dari wajah yang terdeteksi
    X_face_locations = face_recognition.face_locations(X_frame)

    # Jika tidak ada wajah yang terdeteksi, reutn nilai kosong
    if len(X_face_locations) == 0:
        return []

    # Menentukan encoding dari wajah yang terdeteksi
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Gunakan algoritma KNN untuk menemukan wajah terdekat/termirip
    # Perhatikan penggunaan linai n_neighbors, sesuaikan dengan nilai pada saat train model
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=3)
    # Cari wajah yang terdekat
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Return hasil dari algoritma KNN
    return [(pred, loc) if rec else ("Tidak Dikenali", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_labels_on_image(frame, predictions):
    # for name, (top, right, bottom, left) in predictions:
    #     print("Prediction: {}, Top: {}, Right: {}, Bottom: {}, Left: {}".format(name, top, right, bottom, left))
    
    height_frame, width_frame, _ = frame.shape

    # Draw a box around the face using the Pillow module
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("Ubuntu.ttf", 16)

    time_str = time.strftime("%A, %d-%m-%Y %H:%M:%S", time.localtime())
    text_width, text_height = draw.textsize(time_str)
    draw.rectangle(((5, 5), (text_width + 70, text_height + 12)), fill=(0, 0, 0, 100), outline=(0, 0, 0, 100))
    draw.text((10, 5), time_str, fill=(255, 255, 255), font=font)
    
    for name, (top, right, bottom, left) in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width = 3)
        
        # Draw a solid rectangle below the rectangle, fill it with name
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, top - text_height - 10), (right, top)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 10, top - text_height - 5), name, fill=(255, 255, 255), font=font)

        draw.text((10, height_frame - text_height - 10), name, fill=(255, 255, 255), font=font)
        
    # Remove the drawing library from memory as per the Pillow docs.
    del draw

    opencvimage = np.array(pil_image)
    return opencvimage

if __name__ == "__main__":
    # process one frame in every 30 frames for speed
    process_this_frame = 39
    print('Setting cameras up...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    url = 'http://admin:admin@192.168.0.106:8081/'
    url2 = 'http://192.168.15.183:4747/video'
    cap = cv2.VideoCapture(0)
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 40 == 0:
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_labels_on_image(frame, predictions)
            # cv2.namedWindow('Face Regognition', cv2.WINDOW_NORMAL)
            cv2.imshow('Face Regognition', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

