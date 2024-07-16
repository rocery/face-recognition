import cv2
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np
import time
import os
import sys
import argparse
import warnings

# Add the src directory to the Python path

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def liveness_check(frame = None):
    
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="frame",
        help="image used to test")
    args = parser.parse_args()
    
    model_test = AntiSpoofPredict(args.device_id)
    image_cropper = CropImage()
    # image = cv2.imread(frame)
    
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    
    for model_name in os.listdir(args.model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(args.model_dir, model_name))
    
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    
    if label == 1:
        return [True, value, label]
    else:
        return [False, value, label]

def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier and checks for liveness.

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
    
    predictions = []
    
    # Muat model yang sudah ditrain
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Cari lokasi dari wajah yang terdeteksi
    X_face_locations = face_recognition.face_locations(X_frame)

    # Jika tidak ada wajah yang terdeteksi, reutn nilai kosong
    if len(X_face_locations) == 0:
        return []
    
    # if len(X_face_locations) > 1:
    #     return [("Terdeteksi Lebih dari Satu Wajah", X_face_locations[0], 1, 1)]
    
    # Check Liveness
    liveness = liveness_check(X_frame)
    liveness_status = liveness[0]
    liveness_value = liveness[1]
    liveness_label = liveness[2]
    
    # if liveness_status == False:
        # return [("Terdeteksi Palsu", X_face_locations[0], liveness_value, liveness_label)]

    # Menentukan encoding dari wajah yang terdeteksi
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Gunakan algoritma KNN untuk menemukan wajah terdekat/termirip
    # Perhatikan penggunaan linai n_neighbors, sesuaikan dengan nilai pada saat train model
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=3)
    # Cari wajah yang terdekat
    # Gunakan algoritma KNN untuk menemukan wajah terdekat/termirip
    # Perhatikan penggunaan linai n_neighbors, sesuaikan dengan nilai pada saat train model
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=3)
    # Cari wajah yang terdekat
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Return hasil dari algoritma KNN
    #return [(pred, loc, liveness_value, liveness_label) if rec else ("Tidak Dikenali", loc, liveness_value, liveness_label) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    
    for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches):
        if rec and liveness_status == True:
            predictions.append((pred, loc, liveness_value, liveness_label))
        else:
            predictions.append(("Tidak Dikenali", loc, liveness_value, liveness_label))
    
    return predictions

def show_labels_on_image(frame, predictions):
    # Draw a box around the face using the Pillow module
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("Ubuntu.ttf", 14)
    print(predictions)

    time_str = time.strftime("%A, %d-%m-%Y %H:%M:%S", time.localtime())
    draw.text((10, 5), time_str, fill=(0, 0, 0), font=font)
    
    for name, (top, right, bottom, left), value, label in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        
        fig_label = "{}, Value: {:.2f}".format(name, value)
        
        if label == 1:
            fig_outline = (0, 200, 0)
        else:
            fig_outline = (0, 0, 200)
            
        draw.rectangle(((left, top), (right, bottom)), outline = fig_outline, width = 3)
        
        # Draw a solid rectangle below the rectangle, fill it with name
        # draw.rectangle(((left, top - 20), (right, top)), fill = fig_outline, outline=(0, 0, 255))
        draw.text((left + 5, top - 15), fig_label, fill=fig_outline, font=font)
        
    # Remove the drawing library from memory as per the Pillow docs.
    del draw

    opencvimage = np.array(pil_image)
    return opencvimage

if __name__ == "__main__":
    process_this_frame = 39
    print('Setting cameras up...')
    url = 'http://admin:admin@192.168.0.106:8081/'
    url2 = 'http://192.168.15.183:4747/video'
    cap = cv2.VideoCapture(0)
    predictions = []
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 40 == 0:
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_labels_on_image(frame, predictions)
            cv2.imshow('Face Recognition', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
