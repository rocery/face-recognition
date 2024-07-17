import os
import cv2
import numpy as np
import argparse
import warnings
import time
import face_recognition
import pickle
from PIL import Image, ImageDraw, ImageFont

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def liveness_check(frame, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    test_result = []
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_loccations = face_recognition.face_locations(rgb_frame)
    
    if len(face_loccations) == 0:
        return []
        
    for face_location in face_loccations:
        top, right, bottom, left = face_location
        image_bbox = [left, top, right - left, bottom - top]
        prediction = np.zeros((1, 3))
        
        for model_name in os.listdir(model_dir):
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
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        
        lab_val = (label, value, face_location)
        test_result.append(lab_val)
        
    return test_result

def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("File Encoding Tidak Ditemukan")
    
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    
    predictions = []
    
    desc = "liveness_check"
    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test"
    )
    args = parser.parse_args()
    
    liveness = liveness_check(X_frame, args.model_dir, 0)
    if len(liveness) == 0:
        return []
    
    if len(liveness) > 1:
        return [("Terdeteksi Lebih dari Satu Wajah", liveness[0][2], 10, 10)]
    
    X_label = []
    X_value = []
    X_face_locations = []
    for data in liveness:
        X_label.append(data[0])
        X_value.append(data[1])
        X_face_locations.append(data[2])
        
    face_encodings = face_recognition.face_encodings(X_frame, known_face_locations = X_face_locations)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors = 3)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_encodings))]

    for pred, loc, rec, label, value in zip(knn_clf.predict(face_encodings), X_face_locations, are_matches, X_label, X_value):
        if rec and label == 1:
            predictions.append((pred, loc, label, value))
        elif rec and label == 1 or label == 2:
            pred_ = "{}, Palsu".format(pred)
            predictions.append(("Palsu", loc, label, value))
        else:
            predictions.append(("Tidak Diketahui", loc, label, value))
    
    return predictions

def show_labels_on_image(frame, predictions):
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("Ubuntu.ttf", 14)
    print(predictions)
    
    time_str = time.strftime("%A, %d-%m-%Y %H:%M:%S", time.localtime())
    draw.text((10, 5), time_str, fill = (0, 0, 0), font = font)
    
    for name, (top, right, bottom, left), label, value in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        
        if label == 1:
            fig_outline = (200, 0, 0)
        else:
            fig_outline = (0, 0, 200)
        
        fig_label = "{}, Value: {:.2f}".format(name, value)
        draw.rectangle(((left, top), (right, bottom)), outline = fig_outline, width = 3)
        
        # Draw a solid rectangle below the rectangle, fill it with name
        # draw.rectangle(((left, top - 20), (right, top)), fill = fig_outline, outline = fig_outline)
        draw.text((left + 5, top - 15), fig_label, fill = fig_outline, font = font)
        
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
            img = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 40 == 0:
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_labels_on_image(frame, predictions)
            cv2.imshow('Face Recognition', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)