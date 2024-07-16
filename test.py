import os
import cv2
import numpy as np
import argparse
import warnings
import time
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import pickle

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

def liveness_check(frame, model_dir, device_id):
    
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    # image = cv2.imread(SAMPLE_IMAGE_PATH + frame)
    result_test = []
    
    # Use face_recognition to detect face locations
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    for face_location in face_locations:
        top, right, bottom, left = face_location
        image_bbox = [left, top, right - left, bottom - top]  # x, y, w, h format
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
        
        # draw result of prediction for each face
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        # if label == 1:
        #     result_text = "RealFace Score: {:.2f}".format(value)
        #     color = (255, 0, 0)
        #     # result = (label, value)
        #     # result_test.append(result)
        # else:
        #     result_text = "FakeFace Score: {:.2f}".format(value)
        #     color = (0, 0, 255)
        #     # result = (label, value)
        #     # result_test.append(result)
        
        lab_val = (label, value, face_location)
        result_test.append(lab_val)
        
        # Draw bounding box and result text on the original image
        # cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        # cv2.putText(image, result_text, (left, top - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)
    #for result in result_test:
    # print(result_test)
    # print(face_locations)
    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    # for data_face in result_test:
    #     print(data_face[2])
    # print(result_test)
    # print(face_locations)
    return result_test

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
    # X_face_locations = face_recognition.face_locations(X_frame)

    # # Jika tidak ada wajah yang terdeteksi, reutn nilai kosong
    
    # if len(X_face_locations) > 1:
    #     return [("Terdeteksi Lebih dari Satu Wajah", X_face_locations[0], 1, 1)]
    
    # for faces in X_face_locations:
    #     for face_data in faces:
    desc = "liveness_check"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    args = parser.parse_args()
    # Check Liveness
    liveness = liveness_check(X_frame, args.model_dir, 0)
    X_label = []
    X_value = []
    X_face_locations = []
    for data in liveness:
        X_label.append(data[0])
        X_value.append(data[1])
        X_face_locations.append(data[2])
        
    if len(X_face_locations) == 0:
        return []
    
    # if liveness_status == False:
        # return [("Terdeteksi Palsu", X_face_locations[0], liveness_value, liveness_label)]

    # Menentukan encoding dari wajah yang terdeteksi
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Cari wajah yang terdekat
    # Gunakan algoritma KNN untuk menemukan wajah terdekat/termirip
    # Perhatikan penggunaan linai n_neighbors, sesuaikan dengan nilai pada saat train model
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=3)
    # Cari wajah yang terdekat
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Return hasil dari algoritma KNN
    #return [(pred, loc, liveness_value, liveness_label) if rec else ("Tidak Dikenali", loc, liveness_value, liveness_label) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    
    for pred, loc, rec, label, value in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches, X_label, X_value):
        # if rec and label == 1:
        #     predictions.append((pred, loc, label, value))
        # elif rec and label != 1:
        #     predictions.append(("Error", loc, label, value))
        # else:
        #     predictions.append(("Tidak Terdeteksi", loc, 2, 2))
        predictions.append((pred, loc, label, value))
    print("Pred: {}".format(predictions))
    return predictions

def show_labels_on_image(frame, predictions):
    # Draw a box around the face using the Pillow module
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("Ubuntu.ttf", 14)
    # print(predictions)

    time_str = time.strftime("%A, %d-%m-%Y %H:%M:%S", time.localtime())
    draw.text((10, 5), time_str, fill=(0, 0, 0), font=font)
    
    for name, (top, right, bottom, left), label, value in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        
        
        
        if label == 1:
            fig_outline = (0, 200, 0)
            fig_label = "{}, Value: {:.2f}".format(name, value)
        else:
            fig_outline = (0, 0, 200)
            fig_label = "{}, Value: {:.2f}".format("Error", value)
            
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
