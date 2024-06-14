import pickle
import face_recognition
import cv2

model_save_path = "trained_knn_model.clf"
distance_threshold = 0.5
n_neighbors_value = 3
image = "a.png"

def predict(frame):
    with open(model_save_path, 'rb') as f:
        knn_clf = pickle.load(f)

    frame_face_locations = face_recognition.face_locations(frame)

    # Jika tidak ada wajah yang terdeteksi, reutn nilai kosong
    if len(frame_face_locations) == 0:
        return []

    # Encoding wajah yang terdeteksi
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations = frame_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors = n_neighbors_value)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(frame_face_locations))]

    # print(closest_distances)
    # print(are_matches)

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("Tidak Diketahui", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), frame_face_locations, are_matches)]

def show_prediction_labels_on_image(frame, predictions):
    """
    Draws bounding boxes and prediction labels on a frame.

    Args:
        frame (numpy.ndarray): Image frame.
        predictions (list): List of tuples (name, bounding box coordinates)

    Returns:
        numpy.ndarray: Image frame with bounding boxes and labels drawn on it.
    """
    # Draw bounding boxes and labels for each prediction
    for name, (top, right, bottom, left) in predictions:
        # Draw a rectangle around the face with the supplied color
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a filled rectangle to create a box behind the name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        # Use HersheyDuplex font and putText function to draw name on the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame

if __name__ == "__main__":
    # print("Training KNN classifier...")
    # classifier = train("encoding/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")
    # process one frame in every 30 frames for speed
    process_this_frame = 29
    print('Setting cameras up...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    url = 'rtsp://admin:admin123@192.168.10.245:554/Streaming/channels/301'
    url2 = 'http://192.168.15.184:4747/video'
    cap = cv2.VideoCapture(url2)
    # cap.open()
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                predictions = predict(img)
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.namedWindow('Face Regognition', cv2.WINDOW_NORMAL)
            cv2.imshow('Face Regognition', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

    #     else:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
