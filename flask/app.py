from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, stream_with_context
from predict import predict, show_labels_on_image
from read_data import *
import cv2
import csv
import os
import time

app = Flask(__name__)
app.secret_key = 'itbekasioke'  # Necessary for using flash messages

# Path to the CSV file
CSV_FILE_PATH = 'static/data/data.csv'
CSV_FILE_PATH_GROUP = 'static/data/group_data.csv'

# Ensure the CSV file exists and has the correct headers
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Category', 'Time'])

if not os.path.exists(CSV_FILE_PATH_GROUP):
    with open(CSV_FILE_PATH_GROUP, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Names', 'Time'])
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_recognition')
def face_recognition():
    return render_template('face_recognition/index.html')

@app.route('/face_recognition/facerec')
def facerec():
    data_csv = read_csv(CSV_FILE_PATH)
    return render_template('face_recognition/facerec.html', data_csv=data_csv)

@app.route('/face_recognition/facerec_group')
def facerec_group():
    with open(CSV_FILE_PATH_GROUP, 'r') as file:
        reader = reversed(list(csv.reader(file)))
        last_row = next(reader)
        last_index = last_row[0]
    data_csv = read_csv_group(CSV_FILE_PATH_GROUP)
    return render_template('face_recognition/facerec_group.html', data_csv=data_csv, last_index=last_index)

@app.route('/submit_facerec', methods=['POST'])
def submit_facerec():
    if request.method == 'POST':
        # Get data from the form
        time_category = request.form.get('time_category')
        name_input = request.form.get('name_input')
        time_str = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        
        if name_input in ['-', '', 'Tidak Dikenali', 'Tidak Terdeteksi']:
            flash(f"Data terdeteksi salah, silahkan ulangi proses Face Recognition. Nama: {name_input}")
            return redirect(url_for('facerec'))           
        else:
            # Save data to CSV
            with open(CSV_FILE_PATH, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name_input, time_category, time_str])
            flash(f"Berhasil! Nama: {name_input}, Kategori: {time_category}, Waktu: {time_str}")
            return redirect(url_for('facerec'))
        
    else:
        flash("Error, data terdeteksi tidak ada, silahkan ulangi proses Face Recognition")
        return redirect(url_for('facerec'))

@app.route('/group_submit_facerec', methods=['POST'])
def group_submit_facerec():
    if request.method == 'POST':
        # Get data from the form
        name_input = request.form.get('name_input')
        time_str = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        
        # Save data to CSV
        with open(CSV_FILE_PATH_GROUP, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name_input, time_str])
        return redirect(url_for('facerec_group'))
        
    else:
        flash("Error, data terdeteksi tidak ada, silahkan ulangi proses Face Recognition")

url1 = 'http://192.168.15.195:4747/video'
cap = cv2.VideoCapture(0)
predictions = []
def generate_frames():
    process_this_frame = 39
    while True:
        # Baca frame dari webcam
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode frame ke format JPEG
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 40 == 0:
                global predictions
                predictions = predict(img, model_path="static/clf/trained_knn_model.clf")
            frame = show_labels_on_image(frame, predictions)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Hasilkan frame sebagai aliran byte
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Rute untuk streaming video dengan deteksi wajah
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pred')
def pred():
    # Initialize name_ outside the loop
    name_ = None
    for name, _ in predictions:
        if name is None:
            name_ = "Tidak Terdeteksi"
        else:
            name_ = name
    return jsonify({'name_': name_})

@app.route('/group_pred')
def group_pred():
    names = []
    for name, _ in predictions:
        if name is not None:
            names.append(name)

    return jsonify({'names': names})

if __name__ == '__main__':
    # if debug True, camera only run once then blank
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
 