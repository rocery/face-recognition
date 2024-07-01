#H1 How to Install in Raspberry Pi 4
```
1. Cmake
sudo apt install cmake

2. dlib
wget https://github.com/prepkg/dlib-raspberrypi/releases/latest/download/dlib_64.deb
sudo apt install -y ./dlib_64.deb

3. opencv-python
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y

remove all library from bash if any error occureed (libjasper-dev)

pip install opencv-python
if error
sudo apt-get install python-OpenCV

4. NumPy 
pip install numpy==1.26.4

5. face_recognition_models
wget https://www.piwheels.org/simple/face-recognition-models/face_recognition_models-0.3.0-py2.py3-none-any.whl
pip3 install face_recognition_models-0.3.0-py2.py3-none-any.whl

6. face_recognition
git clone https://github.com/ageitgey/face_recognition.git
cd face_recognition
edit setup.py -> comment dlib, face_recognition_models, and numpy
python3 setup.py install
```


Please make your own venv,

in ubuntu update or isntall lib using this command:
pip install --break-system-packages --user <package name>

Future Project:

1. Face Recognition
2. Live OCR
3. Counter -> YOLO

python 3.10.12
cmake
dlib
face_recognition

if lmdb can't installed in windows os use python > 3.12 try use this repo -> 
-> git clone https://github.com/Bye-lemon/py-lmdb/tree/master
-> cd py-lmdb 
-> use pip install .
