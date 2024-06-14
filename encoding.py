import face_recognition
import os
import pickle

def encode_faces_from_train_folder():
    # Path to the train folder inside the encoding folder
    train_folder_path = 'encoding/train'
    
    # Dictionary to hold the encodings
    face_encodings = {}
    
    # Iterate over each file in the train folder
    for filename in os.listdir(train_folder_path):
        print(filename)
#         if filename.endswith('.jpg') or filename.endswith('.png'):
#             # Construct full file path
#             file_path = os.path.join(train_folder_path, filename)
#             print(f"Processing file: {filename}")  # Status update for current file in encoding progress
#             # Load the image file
#             image = face_recognition.load_image_file(file_path)
#             # Find face encodings
#             encodings = face_recognition.face_encodings(image)
#             if encodings:
#                 # Assuming each image has one face for simplicity
#                 face_encodings[filename] = encodings[0]
    
#     # Save the encodings to a file for later use
#     with open('encoded_faces.pkl', 'wb') as f:
#         pickle.dump(face_encodings, f)

#     return face_encodings

# # Example usage
# encoded_faces = encode_faces_from_train_folder()
# print("Encoded faces saved to file.")



