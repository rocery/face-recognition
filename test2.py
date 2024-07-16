from PIL import Image
import face_recognition
import cv2

# Load the image file into a numpy array
image = face_recognition.load_image_file("images/sample/sample2.png")

# Find all face locations in the image
face_locations = face_recognition.face_locations(image)

# Define a margin (e.g., 20% of the face dimensions)
margin = 1

# Get the dimensions of the image
image_height, image_width, _ = image.shape

# Create a list to store the cropped face images
cropped_faces = []

# Iterate through each face location
for i, face_location in enumerate(face_locations):
    top, right, bottom, left = face_location

    # Calculate the width and height of the face
    face_width = right - left
    face_height = bottom - top

    # Calculate the margin in pixels
    margin_pixels_w = int(face_width * margin)
    margin_pixels_h = int(face_height * margin)

    # Adjust the coordinates to include the margin
    top = max(0, top - margin_pixels_h)
    right = min(image_width, right + margin_pixels_w)
    bottom = min(image_height, bottom + margin_pixels_h)
    left = max(0, left - margin_pixels_w)

    # Extract the face image using the adjusted coordinates
    face_image = image[top:bottom, left:right]

    # Convert the face image from a numpy array to a BGR image fo    OpenCV
    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

    # Append the cropped face image to the list
    cropped_faces.append(face_image_bgr)

# Wait for a key press and close all windows
print(cropped_faces)

# Display each cropped face image
for i, face in enumerate(cropped_faces):
    cv2.imshow(f'Face {i+1}', face)
    print(f"Press any key to see Face {i+1} and close the window to proceed.")
    cv2.waitKey(0)
    cv2.destroyWindow(f'Face {i+1}')

# Close all OpenCV windows
cv2.destroyAllWindows()
