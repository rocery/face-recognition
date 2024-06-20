import cv2
import numpy as np

# Load the image
image = cv2.imread('b.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to find rectangles (potential plates)
plates = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:  # Assuming license plate has 4 corners
        plates.append(approx)

# Draw contours on the original image
cv2.drawContours(image, plates, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('License Plate Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
