import cv2

image = cv2.imread('e.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 80, 200, cv2.THRESH_BINARY)
invertedImage = cv2.bitwise_not(blackAndWhiteImage)


# reverse color from white to black, and black to white
# blackAndWhiteImage = 255 - blackAndWhiteImage

# cv2.imshow('image', image)
# cv2.imshow('grayImage', grayImage)
cv2.imshow('blackAndWhiteImage', blackAndWhiteImage)
cv2.imshow('invertedImage', invertedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
