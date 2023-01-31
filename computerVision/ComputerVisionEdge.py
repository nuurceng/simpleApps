import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

original_image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)

# we have to transform the image into grayscale
# OpenCV handles BGR instead of RGB
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

result_image = cv2.Laplacian(gray_image, -1)

cv2.imshow('Original Image', gray_image)
cv2.imshow('Result Image', result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

