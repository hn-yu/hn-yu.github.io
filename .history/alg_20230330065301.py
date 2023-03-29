# Import required libraries
import cv2
import scipy.ndimage
import numpy as np
import pywt
import matplotlib.pyplot as plt
import sys

# Read the images
imgA = cv2.imread('images/source10_1.tif')
imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
print(imgA.shape)
aA, aD = pywt.dwt(imgA, 'haar')
imgB = cv2.imread('images/source10_2.tif')
bA, bD = pywt.dwt(imgA, 'haar')

# # Check the file type by:
# print(open('source10_1.tif', 'rb').read(5))

# This function applies an approximation of fractional-order differentation on the primitive detail map extracted from the PAN image and a linear combination of MS image.
def FractionalDiff(primitive_map):
    TFval = 0.10
    # create a 5x5 matrix
    M = np.matrix([[(-TFval)*(-TFval+1)/2, 0, ((-TFval)*(-TFval+1))/2, 0, ((-TFval)*(-TFval+1))/2], 
        [0, -TFval, -TFval, -TFval, 0],
        [(-TFval)*(-TFval+1)/2, -TFval, 8, -TFval, ((-TFval)*(-TFval+1))/2],
        [0, -TFval, -TFval, -TFval, 0],
        [(-TFval)*(-TFval+1)/2, 0, ((-TFval)*(-TFval+1))/2, 0, ((-TFval)*(-TFval+1))/2]])
    SEl = np.sum(M)
    M  = (1/SEl)*M
    refined_map = scipy.ndimage.convolve(primitive_map, M, mode='nearest')
    return refined_map

# Main Function
temp = FractionalDiff(imgA)
# concatenate image 
Verti = np.concatenate((imgA, temp), axis=0)
while True:
    cv2.imshow('VERTICAL', Verti)
    cv2.waitKey(0)
    sys.exit()