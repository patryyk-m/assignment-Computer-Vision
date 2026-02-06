import cv2 as cv
import numpy as np
import time

#read in an image into memory
img = cv.imread('cameraman.png', 0)
copy = img.copy()
#check out some of its pixel values...img[x,y]..try different x and y values
x = 100
y = 100
pix = img[x,y]
print("The pixel value at image location [" + str(x) + "," + str(y) + "] is:" + str(pix))

#implement thresholding ourselves using loops (soooo slow in python)
before = time.time()
thresh = 100
for x in range(0, img.shape[0]):
    for y in range(0, img.shape[1]):
        if img[x,y] > thresh:
            img[x,y] = 255
        else:
            img[x,y] = 0
after = time.time()
print("Time taken to process hand coded thresholding: " + str(after-before))
cv.imshow('thresholded image 1',img)
cv.waitKey(0)

#now lets use the opencv built in function to threshold the image
before = time.time()
ret,copy = cv.threshold(copy,100,255,cv.THRESH_BINARY)
after = time.time()
print("Time taken to process opencv built in thresholding: " + str(after-before))
cv.imshow('thresholded image 2',copy)
cv.waitKey(0)
cv.destroyAllWindows()
