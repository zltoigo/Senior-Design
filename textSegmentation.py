# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:40:02 2019

@author: atm52
"""
import cv2
import numpy as np

RESIZED_HEIGHT = 72

def resizeToHeight(newHeight, oldImage):
    """Maintains aspect ratio when resizing to specified height"""
    (h,w) = oldImage.shape[:2]
    r = newHeight / float(h)
    dim = (int(w * r), newHeight)
    resized = cv2.resize(oldImage, dim, interpolation=cv2.INTER_AREA)
    return resized

image = cv2.imread('test.png')

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

#dilation
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

#roi_marked = 0
streetnameWords = list()

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]
    height, width, channels = roi.shape

    #only choose words of a certain size
    if height >= image.shape[0]* 0.3:
        if width >= image.shape[1]*0.2 and width <= image.shape[1]*0.9:
             roi = resizeToHeight(RESIZED_HEIGHT, roi)
             streetnameWords.append(roi)
             space = np.zeros((RESIZED_HEIGHT,int(RESIZED_HEIGHT/2.5), 3), np.uint8)
             space.fill(255)
             streetnameWords.append(space)

#remove extra space
streetnameWords.pop()
joinedSegments = np.hstack(streetnameWords)

cv2.imshow( 'marked areas', joinedSegments )
cv2.imwrite('test.png', joinedSegments)
cv2.waitKey(0)