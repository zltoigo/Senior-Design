# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:40:02 2019

@author: atm52
"""
import cv2
import numpy as np

image = cv2.imread('test.png')
#cv2.imshow('orig',image)
#cv2.waitKey(0)

image = cv2.resize( image, (144,144))

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
#cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh)
#cv2.waitKey(0)

#dilation
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated',img_dilation)
#cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

#roi_marked = 0

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    height, width, channels = roi.shape

    if height >= 25:
        if width >= 25:
             # show ROI
             roi = cv2.resize( roi, (144, 144) )
             #roi_marked = roi_marked + roi;
             cv2.imshow('segment no:'+str(i),roi)
             cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
             cv2.waitKey(0)
    
    # show ROI
#    cv2.imshow('segment no:'+str(i),roi)
#    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
#    cv2.waitKey(0)

#cv2.imshow( 'marked areas', roi_marked )
#cv2.waitKey(0)