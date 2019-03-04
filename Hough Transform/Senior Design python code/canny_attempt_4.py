# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:35:07 2019

@author: atm52
"""

import cv2
from matplotlib import pyplot as plt
 
def nothing(x):
    pass
 
 
img_noblur = cv2.imread('blue.png', 0) #loading in the picture
img = cv2.GaussianBlur(img_noblur, (7,7),0)
 
canny_edge = cv2.Canny(img, 0, 0) #using open source library to creat
 
cv2.imwrite( 'canny_edge.jpg', canny_edge)
cv2.imshow('image', img)
cv2.imshow('canny_edge', canny_edge)
 
cv2.createTrackbar('min_value','canny_edge',0,500,nothing) #creating a trackbar to take out background noise
cv2.createTrackbar('max_value','canny_edge',0,500,nothing) # setting the min and max hysteresis values
 
while(1):
    cv2.imshow('image', img)  #displaying the image
    cv2.imshow('canny_edge', canny_edge)
     
    min_value = cv2.getTrackbarPos('min_value', 'canny_edge')
    max_value = cv2.getTrackbarPos('max_value', 'canny_edge')
 
    canny_edge = cv2.Canny(img, min_value, max_value)
     
    k = cv2.waitKey(37)
    if k == 0:
        break
        #exit(0)
        