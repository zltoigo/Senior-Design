# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:21:16 2019

@author: Alexander
"""

import cv2
import numpy as np

#cap = cv2.VideoCapture(0)
img = cv2.imread( 'Zoomedin.png', 0 )



# Take each frame
#_, frame = cap.read()

# Convert BGR to HSV
#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )

# define range of blue color in HSV
lower_green = np.array([50,0,0])
upper_green = np.array([80,255,255])
#lower_green = np.array([100,100,200])
#upper_green = np.array([255,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(img, lower_green, upper_green)

imask = mask > 0
green = np.zeros_like( img, np.uint8 )
green[ imask ] = img[ imask ]

#r = cv2.boundingRect( green[ imask ] )

#cv2.imwrite( 'zoomedIn_blue.png', r )
 
#cv2.imshow( green )

cv2.imwrite( "Zoomedin_color.png", green )


# Bitwise-AND mask and original image
#res = cv2.bitwise_and(frame,frame, mask= mask)

#cv2.imshow('frame',frame)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)
#k = cv2.waitKey(5) & 0xFF
#if k == 27:
    #break

#cv2.destroyAllWindows()