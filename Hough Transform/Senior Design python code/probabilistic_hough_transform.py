# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:45:56 2019

@author: Alexander
"""

import cv2
import numpy as np

img_noblur = cv2.imread('Zoomedin.png', 0) #loading in the picture
img = cv2.GaussianBlur(img_noblur, (7,7),0)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny_edge = cv2.Canny(img_noblur, 25, 0 )

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(canny_edge,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)