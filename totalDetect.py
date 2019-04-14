# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:50:07 2019

@author: Alexander
"""

from imageai.Detection import ObjectDetection
import os
import cv2
import numpy as np
import imutils

execution_path = os.getcwd()

FILENAME = input("\nName the file (without .png): " )

file = FILENAME + '.png' 

print('\n')

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


custom_objects = detector.CustomObjects(traffic_light = True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , file ), output_image_path=os.path.join(execution_path ,FILENAME + 'custom.png'), minimum_percentage_probability=30)

#print(detections)

length = len(detections)

print (length)
 
tuple1 = detections[length - 1]

for key, value in tuple1.items():
    #print( key, value )
    if key == 'box_points' :
        coordinates1 = value 
        #print( coordinates )
        
yMin = coordinates1[1]-25
yMax = coordinates1[3]+25
xMin = coordinates1[0]

image = cv2.imread( FILENAME + 'custom.png' )
height, width, _ = image.shape
cropped1 = image[yMin:yMax, xMin:width ]
cv2.imshow( 'traffic lights', cropped1 )
cv2.waitKey(0)

#############################################################
def resizeToHeight(newHeight, oldImage):

    """Maintains aspect ratio when resizing to specified height"""

    (h,w) = oldImage.shape[:2]

    r = newHeight / float(h)

    dim = (int(w * r), newHeight)

    resized = cv2.resize(oldImage, dim, interpolation=cv2.INTER_AREA)

    return resized

 

def transform(pos):

    # This function is used to find the corners of the object and the dimensions of the object

    pts=[]

    n=len(pos)

    for i in range(n):

        pts.append(list(pos[i][0]))

      

    sums={}

    diffs={}

    tl=tr=bl=br=0

    for i in pts:

        x=i[0]

        y=i[1]

        sum=x+y

        diff=y-x

        sums[sum]=i

        diffs[diff]=i

    sums=sorted(sums.items())

    diffs=sorted(diffs.items())

    n=len(sums)

    rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]

    #      top-left   top-right   bottom-left   bottom-right

  

    h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)     #height of left side

    h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)     #height of right side

    h=max(h1,h2)

  

    w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)     #width of upper side

    w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)     #width of lower side

    w=max(w1,w2)

  

    return int(w),int(h),rect

 

# load image

img = cv2.imread(FILENAME + 'custom.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

 

# define range of blue color in HSV

lower_blue = np.array([110,50,50])

upper_blue = np.array([130,255,255])

 

mask = cv2.inRange(img, lower_blue, upper_blue)

 

# Threshold the HSV image to get only blue colors

mask = cv2.inRange(hsv, lower_blue, upper_blue)

 

# Bitwise-AND mask and original image

res = cv2.bitwise_and(img,img, mask= mask)

 

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 

# for cnt in contours:

#   cv2.drawContours(mask, [cnt], -1, (0, 255, 0), 3)   

#   cv2.imshow("mask", mask)

#   cv2.waitKey(0)

contourLenghts = list(map(lambda x: len(x), contours))

maxIndex = contourLenghts.index(max(contourLenghts))

M = cv2.moments(contours[maxIndex])

x,y,w,h = cv2.boundingRect(contours[maxIndex])

 

#draw bounding box on image

cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

 

#crop out everything but area in bounding box

cropped = img[y:y+h, x:x+w]

 

cv2.imshow("", cropped)

cv2.waitKey(0)

 

#convert to grayscale

gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

 

gray = cv2.bilateralFilter(gray_image, 11, 17, 17)

edged = cv2.Canny(gray, 30, 200)

 

#find squares

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

 

signContour = None

for c in cnts:

    # approximate the contour

    peri = cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, 0.015 * peri, True)


    # rectangles have four points

    if len(approx) == 4:

        signContour = approx

        break

 

cv2.drawContours(cropped, [signContour], -1, (0, 255, 0), 3)

cv2.imshow("", cropped)

cv2.waitKey(0)

 

w,h,arr=transform(signContour)

 

pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])

pts1=np.float32(arr)

M=cv2.getPerspectiveTransform(pts1,pts2)

dst=cv2.warpPerspective(cropped,M,(w,h))

image=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

image = cv2.resize(image,(w,h),interpolation = cv2.INTER_AREA)

cv2.imwrite( "test4.png", image)

cv2.imshow("", image)

cv2.waitKey(0)

 
