# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:50:07 2019

@author: Alexander
"""


import tensorflow as tf
from imageai.Detection import ObjectDetection
import os
import logging
import cv2
import numpy as np
import imutils
import pytesseract
from PIL import Image
import time
import glob 
import editdistance
import csv

from PIL import Image
RESIZED_HEIGHT = 32


def initDetector():
    print("Initializing Model")
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(os.getcwd() , "resnet50_coco_best_v2.0.1.h5"))
    print("Loading Model")
    detector.loadModel()
    return detector

def getTrafficLightBBox(detector, inputImageName):
    custom_objects = detector.CustomObjects(traffic_light = True)
    print("Looking for Traffic Lights")
    start_time = time.time()
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, 
                                                        input_image=os.path.join(os.getcwd(),inputImageName),
                                                        output_image_path=os.path.join(os.getcwd(), "output.png"), 
                                                        minimum_percentage_probability=30)
    elapsed_time = time.time() - start_time
    boundingBoxes = list()
    if detections: 
        boundingBoxes = list([x['box_points'] for x in detections])
        sortedBoxes = sorted(boundingBoxes, key = lambda x: x[1])
        topBoundingBox = sortedBoxes[0] 
        return topBoundingBox, elapsed_time
    return None, elapsed_time

def cropAroundTopLight(image, topBoundingBox):
    #the top right corner is max(max(x)-x+y),where max(x) is the max of all x coordinates
    # maxX = max(map(lambda x: x[0], boundingBoxes))
    #sort by y1 in ascendingOrder
    # sortedBoxes = sorted(boundingBoxes, key = lambda coordinates:maxX-coordinates[0]+coordinates[1], reverse = True)
    
    #  The internal Detectron box format is 
    #  [x1, y1, x2, y2] where (x1, y1) specify the top-left box corner and (x2, y2) 
    #  specify the bottom-right box corner.
    # for (x1, y1, x2, y2) in boundingBoxes:
    #     cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
    x1,y1,x2,y2 = topBoundingBox

    w,h,_ = image.shape 
    buffer = y2 - y1 #height of traffic light
    yMin = max(topBoundingBox[1]-buffer, 0)
    yMax = min(topBoundingBox[3] + buffer, x1*h   )
    xMin = topBoundingBox[0]
    cropped1 = image[yMin:yMax, xMin: ]
    return cropped1

def resizeToHeight(newHeight, oldImage):
    """Maintains aspect ratio when resizing to specified height"""
    (h,w) = oldImage.shape[:2]
    r = newHeight / float(h)
    dim = (int(w * r), newHeight)
    resized = cv2.resize(oldImage, dim, interpolation=cv2.INTER_AREA)
    return resized

def resizeToWidth(newWidth, oldImage):
    """Maintains aspect ratio when resizing to specified height"""
    (h,w) = oldImage.shape[:2]
    r = newWidth / float(w)
    dim = (newWidth, int(h*r))
    resized = cv2.resize(oldImage, dim, interpolation=cv2.INTER_AREA)
    return resized

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (int(maxWidth*0.95), int(maxHeight*0.9)))

	# return the warped image
	return warped

def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 
def thresholdBlue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([140,255,255])   
    mask = cv2.inRange(img, lower_blue, upper_blue)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    return res, mask

def findRectangle(image, trafficLightBBox):
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    showImage("denoised", image)

    #make a copy
    flat_object_resized_copy = image.copy()
    # flat_object_resized_copy = cv2.convertScaleAbs(flat_object_resized_copy, alpha = 1.1, beta = 10)

    # flat_object_resized_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #convert to HSV color scheme
    flat_object_resized_hsv = cv2.cvtColor(flat_object_resized_copy, cv2.COLOR_BGR2HSV)
    # split HSV to three chanels
    hue, saturation, value = cv2.split(flat_object_resized_hsv)
    # threshold to find the contour by looking for sharp changes in colo
    #gets rid of blue specks on black background
    retval, satThresholded = cv2.threshold(saturation, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  
    # morphological operations - clean up the mask
    thresholded_open = cv2.morphologyEx(satThresholded, cv2.MORPH_OPEN, (7,7))
    thresholded_close = cv2.morphologyEx(thresholded_open, cv2.MORPH_CLOSE, (7,7))
    # showImage("mask", cv2.bitwise_not(thresholded_close))
    noBackgroundSpecks = cv2.bitwise_and(flat_object_resized_copy, flat_object_resized_copy, mask = cv2.bitwise_not(thresholded_close))
    showImage("no background specks", noBackgroundSpecks)
    
    #detect changes in blues
    noBackgroundSpecks = cv2.convertScaleAbs(noBackgroundSpecks, alpha = 1.3, beta = 40)
    hue, saturation, value = cv2.split(noBackgroundSpecks)
    retval, thresholded = cv2.threshold(hue,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresholded_close = cv2.morphologyEx(cv2.bitwise_not(thresholded), cv2.MORPH_CLOSE, (7,7))
    showImage("hue contour", thresholded_close)

    cnts = cv2.findContours(thresholded_close.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    largestRect = None
    largestArea = 0
    x1,y1,x2,y2 = trafficLightBBox
    lightH = y2-y1
    for c in cnts:
        rect = cv2.minAreaRect(c)
        (centerPoint, (width, height), rotationAngle) = rect
        area = width*height
        #one largest rectangle (that's not a square)
        if area > largestArea and height < lightH:
            largestRect = rect
            largestArea = area
        if cv2.contourArea(c) > 10:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(flat_object_resized_copy,[box],0,(0,0,255),3)
            showImage("contour attempt", flat_object_resized_copy)
    if largestRect is not None:
        maxBox = cv2.boxPoints(largestRect)
        maxBox = np.int0(maxBox)
        cv2.drawContours(flat_object_resized_copy,[maxBox],0,(0,0,255),2)
        showImage("found sign", flat_object_resized_copy)
    return largestRect

def improveContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # showImage('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    # showImage('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # showImage('final', final)

def getSignFromImage(detector, filename):
    start_time = time.time()
    image = cv2.imread(filename)
    image = resizeToWidth(1024, image)
    cv2.imwrite(filename, image)
    cv2.imshow( "resized image", image )
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    #returns list of arrays with format [x1, y1, x2, y2]

    trafficLight, elapsed_time_light = getTrafficLightBBox(detector, filename)
    thresholded = None
    if trafficLight is not None: 
        print("did not find traffic lights")
        thresholded, mask = thresholdBlue(image)
        cv2.imshow( "only Blue", thresholded )
        cv2.waitKey(0)

    generalSignArea = cropAroundTopLight(image, trafficLight)
    # improveContrast(generalSignArea)
    showImage("traffic lights", generalSignArea)
    thresholded, mask = thresholdBlue(generalSignArea)
    cv2.imshow( "only Blue", thresholded )
    cv2.waitKey(0)
    
    largestRectangle = findRectangle(thresholded, trafficLight)
    if largestRectangle is not None:
        # get width and height of the detected rectangle
        box = cv2.boxPoints(largestRectangle)
        # directly warp the rotated rectangle to get the straightened rectangle
        flattened = four_point_transform(generalSignArea, box)
        #help out under-exposed images
        # flattened = cv2.convertScaleAbs(flattened, alpha = 1.5, beta = 0)
        showImage("flattned", flattened)
        #white text on black background
        grayImage = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)
        showImage("gray", grayImage)
        w,h = grayImage.shape
        if h < 300:
            grayImage = cv2.resize(grayImage, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        showImage("resized gray", grayImage)
        (_, blackAndWhite) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # showImage("denoised resized", denoisedResized)
        showImage("black and white", blackAndWhite)
        inverted = cv2.bitwise_not(blackAndWhite)
        kernel = np.ones((1, 1), np.uint8)
        inverted = cv2.dilate(inverted, kernel, iterations=1)
        inverted = cv2.erode(inverted, kernel, iterations=1)
        showImage("inverted", inverted)
    else: 
        print("contours not found on thresholded hue")
        inverted = None
    elasped_time_sign = time.time() - start_time


    return inverted, elasped_time_sign, elapsed_time_light

execution_path = os.getcwd()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
tf.get_logger().setLevel(logging.ERROR)
 
##################################################
#               MAIN CODE                         #
###################################################

file = input("Enter file name with.png: ")
detector = initDetector()

sign, _, _ = getSignFromImage(detector, file)
if sign is not None:
    
    text = pytesseract.image_to_string(Image.fromarray(sign))  
    print(text)
    rows = text.splitlines()
    if len(rows) > 0:
        text = rows[len(rows)-1]
    print("recognized text: " + text)
else:
    print("no sign found")








