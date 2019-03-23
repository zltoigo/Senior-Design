# -*- coding: utf-8 -*-
import cv2
import numpy as np

#height of the final image containing detected words
RESIZED_HEIGHT = 72


def resizeToHeight(newHeight, oldImage):
    """Maintains aspect ratio when resizing to specified height"""
    (h,w) = oldImage.shape[:2]
    r = newHeight / float(h)
    dim = (int(w * r), newHeight)
    resized = cv2.resize(oldImage, dim, interpolation=cv2.INTER_AREA)
    return resized

img = cv2.imread('Forbes3.png') 
width = 640
height = 480
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow( 'Forbes', img )
cv2.waitKey(0)
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
contourLenghts = list(map(lambda x: len(x), contours))
maxIndex = contourLenghts.index(max(contourLenghts))
M = cv2.moments(contours[maxIndex])
x,y,w,h = cv2.boundingRect(contours[maxIndex])

#draw bounding box on image
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#crop out everything but area in bounding box
cropped = img[y:y+h, x:x+w]

#convert to grayscale
gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

#white text on black background
(thresh, blackWhite) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#black text on white background
inverted = cv2.bitwise_not(blackWhite)

# display
cv2.imshow("Inverted", inverted) 
cv2.imwrite("test.png", inverted)

cv2.waitKey(0)

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
    if height >= image.shape[0]* 0.4:
        if width >= image.shape[1]*0.2 and width <= image.shape[1]*0.9:
             roi = resizeToHeight(RESIZED_HEIGHT, roi)
             streetnameWords.append(roi)
             space = np.zeros((RESIZED_HEIGHT,int(RESIZED_HEIGHT/2.5), 3), np.uint8)
             space.fill(255)
             streetnameWords.append(space)

#remove extra space
streetnameWords.pop()
joinedSegments = np.hstack(streetnameWords)

joinedSegments = cv2.GaussianBlur(joinedSegments,(9,9),0)

cv2.imshow( 'marked areas', joinedSegments )
cv2.imwrite('test.png', joinedSegments)
cv2.waitKey(0)