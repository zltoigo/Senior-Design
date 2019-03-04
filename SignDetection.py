import cv2
import numpy as np

# load image
img = cv2.imread('Forbes.png') 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

mask = cv2.inRange(img, lower_blue, upper_blue)

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)

_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
