from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
import tensorflow as tf


def loadImages(filePath, gtText):
	image_string = tf.read_file(filePath)

	# Don't use tf.image.decode_image, or the output shape will be undefined
	#channels=1 -->grayscale
	image = tf.image.decode_png(image_string, channels=1)

	# This will convert to float values in [0, 1]
	image = tf.image.convert_image_dtype(image, tf.float32)
	return image, gtText

def preprocess(img, gtText):
	resizedImage = tf.image.resize_images(img, [128,32])

	# normalize
	normalizedImage = tf.image.per_image_standardization(resizedImage)
	# print("normalized image")
	# print(normalizedImage)
	return normalizedImage, gtText

def dataAugmentation(img, gtText):
	stretch = (random.random() - 0.5) # -0.5 .. +0.5
	shapeArray = img.get_shape().as_list() #get dimensions as ints so they can be mulitplied
	wStretched = int(max(shapeArray[1]*(1+stretch), 1)) # random width, but at least 1
	img = tf.image.resize(img, [wStretched, shapeArray[0]])
	print(img.shape)
	return img, gtText
