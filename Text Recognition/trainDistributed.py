from __future__ import division
from __future__ import print_function

#python 3 libraries
import sys, os
import argparse
import time, datetime

#3rd party libraries
import cv2
import editdistance
import  numpy as np
import tensorflow as tf

#my classes
import common
from DataLoader import DataLoader, Batch
from model import Model, DecoderType, Mode
from Preprocessor import preprocess

FLAGS = None
		

def getPlaceholderIterator(batchSize):
	placeholderImages = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
	placeholderGtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]), 
						tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
	dataset = tf.data.Dataset.from_tensor_slices((placeholderImages, placeholderGtTexts))
	dataset = dataset.batch(batchSize)
	iterator = dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()
	return (features, labels)

def addArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--validate", help="validate the NN", action="store_true")
	return parser.parse_args()

def main():
	args = addArgs()
	startTime = 0
	# features, labels = getPlaceholderIterator(50)
	model = Model(mode = Mode.Train)
	
	if args.train:
		startTime = time.time()
		model.train_and_evaluate(numEpochs=5)

	elif args.validate:
		startTime = time.time()
		model.validate()

	elapsedTime = time.time() - startTime
	print('{:02f}:{:02f}:{:02f}'.format(elapsedTime // 3600, (elapsedTime % 3600 // 60), elapsedTime % 60))
	model.saveLogs()
			
if __name__ == '__main__':
	main()