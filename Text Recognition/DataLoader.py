from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
import tensorflow as tf
import common
import Preprocessor


class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts

class DataLoader:
	numEpochs = 5
	# batchSize = 50
	trainingIterator = None
	validationIterator = None
	
	def __init__(self, batchSize, imgSize, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"
		self.currIdx = 0

		#dict with filename: gtText
		samples = {}
		bad_samples = []

		f=open(common.fnWords)
		chars = set()

		for line in f:
				if(not line or  line[0] == "#"):
					continue

				lineSplit = line.strip().split(' ')
				fileName = "{}/{:08d}.png".format(common.imgPath, int(lineSplit[0]))
				gtText = self.truncateLabel(' '.join(lineSplit[1: ]), maxTextLen)
				chars = chars.union(set(list(gtText)))
		
				# check if image is not empty
				if not os.path.getsize(fileName):
					bad_samples.append(lineSplit[0] + '.png')
					continue

				# put sample into list
				samples[fileName] = gtText

		self.charList = sorted(list(chars))
		# for(fileName, gtText) in samples.items():
		# 	samples[fileName] = [self.charList.index(c) for c in gtText]
			
		splitIdx = int(0.80 * len(samples.keys()))
		trainSamples = dict(list(samples.items())[:splitIdx])
		validationSamples = dict(list(samples.items())[splitIdx:])

		# put words into lists
		self.trainWords = list(trainSamples.values())
		self.trainFiles = list(trainSamples.keys())
		self.validationWords = list(validationSamples.values())
		self.validationFiles = list(validationSamples.keys())
		# list of all chars in dataset

		self.traingIterator = self.createTrainingSet(batchSize)
		self.validationIterator = self.createValidationSet(batchSize)

	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text

	def createTrainingSet(self, batchSize):
		trainingSet = tf.data.Dataset.from_tensor_slices((self.trainFiles, self.trainWords))
		trainingSet = trainingSet.shuffle(len(self.trainFiles))
		trainingSet = trainingSet.map(Preprocessor.loadImages, num_parallel_calls=4)
		trainingSet = trainingSet.map(Preprocessor.preprocess, num_parallel_calls = 4)
		# trainingSet = trainingSet.map(Preprocessor.dataAugmentation, num_parallel_calls = 4)
		trainingSet = trainingSet.batch(batchSize)
		trainingSet = trainingSet.prefetch(1)
		return trainingSet

	
	def createValidationSet(self, batchSize):
		validationSet = tf.data.Dataset.from_tensor_slices((self.validationFiles, self.validationWords))
		validationSet = validationSet.shuffle(len(self.validationFiles))
		validationSet = validationSet.map(Preprocessor.loadImages, num_parallel_calls=4)
		validationSet = validationSet.map(Preprocessor.preprocess, num_parallel_calls = 4)
		# validationSet = validationSet.map(Preprocessor.dataAugmentation, num_parallel_calls = 4)
		validationSet = validationSet.batch(batchSize)
		validationSet = validationSet.prefetch(1)
		return validationSet

	
		

