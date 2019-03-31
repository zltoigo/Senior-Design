# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:45:00 2019

@author: Alexander
"""

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

#my files
import common
from DataLoader import DataLoader, Batch
from model import Model, DecoderType
from Preprocessor import preprocess

def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 3 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			accFile = open(common.fnAccuracy, 'w+')
			accFile.write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
			accFile.close()
		else:
			print('Character error rate not improved')
			noImprovementSince += 1
			print("Number of epochs without improvement: %d" %noImprovementSince)

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader, outputFile = None):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
			
			if(outputFile):
    				outputFile.write("{},{},{},{}\n".format(dist, len(batch.gtTexts[i]), batch.gtTexts[i], recognized[i]))

	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate


def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = cv2.imread(fnImg,cv2.IMREAD_GRAYSCALE)
	cv2.imshow("infer image", img)
	img = preprocess(img, Model.imgSize)

	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])


def main():
	"main function"
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--validate", help="validate the NN", action="store_true")
	parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
	parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
	args = parser.parse_args()


	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	# train or validate on dataset	
	if args.train or args.validate:
		# load training data, create TF model
		loader = DataLoader('data/images', Model.batchSize, Model.imgSize, Model.maxTextLen)
		# save characters of model for inference mode
		open(common.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open(common.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)

		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=True)
			if(not os.path.exists(".\\model\\validations")):
    				os.mkdir(".\\model\\validations")
			currentDT = datetime.datetime.now()
			formattedDT = currentDT.strftime("%Y_%m_%d %H_%M")
			validateFn = ".\\model\\validations\\{}.csv".format(formattedDT)
			validateFile =  open(validateFn, "w+")
			validateFile.write("Edit Distance, Num Characters, Ground Truth, Recognized\n")
			startTime = time.time()
			validate(model, loader, validateFile)
			validateFile.close()

	# infer text on test image
	else:
		print(open(common.fnAccuracy).read())
		model = Model(open(common.fnCharList).read(), decoderType, mustRestore=True)
		startTime = time.time()
		infer(model, common.fnInfer)
	
	#cleanup
	elapsedTime = time.time() - startTime
	print('{:02f}:{:02f}:{:02f}'.format(elapsedTime // 3600, (elapsedTime % 3600 // 60), elapsedTime % 60))
	model.saveLogs()


if __name__ == '__main__':
	main()