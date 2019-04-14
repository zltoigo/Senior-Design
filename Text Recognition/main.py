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
from model import Model, DecoderType
from Preprocessor import preprocess

FLAGS = None
		

def train(model, loader):
	
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured

	trainingIterator = loader.trainingIterator
	validationIterator = loader.validationIterator
	nextBatch = trainingIterator.get_next()

	with model.sess as sess:

		for epochs in range(model.numEpochs):
			#train
			epoch += 1
			print("Epoch: %d" %(epoch))
			sess.run(trainingIterator.initializer)
			while True:
				try:
					(_, loss) = sess.run(nextBatch, feed_dict={handle: trainingHandle})
					print(loss)
				except tf.errors.OutOfRangeError:
					break #epoch finished
		
			#validate
			sess.run(validationIterator.initializer)
			nextBatch = validationIterator.get_next()
			evalRes = sess.run(nextBatch, feedDict={handle:validationHandle})
			decoded = evalRes[0]
			recognized = model.decoderOutputToText(decoded, model.batchSize)

			#calculate characte rerror rate
			nextBatch
			#update saved model if the error rate is better
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


	# while True:
	# 	epoch += 1
	# 	print('Epoch:', epoch)

	# 	# train
	# 	print('Train NN')
	# 	loader.trainSet()
	# 	while loader.hasNext():
	# 		iterInfo = loader.getIteratorInfo()
	# 		batch = loader.getNext()
	# 		loss = model.trainBatch(batch)
	# 		print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

	# 	# validate
	# 	charErrorRate = validate(model, loader)
		
	# 	# if best validation accuracy so far, save model parameters
	# 	if charErrorRate < bestCharErrorRate:
	# 		print('Character error rate improved, save model')
	# 		bestCharErrorRate = charErrorRate
	# 		noImprovementSince = 0
	# 		model.save()
	# 		accFile = open(common.fnAccuracy, 'w+')
	# 		accFile.write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
	# 		accFile.close()
	# 	else:
	# 		print('Character error rate not improved')
	# 		noImprovementSince += 1
	# 		print("Number of epochs without improvement: %d" %noImprovementSince)

	# 	# stop training if no more improvement in the last x epochs
	# 	if noImprovementSince >= earlyStopping:
	# 		print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
	# 		break

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

def createArgParser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--validate", help="validate the NN", action="store_true")
	parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
	parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
	args = parser.parse_args()
	return args

def main():
	"main function"
	# optional command line args
	args = createArgParser()

	print("Initializing Tensorflow")
	print('Python: '+sys.version)
	print('Tensorflow: '+tf.__version__)

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
			
			startTime = time.time()
			validate(model, loader, validateFile)
			validateFile.close()

	# defaults to infer text on test image
	else:
		model = Model(open(common.fnCharList).read(), decoderType, mustRestore=True)
		startTime = time.time()
		infer(model, common.fnInfer)
	
	#cleanup
	elapsedTime = time.time() - startTime
	print('{:02f}:{:02f}:{:02f}'.format(elapsedTime // 3600, (elapsedTime % 3600 // 60), elapsedTime % 60))
	model.saveLogs()

def main():
    args = createArgParser()

	with tf.session() as sess:
		model = Model(mode = args.mode, sess)
		model.buildModel()
		if(args.train or args.validate):
    			model.prepareDataset()
		
		if args.train:
    		model.train()

		else if args.validate:
    		model.validate()

		else:
    		model.infer()
			
if __name__ == '__main__':
	main()