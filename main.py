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
	#parser.add_argument("--train", help="train the NN", action="store_true")
	args = parser.parse_args()
    
    #initialize model
    print(open(common.fnAccuracy).read())
    model = Model(open(common.fnCharList).read(), DecoderType.BestPath, mustRestore=True)

    #respond to incoming image
    startTime = time.time()
    infer(model, common.fnInfer)


	#cleanup
    elapsedTime = time.time() - startTime
    print('{:02f}:{:02f}:{:02f}'.format(elapsedTime // 3600, (elapsedTime % 3600 // 60), elapsedTime % 60))
    model.saveLogs()

if __name__ == '__main__':
    main()