from __future__ import division
from __future__ import print_function

import sys, os, time, datetime

import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
import editdistance


import common
from DataLoader import DataLoader

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2

class Mode:
	Train = 0
	Validate = 1
	Predict = 2

class Model:
	"minimalistic TF model for HTR"

	# model constants
	imgSize = (128, 32)
	maxTextLen = 32
	numEpochs = 5
	batchSize = 50

	def __init__(self, mode):
		"init model: add CNN, RNN and CTC and initialize TF"
		self.decoderType = DecoderType.BestPath
		self.snapID = 0

		# Whether to use normalization over a batch or a population
		self.is_train = tf.placeholder(tf.bool, name="is_train")
		# self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
		# self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
		
		
		if mode == Mode.Train or mode == Mode.Validate:
			#load dataset
			self.dataLoader = DataLoader(self.batchSize, self.imgSize, self.maxTextLen)
			trainingDataset = self.dataLoader.createTrainingSet(self.batchSize)
		
		validationDataset = self.dataLoader.createValidationSet(self.batchSize)
		iterator=tf.data.Iterator.from_structure(validationDataset.output_types, 
										 validationDataset.output_shapes)
		images, gtTexts = iterator.get_next()
	

		#trainingINitOp and validationInit op do not require additional initialization
		self.trainingInitOp = iterator.make_initializer(trainingDataset)
		self.validationInitOp = iterator.make_initializer(validationDataset)

		self.buildModel(mode, images, gtTexts)
		(self.sess, self.saver) = self.setupSaver()
		self.writer = tf.summary.FileWriter(".\model\logs\log", self.sess.graph) 

	def setupCNN(self, inputImages):
		"create CNN layers and return output of these layers"
		cnnIn4d = tf.expand_dims(input=inputImages, axis=3)
		cnnIn4d = inputImages
		# print("cnnIn4d")
		# print(cnnIn4d)

		# list of parameters for the layers
		kernelVals = [5, 5, 5, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals 	= poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		# create layers
		pool = cnnIn4d # input to first CNN layer
		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

		return pool


	def setupRNN(self, cnnOut4d):
		"create RNN layers and return output of these layers"
		rnnIn3d = tf.squeeze(cnnOut4d, axis=[2])

		# basic cells which is used to build RNN
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# stack basic cells
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		# BxTxF -> BxTx2H
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)

		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		return tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


	def setupCTC(self, rnnOut3d, gtTexts):
		"create CTC loss and decoder and return them"
		# BxTxC -> TxBxC
		self.ctcIn3dTBC = tf.transpose(rnnOut3d, [1, 0, 2])
		# calc loss for batch
		self.seqLen = tf.placeholder(tf.int32, [None])
		# self.seqLen = [Model.maxTextLen] * self.batchSize	
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		# calc loss for each element to compute label probability
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

			# prepare information about language (dictionary, characters in dataset, characters forming words)
			chars = str().join(self.charList)
			wordChars = open('./model/wordCharList.txt').read().splitlines()[0]
			corpus = open('/data/corpus.txt').read()

			# decode using the "Words" mode of word beam search
			self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

	def toSparse(self, textTensor):
		table = tf.contrib.lookup.index_table_from_tensor(mapping = tf.constant(self.dataLoader.charList),
															num_oov_buckets=1)
		encoded = table.lookup(textTensor)
		encoded = tf.cast(encoded, tf.int32)
		indices = tf.where(tf.not_equal(encoded, 0))
		values = tf.gather_nd(encoded, indices)
		shape = tf.shape(encoded, out_type = tf.int64)
		return tf.SparseTensor(indices, values, shape)
	# 	int32 required for ctc loss
	# 	return tf.cast(sparse, tf.int32) 

	def buildModel(self, mode, inputImages, gtTexts):
		print("Building Model")
		#handle char list
		if mode in [Mode.Train, Mode.Validate]:
			# save characters of model for inference mode
			open(common.fnCharList, 'w').write(str().join(self.dataLoader.charList))
			# save words contained in dataset into file
			open(common.fnCorpus, 'w').write(str(' ').join(self.dataLoader.trainWords + self.dataLoader.validationWords))
			self.charList = self.dataLoader.charList
		else:
			self.charList = open(common.fnCharList).read()

		# setup CNN, RNN and CTC
		print("Setting up CNN")
		cnnOut4d = self.setupCNN(inputImages)
		print("Setting up RNN")
		rnnOut3d = self.setupRNN(cnnOut4d)
		print("Setting up CTC")

		gtTexts = self.toSparse(gtTexts)

		# gtTexts = self.toSparse(tf.Session().run(gtTexts))
		self.setupCTC(rnnOut3d, gtTexts)

		# setup optimizer to train NN
		print("setting up Optimizer")
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

	def setupSaver(self):
		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '.\model\snapshots'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?
		sess = tf.Session()
		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run([tf.global_variables_initializer(), tf.initializers.tables_initializer()]	)
		return (sess, saver)

	# def toSparse(self, texts):
	# 	"put ground truth texts into sparse tensor for ctc_loss"
	# 	indices = []
	# 	values = []
	# 	shape = [len(texts), 0] # last entry must be max(labelList[i])

	# 	# go over all texts
	# 	for (batchElement, text) in enumerate(texts):
	# 		# convert to string of label (i.e. class-ids)
	# 		labelStr = [self.charList.index(c) for c in text]
	# 		# sparse tensor must have size of max. label-string
	# 		if len(labelStr) > shape[1]:
	# 			shape[1] = len(labelStr)
	# 		# put each label into sparse tensor
	# 		for (i, label) in enumerate(labelStr):
	# 			indices.append([batchElement, i])
	# 			values.append(label)

	# 	return (indices, values, shape)

	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor
			decoded=ctcOutput[0][0]

			# go over all indices and save mapping: batch -> values
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

	def validate(self, verbose = True, saveToFile = False):
		if(saveToFile):
			if(not os.path.exists(".\\model\\validations")): os.mkdir(".\\model\\validations")
			currentDT = datetime.datetime.now()
			formattedDT = currentDT.strftime("%Y_%m_%d %H_%M")
			validateFn = ".\\model\\validations\\{}.csv".format(formattedDT)
			validateFile =  open(validateFn, "w+")
			validateFile.write("Edit Distance, Num Characters, Ground Truth, Recognized\n")
		numCharErr = 0
		numCharTotal = 0
		numWordOK = 0
		numWordTotal = 0

		#validate
		try:
			while True:
				decoded, _ = self.sess.run([self.decoder, self.ctcIn3dTBC],
											feed_dict = {self.seqLen: [Model.maxTextLen] * self.batchSize, self.is_train:False})
				recognized = decoderOutputToText(decoded, self.batchSizeftrain_a)

				for i in range(len(recognized)):
					numWordOK += 1 if self.gtTexts[i] == recognized[i] else 0
					numWordTotal += 1
					dist = editdistance.eval(recognized[i], self.gtTexts[i])
					numCharErr += dist
					numCharTotal += len(self.gtTexts[i])
					if(verbose):
						print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + self.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

					if(saveToFile):
						validateFile.write("{},{},{},{}\n".format(dist, len(self.gtTexts[i]), self.gtTexts[i], recognized[i]))
		except tf.errors.OutOfRangeError:
			pass

		if(saveToFile): validateFile.close()

		charErrorRate = numCharErr / numCharTotal
		wordAccuracy = numWordOK / numWordTotal
		print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
		return charErrorRate

	def train_and_evaluate(self, numEpochs, verbose = True):
		for epoch_no in range(numEpochs):
			print('\nEpoch No: {}'.format(epoch_no + 1))

			# Initialize iterator with training data
			self.sess.run(self.trainingInitOp)
			# decay learning rate
			rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) 
			trainingDict = {self.seqLen: [self.maxTextLen]*self.batchSize, self.learningRate: rate, self.is_train:True}
			try:
				with tqdm(total = len(self.dataLoader.trainWords)) as pbar:
					while True:
						_, loss = self.sess.run([self.optimizer, self.loss], trainingDict)
						pbar.update(self.batchSize)
						self.batchesTrained+=1
			except tf.errors.OutOfRangeError:
				pass

			# Initialize iterator with validation data
			self.sess.run(self.validationInitOp)
			charErrorRate = self.validate()

			# if best validation accuracy so far, save model parameters
			if charErrorRate < bestCharErrorRate:
				print('Character error rate improved, save model')
				bestCharErrorRate = charErrorRate
				noImprovementSince = 0
				self.save()
				accFile = open(common.fnAccuracy, 'w+')
				accFile.write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
				accFile.close()
			else:
				print('Character error rate not improved')

	def encodeLabel(self, text):
		return [self.charList.index(c) for c in text]

	def save(self,saver):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '.\model\snapshots\snapshot', global_step=self.snapID)

	def saveLogs(self):
		self.writer.close()