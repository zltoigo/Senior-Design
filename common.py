# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:39:19 2019

@author: Alexander
"""

__all__ = (
    'DIGITS',
    'LETTERS',
    'CHARS',
    'sigmoid',
    'softmax',
)

import numpy, os


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STREETS = ["AVE","BLVD","CIR","CT", "DRIVE","LN", "RD","SQ","ST"]
CHARS = LETTERS + DIGITS

fontDirectory = ".\\data\\fonts"
backgroundDirectory = ".\\data\\bgs"
imageDirectory = ".\data\images"
fnCorpus = '.\data\corpus.txt'
fnWords = '.\data\words.txt'
fnCharList = '.\model\charList.txt'
fnStreetList = '.\\data\\allstreets.txt'
fnInfer = ".\\data\\test.png"
fnAccuracy = '.\\model\\accuracy.txt'




def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()