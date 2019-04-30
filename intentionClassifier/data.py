import unicodedata
import json
import string
import tensorflow as tf
import tflearn
from keras.utils import to_categorical
import glob
import os
from encode import *
import numpy as np
import json

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_class = 9
def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

	# Read a file and split into lines
def readLines(filename):
	'''
	Read lines from txt files.
	Return:
	- Array in ASCII
	'''
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]

def letterToIndex(letter):
	'''
	Convert letter to index. eg a -> 1. 0 is reserved for empty letter (for pad_sequence latter)
	Return
	- Integer representing index.
	'''
	return all_letters.find(letter) + 1

def sentenceToIndex(sentence):
	'''
	Convert sentence to array of letter index
	Return
	- Array of integer
	'''
	return [letterToIndex(c) for c in sentence]


def sentenceToOneHotVectors(sentence):
	'''
	Convert sentence to array of one hot vector
	Return
	- Array of one-hot vectors
	'''
	with tf.Session() as sess:
		arr = tf.one_hot(sentence, n_letters + 1).eval()
	return arr

def mapIntentToNumber(intent):
	if intent == 'greetings':
		return 0
	elif intent == 'thanks':
		return 1
	elif intent == 'bye':
		return 2
	elif intent == 'news':
		return 3
	elif intent == 'weather':
		return 4
	elif intent == 'worldCup':
		return 5
	elif intent == 'pkmGo':
		return 6
	elif intent == 'help':
		return 7
	elif intent == 'compliment':
		return 8

def mapNumberToIntent(n):
	if n == 0:
		return 'greetings'
	elif n == 1:
		return 'thanks'
	elif n == 2:
		return 'bye'
	elif n == 3:
		return 'news'
	elif n == 4:
		return 'weather'
	elif n == 5:
		return 'worldCup'
	elif n == 6:
		return 'pkmGO'
	elif n == 7:
		return 'help'
	elif n == 8:
		return 'compliment'

def embed(X):
	x = X.lower()
	x = sentenceToIndex(x)
	x = sentenceToOneHotVectors(x)
	to_concat = np.zeros((100 - len(x), n_letters + 1))
	res = np.empty((100, n_letters + 1))
	np.concatenate([x, to_concat], out=res)
	return res

def getData():
	X = []
	Y = []
	num_classes = 0
	for filename in findFiles('cases/*.txt'):
		num_classes += 1
		category = os.path.splitext(os.path.basename(filename))[0]
		lines = readLines(filename)
		for line in lines:
			X.append(embed(line))
			Y.append(mapIntentToNumber(category))
	Y = to_categorical(Y)
	return X, Y
