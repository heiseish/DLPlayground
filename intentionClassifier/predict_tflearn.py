# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
model_path = './saved/model.tfl'
from data import *
from encode import *
# Building convolutional network
network = input_data(shape=[None, 100], name='input')
network = tflearn.embedding(network, input_dim=10000, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 9, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load(model_path)

def predict(input):
    testX = [sentenceToIndex(input)]
    testX = pad_sequences(testX, maxlen=100, value=0.)
    y = model.predict(testX)
    y_ = np.argmax(y, axis=1)
    confidence = y[0][y_]
    y_ = [mapNumberToIntent(z) for z in y_]
    print('Intent: {}'.format(y_))
    print('Confidence level: {}'.format(confidence))
    return {
        'intent': y_[0],
        'confidence': encode(confidence[0])
    }
