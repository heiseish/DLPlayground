from keras.layers import Dense, Input, Bidirectional, LSTM, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import json
from data import n_class, getData, n_letters

MAX_LENGTH = 100
FEATURE_VECTOR_LENGTH = 4096
SAVE_PATH = './model.h5'

X, Y = getData()
X = np.array(X)
y = np.array(Y)
n_a = 128
input_layer = Input(shape=(100, n_letters + 1), name='input_layer')
output_layer = Bidirectional(LSTM(n_a, name='bidirectional_lstm'))(input_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = Dense(128, activation = 'relu', name='dense_1')(output_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = Dense(n_class, activation = 'softmax', name='dense_2')(output_layer)
model = Model(inputs=input_layer, outputs=output_layer)
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
model.fit(X, Y, epochs=300, batch_size=32, callbacks=[checkpoint], shuffle=True)
model.save(SAVE_PATH)