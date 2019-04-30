from keras.models import load_model
from data import *
import numpy as np
import json
from encode import *

model = load_model('model.h5');
def predict(s: string):
	X = np.array([embed(s)]);
	y = model.predict(X)
	y_in = np.argmax(y, axis=1)
	y_ = [mapNumberToIntent(z) for z in y_in][0]
	res = {}
	res['confidence'] = y[0][y_in][0]
	res['intent'] = y_
	return res