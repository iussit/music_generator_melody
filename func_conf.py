import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),0]
		dataX.append(a)
		dataY.append(dataset[i + look_back,0])
	return np.array(dataX), np.array(dataY)

def create_model(look_back = 1, layers = 3):
	model = Sequential()
	model.add(LSTM(256, input_shape=(1, look_back), return_sequences=True))
	for layer in range(layers):
		model.add(LSTM(256, return_sequences=True))
	# model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model


# model = Sequential()
#     model.add(LSTM(24, input_shape=(1, look_back)))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
#     print('fit')