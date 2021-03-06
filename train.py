import func_conf as func

import os
import numpy as np
import scipy.io.wavfile as wav

from keras.callbacks import ModelCheckpoint


look_back = 1000
layers = 3
epochs = 1
verbose = 1



directory = "./datasets/"
data = []
names = []


directory_wave=directory + 'wave/'
for i in os.listdir(directory_wave):
    names.append(i)


for name in names:
    #convert wav to numpy array
    data = wav.read(directory_wave + name)
    data = np.array(data[1],dtype=float)

    train_size = int(len(data) * 0.1)
    data = data[0:train_size]

    # print(data)


    # print(trainX)
    # print(data.shape)


    print(data.shape)
    trainX, trainY = func.create_dataset(data, look_back)
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # trainX = np.reshape(trainX, (trainX.shape[0],  trainX.shape[1], 1))
    # trainY = np.reshape(trainY, (trainX.shape[0], 1, trainY.shape[1]))
    print(trainX)
    print(trainY)
    # print(trainX)
    print(trainX.shape)
    print(trainY.shape)

    model = func.create_model(look_back=look_back,layers=layers)

    #define checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print('fit')
    model.fit(trainX, trainY, epochs=epochs, batch_size=256, verbose=verbose,callbacks=callbacks_list)
    model.save("model.h5")
    model.save_weights('my_model_weights.h5')


