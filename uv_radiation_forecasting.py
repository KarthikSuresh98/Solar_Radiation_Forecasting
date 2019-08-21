import pandas as pd
import numpy as np
import collections
from keras.layers import LSTM , TimeDistributed , Dense , RepeatVector
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.optimizers import Adam
import tensorflow as tf


def data_preprocess():
    df = pd.read_csv(r'data/uv_data.csv')
    x = np.asarray(df['UVA'] , dtype = np.float64)
    x = to_categorical(x , num_classes = 55)
    y = x[1:]
    x = x[0:len(x)-1]


    x_train , x_test , y_train , y_test = train_test_split(x , y , shuffle = False)
    x_train = np.reshape(x_train , (1 , x_train.shape[0] , 55))
    x_test = np.reshape(x_test , (1 , x_test.shape[0] , 55))
    y_train = np.reshape(y_train , (1 , y_train.shape[0] , 55))
    y_test = np.reshape(y_test , (1 , y_test.shape[0] , 55))
    return x_train , x_test , y_train , y_test


x_train , x_test , y_train , y_test = data_preprocess()
n_in = x_train.shape[0]
n_out = y_train.shape[1]

model = Sequential()
model.add(LSTM(500 ,  activation = 'relu' ,  input_shape = (None , 55)))
model.add(RepeatVector(n_out))
model.add(LSTM(500, activation = 'relu' ,  return_sequences = True))
model.add(TimeDistributed(Dense(55 , activation = 'softmax')))
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
model.fit(x_train , y_train , batch_size = 256 , epochs = 100)

#print(model.summary())
print(model.evaluate(x_test , y_test))

model_json = model.to_json()
with open("model_uv_forecasting.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_uv_forecasting.h5")
print("Saved model to disk")
