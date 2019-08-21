import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM , TimeDistributed , Dense

df = pd.read_csv(r'data/SolarPrediction_update.csv')
del df['year']

#print(df.info()
#print(df.corr())

y = np.asarray(df['Radiation'])
del df['Radiation']
x = np.asarray(df)
s = df.columns.values.tolist()

y = (y - min(y))/(max(y) - min(y))

#Random Forest Classifier implemented. Training accuracy of 98.74% and testing accuracy of 93.22% achieved on data using default parameters. Test size used was 0.15 times the training data

X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.15 ,  random_state = 0)
reg = RandomForestRegressor()
reg.fit(X_train , y_train)
print(reg.score(X_train , y_train))
print(reg.score(X_test , y_test))
print(list(zip(s,reg.feature_importances_)))
