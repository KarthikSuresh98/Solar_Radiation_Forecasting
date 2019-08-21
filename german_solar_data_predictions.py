import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

l = r'data/GermanSolarFarm/data/pv_01.csv'

def func(loc):

    df = pd.read_csv(loc)
    del df['time_idx']
    s = df.columns.values
    st = 'SolarRadiationGlobalAt0'
    l = []
    for i in range(len(s)):
        c = df[st].corr(df[s[i]])
        if abs(c) <= 0.2 or abs(c) >= 0.85:
            l.append(s[i])

    y = np.asarray(list(df[st]) , dtype  = np.float32)
    for i in range(len(l)):
        del df[l[i]]

    x = np.asarray(df , dtype = np.float32)
    y = np.reshape(y , (y.shape[0] , 1))
    return x , y


X, Y = func(l)
x_train , x_test , y_train , y_test = train_test_split(X , Y , shuffle = True)

clf = RandomForestRegressor(n_estimators = 25 , max_depth = 10)
clf.fit(x_train , y_train)
print('The training accuracy is:')
print(clf.score(x_train , y_train))
print('The testing accuracy is:')
print(clf.score(x_test , y_test))


print('\n*****')


X , Y = func(r'data/GermanSolarFarm/data/pv_02.csv')
print(clf.score(X  , Y))
