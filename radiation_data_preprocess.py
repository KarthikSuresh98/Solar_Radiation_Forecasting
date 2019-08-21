import pandas as pd
import numpy as np


df = pd.read_csv(r'data/SolarPrediction.csv')
new = df['TimeSunRise'].str.split(':' , expand = True)
new1 = df['TimeSunSet'].str.split(':' , expand = True)

new[0] = pd.to_numeric(new[0])
new[1] = pd.to_numeric(new[1])
new[2] = pd.to_numeric(new[2])

new1[0] = pd.to_numeric(new1[0])
new1[1] = pd.to_numeric(new1[1])
new1[2] = pd.to_numeric(new1[2])

df['Difference_rise_to_set'] = (new1[0] - new[0]) + (new1[1] - new[1])/60

df.to_csv(r'data/SolarPrediction_update.csv')
