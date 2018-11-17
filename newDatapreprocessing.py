# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:16:07 2018

@author: iiit
"""

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('AllData.csv')
X = dataset.iloc[:, :-1].values
df = dataset.iloc[:, 4].values   
Y = pd.DataFrame(df)
#tacking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', startegy = 'mean', axis = 0)

 


#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
Y.values[:, 0] = labelencoder_Y.fit_transform(Y.values[:, 0])
onehotencoder = OneHotEncoder(categorical_features =[0])
Y = onehotencoder.fit_transform(Y).toarray()

#Splitting the dataset into the Training srt and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)