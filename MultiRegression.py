# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:47:14 2018

@author: iiit
"""
# Multiple liner regerssion
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('NewWithPowerbank.csv')
X1 = dataset.iloc[:, :-1].values
X = pd.DataFrame(X1)
df = dataset.iloc[:, 7].values   
Y = pd.DataFrame(df)



#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.values[:, 6] = labelencoder_X.fit_transform(X.values[:, 6])
onehotencoder = OneHotEncoder(categorical_features =[6])
X = onehotencoder.fit_transform(X).toarray()


#tacking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X.values[:, 0:8])
X.values[:, 0:8] = imputer.transform(X.values[:, 0:8])


#Splitting the dataset into the Training srt and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""  
  
# Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predicting the Test set results
Y_perd = regressor.predict(X_test)


#Building the optimal model using Backward Elemination
import statsmodels.formula.api as sn
X = py.append(arr = py.ones((5019, 1)).astype(int), values = X, axis = 1)
X_opt = X[: , [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]]
regressor_OLS = sn.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0, 1, 2, 3, 4,6,7,8,9,10,11]]
regressor_OLS = sn.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0, 1, 2, 3,6,7,8,9,10,11]]
regressor_OLS = sn.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0, 1, 2,6,7,8,9,10,11]]
regressor_OLS = sn.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0, 1, 2,6,7,8,10,11]]
regressor_OLS = sn.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0, 1, 2,7,10,11]]
regressor_OLS = sn.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[: , [0, 1, 2,7,11]]
regressor_OLS = sn.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
