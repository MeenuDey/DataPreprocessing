# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:19:40 2018

@author: Meenu Dey

"""
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('PolyData.csv')
X1=dataset.iloc[:, 1:2].values
X= pd.DataFrame(X1)
df = dataset.iloc[:,2].values   
Y = pd.DataFrame(df)
"""
#tacking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[: ,0:4])
X[:, 0:4] = imputer.transform(X[:, 0:4])
 


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
X_test = sc_X.transform(X_test)"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Ploynmial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg  = PolynomialFeatures(degree =6)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)
  
# Visualising the Linear Regression results
plt.scatter(X, Y , color = 'red')
plt.plot(X, lin_reg.predict(X) , color = 'blue')
plt.title('Magnetic Direction')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Visualising the polynomial Rgression results
plt.scatter(X, Y , color = 'red')
plt.plot(X,lin_reg_2.predict( poly_reg.fit_transform(X)) , color = 'blue')
plt.title('Magnetic Direction')
plt.xlabel('Magnetometer_X-axis')
plt.ylabel('Magnetometer_Y-axis')
plt.show()

