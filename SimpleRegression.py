# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:55:19 2018

@author: iiit
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:16:07 2018

@author: iiit
"""
# simple liner regerssion
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('regression1.csv')
X = dataset.iloc[:, :-1].values
df = dataset.iloc[:, 1].values   
Y = pd.DataFrame(df)
#tacking care of missing data
#tacking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:1])
X[:, 0:1] = imputer.transform(X[:, 0:1])



#Encoding catagorical data


#Splitting the dataset into the Training srt and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""  
  
# fitting simple Linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
 
#Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set Result
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hall Sensor Data vs Magnetometer X-axis Data')
plt.xlabel('Hall Sensor Data')
plt.ylabel('Magnetomete(X-axis Data)')
plt.show() 

# Visualising the testing set result
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hall Sensor Data vs Magnetometer X-axis Data')
plt.xlabel('Hall Sensor Data')
plt.ylabel('Magnetom ete(X-axis Data)')
plt.show() 