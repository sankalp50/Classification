# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:23:39 2019

@author: SANKALP
"""

#Iris Dataset
#data preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Iris.csv')
dataset.head()
x=dataset.iloc[:,1:5].values#independent variables
y=dataset.iloc[:,5].values#dependent variable
#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#encoding the independent variable
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)"""

#fitting the training dataset into classifier model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

#predicting the test dataset results
y_pred=classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

