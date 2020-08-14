# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:51:51 2020
Logistic_Regression
@author: Santosh
"""

#==========Loading Data========#
# https://www.kaggle.com/uciml/pima-indians-diabetes-database

#import pandas
import pandas as pd


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("D:\BeCode\Study Material\Spyder Projects\Classification\Classification\diabetes.csv", header=None, names=col_names)

#============Selecting Feature==========#
#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

#=============Splitting Data==============#

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#============Model Development and Prediction=================#

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
