# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:31:39 2020

@author: Akhil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = r"C:\Users\Akhil\Downloads\Kaggle-Credit-Card-Fraud-Detection-master\Kaggle-Credit-Card-Fraud-Detection-master\creditcard.csv"

data = pd.read_csv(file_path)
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
X[:,:] = imputer.fit_transform(X[:,:])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =  train_test_split(X,Y,test_size=0.35)

from sklearn.linear_model import LogisticRegression

regression = LogisticRegression()
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test, y_pred))

