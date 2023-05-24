#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:57:59 2023

@author: nakagawa
"""

import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier


class kNearestNeighbor():
    def __init__(self,k):
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def fit(self,x_train,y_train):
        self.knn.fit(x_train, y_train)
        
    def predict(self,x_test):
        y_predict = self.knn.predict(x_test)
        y_predict = pd.DataFrame(y_predict,index=x_test.index)
        return y_predict
    
    def dump(self,filename="kNearestNeighbor"):
        joblib.dump(self, f"results/model/{filename}.model")