#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:57:59 2023

@author: nakagawa
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


train = pd.read_csv("train.csv",index_col="id")
test = pd.read_csv("test.csv",index_col="id")


target_str = "price_range"
x_train = train.drop(target_str,axis=1)
y_train = train[target_str]


# x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,
#                                                  test_size=0.1,
#                                                  random_state=0,
#                                                  )

clf = MLPClassifier(solver="adam",
                    hidden_layer_sizes=((100,100,100)),
                    random_state=0,
                    max_iter=1000)
clf.fit(x_train, y_train)
y_predict = clf.predict(test)
y_predict = pd.DataFrame(y_predict,index=test.index)
y_predict.to_csv("submits/neuralnetworkpredict.csv",header=False)