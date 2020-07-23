# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 03:04:43 2020

@author: Shraddha
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

x_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
x_test, y_test = (test.iloc[:, 1:]), test.iloc[:, 0]

def sigmoid(z):
    return 1/(1+np.exp(-z))

cost = []
w = np.zeros((x_train.shape[0],1))
w = np.reshape(-1, 1)
b = 0 
m = x_train.shape[1]
dw, db = 0, 0
for i in range(10000):
    z = np.dot(w.T, x_train)+b
    A = sigmoid(z)
    dz = A-y_train
    db = 1/m*np.sum(dz)
    dw = 1/m*np.dot(x_train, dz.T)
    w = w - 0.01*dw
    b = b - 0.01 * db
    