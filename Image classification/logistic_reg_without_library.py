# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 00:45:24 2020

@author: Shraddha
"""

import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

def eval_cost(A,Y):
    logprobs = np.multiply(np.log(A),Y)
    logprobs1 = np.multiply(np.log(1-A),(1-Y))
    sum = np.sum(logprobs+logprobs1)
    cost = -sum
    return cost


def propagate(w,b,x,y):
    m = x.shape[1]
    z = np.dot(w.T,x)
    a = sigmoid(z)
    cost = 1/m*eval_cost(a,y)
    dz = a-y
    dw = 1/m*np.dot(x,dz.T)
    db = 1/m*np.sum(z)
    return dw,db,cost

def prediction(w,b,x):
    m = x.shape[1]
    y_pred = np.zeros((1,m))
    w = w.reshape(x.shape[0],1)
    z = np.dot(w.T,x)
    a = sigmoid(z)
    for i in range(a.shape[0]):
        if a[0,i] <= 0.5:
            y_pred[0,i] = 0
        else:
            y_pred[0,i] = 1
    return y_pred

def model(x,y,learning_rate,num_iteration):
    w = np.zeros((x.shape[0],1))
    b = 0
    costs = []
    for i in range(num_iteration):
        dw, db, cost = propagate(w,b,x,y)
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
    return w,b,costs

x = np.array([[0,0,1,1],[1,0,0,0]])
y = np.array([0,0,0,1])
learning_rate = 0.5
num_iteration = 2000
w,b,costs = model(x,y,learning_rate,num_iteration)
    
y_pred = prediction(w,b,x)
print(costs)
print(y_pred)
print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred - y)) * 100))
    