# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 01:29:51 2020

@author: Shraddha
"""

#image classification using logistic regression

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

r = train.iloc[3,1:].values
r = r.reshape(28,28).astype('uint8')
plt.imshow(r)

x_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
x_test, y_test = (test.iloc[:, 1:]), test.iloc[:, 0]


model = LogisticRegression(C = 1.0, solver = 'lbfgs', multi_class = 'ovr', intercept_scaling = 1)

model.fit(x_train, y_train)

predy = model.predict(x_test)

print(predy)

mse = mean_squared_error(y_test, predy)
rmse = np.sqrt(mse)
print(rmse)