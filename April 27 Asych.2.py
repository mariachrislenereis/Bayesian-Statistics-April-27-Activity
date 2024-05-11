# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:19:24 2024

@author: mcreis
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

# sample x
np.random.seed(2022)
x = np.random.rand(100)*30

# set parameters
a = 3
b = 20
sigma = 5


# obtain response and add noise
y = a*x+b
noise = np.random.randn(100)*sigma

# create a matrix containing the predictor in the first column
# and the response in the second
data = np.vstack((x,y)).T + noise.reshape(-1,1)

# plot data 
plt.scatter(data[:,0], data[:,1])
plt.xlabel("x")
plt.ylabel("y")
