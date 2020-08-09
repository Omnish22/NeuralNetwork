# train a neural network to draw a curve. The curve takes one input variable,
#  the amount travelled along the curve from 0 to 1, and returns 2 outputs, the 
#  2D coordinates of the position of points on the curve.

import numpy as np 
# ========================================================

# FEED FORWARD NETWORK
a0= np.array([1]).reshape(-1,1)  # this will be input
sigma = lambda z :1/(1 + np.exp(-z))
print(list(map(sigma, [1,2,3])))