# SIMPLE NEURON

# this is simple neural network with one input and one output neuron

import numpy as np 
sigma = np.tanh # activation function
w1 = -10 # weight
b1 = 10 # biase

def activation(a0):
    ''' THIS IS NOT FUNCTION WHICH CONVERT 1 TO 0 AND 0 TO 1
        USING tanh '''
    return sigma(w1*a0 + b1)

# giving input
print(activation(1)) # output : 0
print(activation(0)) # output : 0.9999999958776927
