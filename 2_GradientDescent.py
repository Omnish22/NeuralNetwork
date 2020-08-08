''' NOTE  this code is for single input and not vector of input because 
    we didn't find sum of suqare of errror we just find error for single 
    input '''
# this code is to find the gradient of  error of neuron
# tanh will be activation function 

import numpy as np 

# define sigma i.e. activation function 
sigma = np.tanh

# 0 layer is input layer
# define feed forward equation or output of first layer
def a1(w1,b1,a0):
    z = w1 * a0 + b1
    return sigma(z)


# error : square of network output - training data output
def C(w1,b1,x,y):
    ''' C is cost function ie.sq error '''
    return (a1(w1,b1,a0)-y)**2

# differentiate cost 'C' with respect to weight 'w1'
def dCdw(w1,b1,x,y):
    ''' this will find differentiation of cost 'C' w.r.t. weight 'w1'
        using chain rule ie first we find differentiation of C w.r.t 
        activation function then differentiation activation w.r.t 'z'
        and then of 'z' w.r.t 'w1' ie weight '''
    z= w1 * x + b1
    dCda = 2 * (a1(w1,b1,x) - y) # diff. of C w.r.t a
    dadz = (1/np.cosh(z))**2 # diff. of activation ie tanh(z) w.r.t 'z'
    dzdw = x # diff. z ie w*x+b w.r.t 'w' which is 'x'

    return dCda * dadz * dzdw # chain rule to get dCdw

def dCdb(w1,b1,x,y):
    ''' this will find differentiation of cost 'C' w.r.t. bias 'b1'
        using chain rule ie first we find differentiation of C w.r.t 
        activation function then differentiation activation w.r.t 'z'
        and then of 'z' w.r.t 'b1' ie bias '''

    z = w1 * x + b1
    dCda = 2 * (a1(w1,b1,x)-y)
    dadz = (1/np.cosh(z))**2
    dzdb = 1 
    return dCda * dadz * dzdb


"""Testing """
# unfit weight and bias.
w1 = 2.3
b1 = -1.2
# test on a single data point pair of x and y.
x = 0
y = 1
# Output how the cost would change
# in proportion to a small change in the bias and weight
print( dCdb(w1, b1, x, y) )
print(dCdw(w1,b1,x,y))