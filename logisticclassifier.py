import numpy as np 
import matplotlib.pyplot as plt 
import h5py

# ----------------------------------------------------------
# ======== Dataset Load ==============

# training dataset
TrainImg = h5py.File("train_catvnoncat.h5","r")

train_x = TrainImg.get('train_set_x')
train_x = np.array(train_x)
# print(train_x.shape)=== (209,64,64,3)

train_y = TrainImg.get('train_set_y')
train_y = np.array(train_y).reshape(-1,1).T

classes = TrainImg.get('list_classes')
clss = np.array(classes) # [b'non-cat' b'cat']

# ========================================================

# testing dataset
TestImg = h5py.File("test_catvnoncat.h5","r")

test_x = TestImg.get("test_set_x")
test_x = np.array(test_x)
# print(test_x.shape) ----- (50,64,64,3)

test_y = TestImg.get('test_set_y')
test_y = np.array(test_y).reshape(-1,1).T

print('=================================================================')

# --------------------------------------------------------------------------------------------------

# figure out dimensions and shapes 
m_train = train_x.shape[0]
m_test = test_x.shape[0]
num_px = train_x.shape[1]
print(f'train_x shape : {train_x.shape}')
print(f'train_y shape : {train_y.shape}')
print(f'test_x shape : {test_x.shape}')
print(f'test_y shape : {test_y.shape}')
print(f'm_train : {m_train}')
print(f'm_test : {m_test}')
print(f'height/widht of pixel : {num_px}')
print(f'Size of each image {train_x.shape[1:]}')

print('=================================================================')

X = train_x.reshape(-1,train_x.shape[0])/255 # training 
x = test_x.reshape(-1,test_x.shape[0])/255 # testing

Y = train_y
y = test_y

f = X.shape[0] # features 
m = X.shape[1] # examples for training

print('X and x after reshape and normalize:')
print(X.shape,x.shape)

print('=================================================================')

def sigmoid(z):
    z = 1/(1+np.exp(-z))
    return z 

def costFunction(y,a):
    c = -(1/m)*np.sum(y * np.log(a) + (1-y) * np.log(1-a))
    return c

# ================= initialize weights and bias ============================
W1 = np.zeros((f,1)) # weights for first layer with 12288*1 shape
b1 = 0 
# ----------------------------------------------------------------------------

# ========== OPTIMIZATION OF PARAMETERS ==================
epochs= 2500  # iterations
costlist = list()
learning_rate = 0.005
parameters = dict()

for i in range(epochs):
    Z1 = W1.T @ X + b1 
    A1 = sigmoid(Z1) # (1,m)

    cost = costFunction(Y,A1)
    costlist.append(cost)
    
    dW1 = (1/m)*X @ (A1 - Y).T # dervative of cost w.r.t weights
    db1 = (1/m)* np.sum(A1-Y,axis=1,keepdims=True)

    W1 = W1 - learning_rate *  dW1
    b1 = b1 - learning_rate * db1

    parameters['W1']=W1
    parameters['b1']=b1

    if i % 100 == 0:
        costlist.append(cost)
        print(f'cost at {i} iteration is {costlist[-1]}')
# --------------------------------------------------------------------------

# ================= PREDICTION ===================
def prediction(m,a):
    pred = np.zeros((1,m))
    for i in range(m):
        pred[:,i] = (a[:,i]>0.5)*1
    return pred


def accuracyfunction(prediction, y,m):
    accuracy = 0.
    for i in range(m):
        if y[:,i]==prediction[:,i]:
            accuracy += 1
    return (accuracy/m)*100



# TRAINING
predY = prediction(m,A1)
training_accuracy = accuracyfunction(predY,Y,m)
print(training_accuracy)

# TESTING 

predy= prediction(50,A1)
test_accuracy = accuracyfunction(predy,y,50)
print(test_accuracy)

# -------------------------------------------------------

