import numpy as np 
import h5py

train = h5py.File("train_catvnoncat.h5","r")

train_x = train.get('train_set_x')
train_x = np.array(train_x)
train_y = train.get('train_set_y')
train_y = np.array(train_y)

test = h5py.File("test_catvnoncat.h5",'r')
test_x = test.get('test_set_x')
test_x = np.array(test_x)
test_y = test.get('test_set_y')
test_y = np.array(test_y)

m_train = train_x.shape[0]
m_test = test_x.shape[0]

# ======= reshape ===============
X = train_x.reshape(m_train,-1).T / 255
x = test_x.reshape(-1,m_test)/255

Y = train_y.reshape(1,-1)
y = test_y.reshape(1,-1)



i = 0
def BackwardSigmoid(dA,A):    
    dz = dA * A * np.subtract(1,A)
    return dz


def BackwardRelu(z):
    dz = np.where(z<=0,0,z)
    dz= np.where(dz>0,1,dz)
    return dz 



def sigmoid(z):
    a = 1/(1+np.exp(z))
    cache = z
    return a , cache 

def relu(z):
    a = np.maximum(0,z)
    cache = z
    return a, cache


def InitializeParameters(layer_dims):
    L = len(layer_dims)
    parameters = {}

    np.random.seed(41)

    for l in range(1,L):
        parameters['W'+str(l)] = np.random.rand(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))

    return parameters


nx = X.shape[0]
layer_dims = [nx,10,10,10,10,8,2,1]


# =========================================================================================================
# INITIALIZE PARAMETERS
parameters = InitializeParameters(layer_dims)

def SingleForward(A_prev,W,b):
    ''' This function is forward  for single layer '''
    # print(f'shape of W is {W.shape}')
    # print(f'shape of Al-1 is {A_prev.shape}')
    z = W @ A_prev + b 
    return z 



def UpdateParameters(parameters,grads,learning_rate):
    L = len(parameters) // 2 

    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * grads['db'+str(l+1)]
    return parameters




def Loss(AL,Y):
    m = Y.shape[1]
    c = (-1/m) * np.sum(Y * np.log(AL) + (1-Y)*np.log(1-AL),axis=1,keepdims=True) 
    return c


# --------------------------------------------------------------------------------------------------------

epochs = 500
learning_rate = 0.001

for epoch in range(epochs):
    
    # print(f'epoch is {epoch}')
    caches = [] # it will contain tuple of linear_cache and activation_cache
    L = len(parameters) //2 # total number of layers
    A0 = X

    A_prev = A0
    for l in range(1,L):
        W = parameters['W'+str(l)] # W at l layer
        b = parameters['b'+str(l)] # b at l layer
        Z = SingleForward(A_prev,W,b) # Z at layr
        Al, activation_cache = relu(Z) # A at l layer

        linear_cache = (A_prev,W,b)
        cache = (linear_cache,activation_cache)
        caches.append(cache)
        A_prev = Al

    # in above loop at last loop that is at L-1 iteration A_prev is A at L-1 but not included     
    # for output layer
    WL = parameters['W'+str(L)] 
    bL = parameters['b'+str(L)]

    linear_cache = (A_prev,WL,bL)
    cache = (linear_cache,activation_cache)
    caches.append(cache)

    ZL = SingleForward(A_prev,WL,bL)
    AL, activation_cache = sigmoid(ZL)


    # ============ BACKWARD PROPAGATION ==================

    grads = {}
    dAL = -1 * (np.divide(Y,AL) + np.divide((1-Y) , (1-AL)))

    grads['dA'+str(L)] = dAL

    current_cache = caches[L-1]
    WL = current_cache[0][1]
    bL = current_cache[0][2]
    A_prev = current_cache[0][0] # A at L-1

    dZL = BackwardSigmoid(grads['dA'+str(L)],AL)
    grads['dW'+str(L)] = (1/m_train) * (dZL @ A_prev.T)
    grads['db'+str(L)] = (1/m_train) * np.sum(dZL,axis=1,keepdims=True)
    grads['dA'+str(L-1)] = WL.T @ dZL
    # print(grads['dW'+str(L)].shape)

    for l in reversed(range(2,L)):
        cache = caches[l-1]
        Wl = cache[0][1]
        bl = cache[0][2]
        A_prev = cache[0][0]
        Z = cache[1]

        dZl = BackwardRelu(Z)
        grads['dW'+str(l)] =  (1/m_train) * (dZl @ A_prev.T)
        grads['db'+str(l)] = (1/m_train) * np.sum(dZl,axis=1,keepdims=True)
        grads['dA'+str(l-1)] = Wl.T @ dZl



    current_cache = caches[0]
    W1 = current_cache[0][1]
    b1 = current_cache[0][2]
    A0 = current_cache[0][0] # A at L-1
    Z1 = current_cache[1]
    dZ1 = BackwardRelu(Z1)
    grads['dW'+str(1)] = (1/m_train) * (dZ1 @ A0.T)
    grads['db'+str(1)] = (1/m_train) * np.sum(dZ1,axis=1,keepdims=True)


    
    param = UpdateParameters(parameters,grads,learning_rate)
    cost = Loss(AL,Y)
    
    if epoch%100 ==0:
        print(cost)


# ================= PREDICTION =====================

updated_parameters = param
# print(updated_parameters)