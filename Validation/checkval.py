from data import *
import numpy as np
from scipy.special import softmax

def getArray(X):
    X = X.strip().split('\n')
    X = [i.split() for i in X]
    X = np.asarray(X, dtype = 'float')
    return X

def relu(X):
    return np.maximum(0, X)

def forward_prop(X, W1, W2, b1, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2, axis=1)
    return Z1, A1, Z2, A2

def backward_prop(yhat):
    m = X.shape[1]
    ones = np.ones((1,m))
    yhat_diff_y = yhat-y
    W2T_yhat_y = np.dot(W2.T, yhat_diff_y)
    dW1 = (1/m)*np.dot(relu(W2T_yhat_y), X.T)
    dW2 = (1/m)*np.dot(yhat_diff_y, A1.T)
    db1 = (1/m)*np.dot(relu(W2T_yhat_y), ones.T)
    db2 = (1/m)*np.dot(yhat_diff_y, ones.T)
    
    return dW1, dW2, db1, db2
    
def cost(yhat):
    

X = getArray(X)
y = getArray(y)
W1 = getArray(W1)
W2 = getArray(W2)
b1 = getArray(b1)
b2 = getArray(b2)
alpha = 0.01


for i in range(10):
    Z1, A1, Z2, A2 = forward_prop(X, W1, W2, b1, b2)
    #break
    dW1, dW2, db1, db2 = backward_prop(A2)
    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    #break
    
    

