#creating helper functions

import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def linear_fn(W,X,b):
    logits = np.dot(W,X)+b
    return logits

def softmax_fn(Z):
    smax = np.exp(Z)/np.sum(np.exp(Z),axis=0)
    return smax
