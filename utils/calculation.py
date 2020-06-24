import numpy as np


def flatten(X):
    """Flattens all but the first dimension of a narray.
        
    Args:
        X: A 3d narray representing image data with the shape of (num, width, height). 
    """
    return np.reshape(X, (X.shape[0], -1))

def one_hot(y, k):
    """Encodes labels into a k-category one-hot.
    
    Args:
        y: A 1d narray representing label data with the shape of (num, ).
    """
    return np.array(y[:, None] == np.arange(k), dtype=np.float32)

def soft_thresholding(x, threshold):
    """Soft thresholding function.
    
        Note that the threshold is larger than 0.
    """
    return max(x - threshold, 0) - max(-x - threshold, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))