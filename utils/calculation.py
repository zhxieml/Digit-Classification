import numpy as np
from numpy import linalg as LA


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

def rbf_kernel(x1, x2, sigma):
    return np.exp(-0.5 / sigma ** 2 * LA.norm(x1 - x2) ** 2) 

def poly_kernel(x1, x2, d):
    return np.dot(x1, x2) ** d

def cosine_kernel(x1, x2):
    return np.dot(x1, x2) / (LA.norm(x1) * LA.norm(x2))

def rbf_kernel_matrix_general(X, Y, sigma):
    X_norm = LA.norm(X, axis=1) ** 2
    Y_norm = LA.norm(Y, axis=1) ** 2
    
    return np.exp(-0.5 / sigma ** 2 * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)))

def rbf_kernel_matrix(X, sigma):
    X_norm = LA.norm(X, axis=1) ** 2
    
    return np.exp(-0.5 / sigma ** 2 * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))

def poly_kernel_matrix_general(X, Y, d):
    return np.dot(X, Y.T) ** d

def poly_kernel_matrix(X, d):
    return np.dot(X, X.T) ** d

def cosine_kernel_matrix_general(X, Y):
    X_norm = LA.norm(X, axis=1)
    Y_norm = LA.norm(Y, axis=1)
    
    return np.dot(X, Y.T) / (X_norm[:, None] * Y_norm[None, :])

def cosine_kernel_matrix(X):
    X_norm = LA.norm(X, axis=1)
    
    return np.dot(X, X.T) / (X_norm[:, None] * X_norm[None, :])