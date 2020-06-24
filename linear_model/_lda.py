import numpy as np
from numpy import linalg as LA


class LDA():
    def __init__(self):
        self._w = None
        self._threshold = None
    
    def fit(self, X, y):
        pos_indices, neg_indices = np.nonzero(y == 1)[0], np.nonzero(y != 1)[0]
        X_pos = X[pos_indices, :]
        X_neg = X[neg_indices, :]
        
        X_mean_pos = np.mean(X_pos, axis=0)
        X_mean_neg = np.mean(X_neg, axis=0)
        X_bias_pos = X - X_mean_pos
        X_bias_neg = X - X_mean_neg
        
        # TODO: double check
        S_within = np.matmul(X_bias_pos.T, X_bias_pos) + np.matmul(X_bias_neg.T, X_bias_neg)
        u, s, vh = LA.svd(S_within, full_matrices=False, compute_uv=True)
        S_within_inv = np.dot(vh.T, np.dot(LA.inv(np.diag(s)), u.T))
        
        self._w = np.dot(S_within_inv, X_mean_pos - X_mean_neg)
        self._threshold = 0.5 * np.dot(self._w, X_mean_pos + X_mean_neg)
    
    def project(self, X, threshold=None):
        assert self._w is not None
        
        if threshold is not None:
            return np.dot(X, self._w) > threshold
        
        assert self._threshold is not None
        
        return np.dot(X, self._w) > self._threshold
        