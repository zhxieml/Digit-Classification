import numpy as np
from numpy import random
from numpy import linalg as LA


class LassoLeastSquareRegression():
    def __init__(self, alpha=1.0, batch_size=64, tolerance=1e-3, n_iter_no_change=5, learning_rate=1e-4):
        self._w = None
        self._alpha = alpha
        self._batch_size = batch_size
        self._tolerance = tolerance
        self._n_iter_no_change = n_iter_no_change
        self._learning_rate = learning_rate
    
    def fit(self, X, y, verbose=False):
        num_samples, num_feats = X.shape
        self._w = random.uniform(low=-1, high=1, size=(num_feats, ))

        # TODO: max_iter
        for k in range(100):
            assert num_samples >= self._batch_size
            sampled = random.choice(num_samples, self._batch_size, replace=False)
            
            if verbose:
                print('Loss:\t{:.4f}'.format(self._cal_loss(X[sampled, :], y[sampled], self._w)))

            self._w = self._coordinate_descent_least_squares(X, y, self._w, self._alpha)
    
    @staticmethod
    def _cal_loss(X, y, w):
        num_samples, _ = X.shape
        
        proj = np.dot(X, w)
        loss = LA.norm(y - proj) ** 2
        
        return loss / num_samples
     
    @staticmethod
    def _coordinate_descent_least_squares(X, y, w, init_alpha):
        """Use coordinate descent to solve the L1 constrained least squares problem.
        """ 
        _, num_feats = X.shape
        
        # TODO: a better way
        for alpha in np.linspace(init_alpha, init_alpha / 5, 10):
            for dim in range(num_feats):
                r = y - np.dot(X, w) + X[:, dim] * w[dim]
                norm = LA.norm(X[:, dim]) ** 2
                gamma = np.dot(r, X[:, dim])
                                
                w[dim] = np.sign(gamma) * max(0, np.nan_to_num((abs(gamma) - alpha) / norm))
        
        return w