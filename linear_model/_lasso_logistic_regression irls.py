import numpy as np
from numpy import random
from numpy import linalg as LA

from utils.calculation import sigmoid


class LassoLogisticRegression():
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
        for k in range(1000):
            assert num_samples >= self._batch_size
            sampled = random.choice(num_samples, self._batch_size, replace=False)
            
            if verbose:
                print('Loss:\t{:.4f}'.format(self._cal_loss(X[sampled, :], y[sampled], self._w)))

            self._w = self._IRLS(X[sampled, :], y[sampled], self._w, self._alpha)
        
    def predict(self, X, threshold=0.5):
        assert self._w is not None
        
        return sigmoid(np.dot(X, self._w)) > threshold
    
    @staticmethod
    def _cal_grad(X, y, w):
        num_samples, _ = X.shape
        
        proj = np.dot(X, w)
        grad = np.dot(X.T, 1 - 1 / (1 + np.exp(proj)) - y)
        
        return grad / num_samples
    
    @staticmethod
    def _cal_loss(X, y, w):
        num_samples, _ = X.shape
        
        proj = np.dot(X, w)
        loss = np.sum(np.log(1 + np.exp(proj))) - np.dot(y, proj)
        
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
                
                w[dim] = np.sign(gamma) * max(0, (abs(gamma) - alpha) / norm)
        
        return w
    
    def _IRLS(self, X, y, w, init_alpha):
        """Solve the L1 Regularized Logistic Regression by IRLS-coordinate.
        
            This solution is inspired by 'Efficient L1 Regularized Logistic Regression',
            where LARS is used instead of coordinate descent method.
        """ 
        num_samples, num_feats = X.shape
        
        proj = np.dot(X, w)
        predicted = sigmoid(proj)
        
        d = predicted * (1 - predicted)
        # The diagonal matrix (square-rooted)
        D = np.diag(d ** 0.5)
        # The vector
        z = proj + (1 - sigmoid(y * proj)) * y / d
        
        # Solve the sub L1 constrained least squares problem
        w_pseudo = np.zeros(num_feats)
        X_pseudo = np.dot(D, X)
        y_pseudo = np.dot(D, z)
        
        w_pseudo = self._coordinate_descent_least_squares(X_pseudo, y_pseudo, w_pseudo, init_alpha)
        
        # Update by using backtracking line search
        # hyperparameters: alpha = 0.4, beta = 0.95
        # NOTE: alpha here is for backtrackiing line search
        alpha, beta = 0.4, 0.95
        t = 1

        while (self._cal_loss(X, y, w + t * w_pseudo) > self._cal_loss(X, y, w) + 
               alpha * t * np.dot(self._cal_grad(X, y, w), w_pseudo)):
            t *= beta  
        
        w += t * w_pseudo
        
        return w
    
    
    
    # @staticmethod
    # def _coordinate_descent_step(X, y, w, alpha):
    #     _, num_feats = X.shape
        
    #     for k in range(num_feats):
    #         # use gradient descent to find the local minimum
    #         for i in range(10):
    #             proj = np.dot(X, w)
                
    #             f_prime = np.dot(X[:, k] ** 2 / (1 + proj), np.exp(proj) - 1 / (1 + proj))
                
    #             if f_prime == 0:
    #                 break

    #             f = np.dot(X[:, k] / (1 + proj), np.exp(proj)) - np.dot(y, X[:, k]) + alpha * np.sign(w[k])
    #             w[k] -= f / f_prime
            
    #     return w