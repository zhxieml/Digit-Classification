import numpy as np
from numpy import random
from numpy import linalg as LA

from utils.calculation import sigmoid


class LogisticRegression():
    # solver = ['gd', 'sgd'. 'newton']
    def __init__(self, solver='sgd', batch_size=64, tolerance=1e-3, n_iter_no_change=5, learning_rate=1e-4):
        self._w = None
        self._solver = solver
        self._batch_size = batch_size
        self._tolerance = tolerance
        self._n_iter_no_change = n_iter_no_change
        self._learning_rate = learning_rate
    
    def fit(self, X, y, verbose=False):
        num_samples, num_feats = X.shape
        min_loss = np.inf
        count_no_change = 0
        
        self._w = random.uniform(low=-1, high=1, size=(num_feats, ))

        while count_no_change < self._n_iter_no_change:
            # TODO: a better way
            if count_no_change >= 10:
                self._learning_rate /= 1.001
                
            if self._solver == 'sgd':
                assert num_samples >= self._batch_size
                sampled = random.choice(num_samples, self._batch_size, replace=False)
                loss, grad = self._cal_loss_and_grad(X[sampled, :], y[sampled], self._w)
            else:
                loss, grad = self._cal_loss_and_grad(X, y, self._w)
            
            if verbose:
                print('Loss:\t{:.4f}\tLearning rate:\t{:.3f}'.format(loss, self._learning_rate))
            
            if loss > min_loss - self._tolerance:
                count_no_change += 1
            else:
                count_no_change = 0
                
            min_loss = min(min_loss, loss)
            
            # Updates
            self._w -= self._learning_rate * grad
        
    def predict(self, X, threshold=0.5):
        assert self._w is not None
        
        return sigmoid(np.dot(X, self._w)) > threshold
        
    @staticmethod
    def _cal_loss_and_grad(X, y, w):
        num_samples, _ = X.shape
        
        proj = np.dot(X, w)
        # loss = -\sum_{i=1}^{n}\left[y_{i}^{*} X_{i}^{\top} w+\log \left(1+\exp X_{i}^{\top} w\right)\right]
        loss = np.sum(np.log(1 + np.exp(proj))) - np.dot(y, proj)
        # grad = \sum_{i=1}^{n}\left[y_{i} X_{i}-\frac{e^{X_{i}^{\top} \beta}}{1+e^{X_{i}^{\top} \beta}} X_{i}\right]
        grad = np.dot(X.T, 1 - 1 / (1 + np.exp(proj)) - y)
        
        return loss / num_samples, grad / num_samples
        