import numpy as np
from numpy import random
from numpy import linalg as LA

from utils.calculation import sigmoid
from utils.calculation import rbf_kernel_matrix, rbf_kernel_matrix_general
from utils.calculation import poly_kernel_matrix, poly_kernel_matrix_general
from utils.calculation import cosine_kernel_matrix, cosine_kernel_matrix_general


class KernelLassoLogisticRegression():
    # solver = ['gd', 'sgd'. 'newton']
    def __init__(self, kernel='rbf', alpha=1.0, solver='gd', batch_size=64, tolerance=1e-3, n_iter_no_change=5, learning_rate=1e-4, sigma=1, d=1):
        self._c = None
        self._bases = None
        self._kernel = kernel
        self._alpha = alpha
        self._solver = solver
        self._batch_size = batch_size
        self._tolerance = tolerance
        self._n_iter_no_change = n_iter_no_change
        self._learning_rate = learning_rate
        
        self._sigma = sigma
        self._d = d
    
    def fit(self, X, y, verbose=False):
        num_samples, _ = X.shape
        min_loss = np.inf
        count_no_change = 0

        self._c = random.uniform(low=-1, high=1, size=(num_samples, ))
        self._bases = X
        
        while count_no_change < self._n_iter_no_change:
            # TODO: a better way
            if count_no_change >= 5:
                self._learning_rate /= 1.3
                
            if self._solver == 'sgd':
                assert num_samples >= self._batch_size
                sampled = random.choice(num_samples, self._batch_size, replace=False)
                K_sampled = self._cal_kernel_matrix_sampled(X, sampled)
                loss, grad = self._cal_loss_and_grad(K_sampled, y[sampled], self._c, self._alpha)
            else:
                K = self._cal_kernel_matrix(X)
                loss, grad = self._cal_loss_and_grad(K, y, self._c, self._alpha)
            
            if verbose:
                print('Loss:\t{:.4f}\tLearning rate:\t{:.3f}'.format(loss, self._learning_rate))
            
            if loss > min_loss - self._tolerance:
                count_no_change += 1
            else:
                count_no_change = 0
                
            min_loss = min(min_loss, loss)
            
            # Updates
            self._c -= self._learning_rate * grad
            
    # def fit(self, X, y, verbose=False):
    #     num_samples, _ = X.shape
    #     min_loss = np.inf
    #     count_no_change = 0

    #     self._c = random.uniform(low=-1, high=1, size=(num_samples, ))
    #     self._bases = X
        
    #     while count_no_change < self._n_iter_no_change:
    #         # TODO: a better way
    #         if count_no_change >= 5:
    #             self._learning_rate /= 1.001
                
    #         if self._solver == 'sgd':
    #             assert num_samples >= self._batch_size
    #             sampled = random.choice(num_samples, self._batch_size, replace=False)
    #             K_sampled = self._cal_kernel_matrix_sampled(X, sampled)
    #             loss, grad = self._cal_loss_and_grad(K_sampled, y[sampled], self._c, self._alpha)
    #         else:
    #             K = self._cal_kernel_matrix(X)
    #             loss, grad = self._cal_loss_and_grad(K, y, self._c, self._alpha)
            
    #         if verbose:
    #             print('Loss:\t{:.4f}\tLearning rate:\t{:.3f}'.format(loss, self._learning_rate))
            
    #         if loss > min_loss - self._tolerance:
    #             count_no_change += 1
    #         else:
    #             count_no_change = 0
                
    #         min_loss = min(min_loss, loss)
            
    #         # Updates
    #         self._c -= self._learning_rate * grad
        
    def predict(self, X, threshold=0.5):
        assert self._c is not None and self._bases is not None and self._c.shape[0] == self._bases.shape[0]
        
        if self._kernel == 'rbf':
            y_pred = np.dot(rbf_kernel_matrix_general(X, self._bases, self._sigma), self._c)
        elif self._kernel == 'poly':
            y_pred = np.dot(poly_kernel_matrix_general(X, self._bases, self._d), self._c)
        elif self._kernel == 'cos':
            y_pred = np.dot(cosine_kernel_matrix_general(X, self._bases), self._c)
        
        return sigmoid(y_pred) > threshold
    
    # def predict(self, X, threshold=0.5):
    #     assert self._c is not None and self._bases is not None and self._c.shape[0] == self._bases.shape[0]
        
    #     # Since the number of training samples is large, use batches
    #     num_samples, _ = X.shape
    #     num_bases, _ = self._bases.shape
    #     batch_size = 10000
    #     y_pred = np.zeros((num_samples, ))
        
    #     if self._kernel == 'rbf':
    #         for group in np.array_split(range(num_bases), num_bases // batch_size):
    #             y_pred += np.dot(rbf_kernel_matrix_general(X, self._bases[group, :], self._sigma), self._c[group])
    #     elif self._kernel == 'poly':
    #         for group in np.array_split(range(num_bases), num_bases // batch_size):
    #             y_pred += np.dot(poly_kernel_matrix_general(X, self._bases[group, :], self._d), self._c[group])
    #     elif self._kernel == 'cos':
    #         for group in np.array_split(range(num_bases), num_bases // batch_size):
    #             y_pred += np.dot(cosine_kernel_matrix_general(X, self._bases[group, :]), self._c[group])
        
    #     return sigmoid(y_pred) > threshold
    
    def _cal_kernel_matrix(self, X):
        if self._kernel == 'rbf':
            return rbf_kernel_matrix(X, self._sigma)
        elif self._kernel == 'poly':
            return poly_kernel_matrix(X, self._d)
        elif self._kernel == 'cos':
            return cosine_kernel_matrix(X)
        else:
            raise ValueError
        
    def _cal_kernel_matrix_sampled(self, X, sampled):
        if self._kernel == 'rbf':
            return rbf_kernel_matrix_general(X[sampled, :], X, self._sigma)
        elif self._kernel == 'poly':
            return poly_kernel_matrix_general(X[sampled, :], X, self._d)
        elif self._kernel == 'cos':
            return cosine_kernel_matrix_general(X[sampled, :], X)
        else:
            raise ValueError
                    
    @staticmethod
    def _cal_loss_and_grad(K, y, c, alpha):
        num_samples, _ = K.shape
        
        proj = np.dot(K, c)
        loss = (np.sum(np.log(1 + np.exp(proj))) - np.dot(y, proj)) / num_samples + alpha * LA.norm(c, ord=1)
        grad = np.dot(K.T, 1 - 1 / (1 + np.exp(proj)) - y) / num_samples + alpha * np.sign(c)
        
        return loss, grad
