import numpy as np
from numpy import random
from numpy import linalg as LA

from utils import soft_thresholding
from utils import sigmoid


class LassoRegression():
    def __init__(self, alpha=1.0, batch_size=64, tolerance=1e-3, n_iter_no_change=5, learning_rate=1e-4):
        self._w = None
        self._alpha = alpha
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
                
            assert num_samples >= self._batch_size
            sampled = random.choice(num_samples, self._batch_size, replace=False)
            loss = self._cal_lasso_loss(X[sampled, :], y[sampled], self._w, self._alpha)
            
            if verbose:
                print('Loss:\t{:.4f}'.format(loss))
                        
            if loss > min_loss - self._tolerance:
                count_no_change += 1
            else:
                count_no_change = 0
                
            min_loss = min(min_loss, loss)
            
            # Updates
            self._w = self._coordinate_descent_step(X[sampled, :], y[sampled], self._w, self._alpha)
        
    def predict(self, X, threshold=0.5):
        assert self._w is not None
        
        return 1 / (1 + np.exp(-1 * np.dot(X, self._w))) > threshold
        
    @staticmethod
    def _cal_lasso_loss(X, y, w, alpha):
        num_samples, _ = X.shape
        
        proj = np.dot(X, w)
        loss = np.sum(np.log(1 + np.exp(proj))) - np.dot(y, proj) + alpha * LA.norm(w, ord=1)
        
        return loss / num_samples
    
    @staticmethod
    def _coordinate_descent_step(X, y, w, alpha):
        """Update all coordinates under some fix alpha

            The calculation process is refered to 'Efficient L1 Regularized Logistic Regression'
        """ 
        num_samples, num_feats = X.shape
        
        proj = np.dot(X, w)
        y_est = sigmoid(proj)
        d = y_est * (1 - y_est)
        D = np.diag(y_est * (1 - y_est))
        z = proj + (1 - sigmoid(y * proj)) * y / d
        
        # TODO: 
        
        # for j in range(num_feats):
        #     proj = np.dot(X, w)
        #     y_est = 1 / (1 + np.exp(-1 * proj))
        #     W = np.diag((y_est * (1 - y_est)))
        #     X_j = X[:, j]
            
        #     v_j = (1 / num_samples) * np.matmul(np.matmul(X_j, W), X_j)
        #     # r = np.dot(LA.inv(W), y - y_est)
        #     z_j = (1 / num_samples) * np.dot(X_j, y - y_est) + v_j * w[j]
            
        #     w[j] = soft_thresholding(z_j, alpha) / v_j
            
        #     if v_j > 0:
        #         print(v_j)
            
        return w