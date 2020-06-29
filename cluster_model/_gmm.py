import numpy as np
from numpy import random
from scipy import stats


class GMM():
    def __init__(self, num_components=10, num_iter=100):
        self._num_components = num_components
        self._num_iter = num_iter
        self._M = None
        self._C = None
        self._alpha = None
        
    def fit(self, X):
        num_samples, num_feats = X.shape
        
        # Initialize
        self._M = random.uniform(low=-1, high=1, size=(self._num_components, num_feats))
        self._C = np.eye(num_feats).reshape((-1, num_feats, num_feats))
        self._C = np.repeat(self._C, self._num_components, axis=0)
        self._alpha = np.repeat(1.0 / self._num_components, self._num_components)
        
        for i in range(self._num_iter):
            gamma = self._expectation_step(X, self._M, self._C, self._alpha)
            self._M, self._C, self._alpha = self._maximization_step(X, gamma)
            
    def predict(self, X):
        gamma = self._expectation_step(X, self._M, self._C, self._alpha)
        
        return gamma.argmax(axis=1)
    
    def _expectation_step(self, X, M, C, alpha):
        num_samples, num_feats = X.shape
        
        prior = np.empty((num_samples, self._num_components))

        for k in range(self._num_components):
            prior[:, k] = self._gaussian_distribution(X, M[k, :], C[k, :, :])
            
        post = prior * alpha
        post /= np.sum(post, axis=1)[:, None]
        
        return post
    
    def _maximization_step(self, X, gamma):
        """M step.
        
            gamma: n x k
        """
        num_samples, num_feats = X.shape
        
        N = np.sum(gamma, axis=0)
        M = np.matmul(gamma.T, X) / N[:, None]
        alpha = N / num_samples
        C = np.empty((self._num_components, num_feats, num_feats))
        
        # TODO: a better way
        for k in range(self._num_components):
            bias = (X - M[k, :])
            C[k, :, :] = np.matmul(bias.T, np.multiply(bias, gamma[:, [k]])) / N[k]
        
        return M, C, alpha
    
    @staticmethod
    def _gaussian_distribution(x, mean, cov):
        gaussian = stats.multivariate_normal(mean=mean, cov=cov)
        
        return gaussian.pdf(x)
    
    
if __name__ == '__main__':
    # generate random data
    cov1 = np.array([[0.3, 0], [0, 0.1]])
    cov2 = np.array([[0.2, 0], [0, 0.3]])

    mu1 = np.array([0, 1])
    mu2 = np.array([2, 1])

    data = np.empty((100, 2))
    data[:30, :] = np.random.multivariate_normal(mean=mu1, cov=cov1, size=30)
    data[30:, :] = np.random.multivariate_normal(mean=mu2, cov=cov2, size=70)
    
    gmm = GMM(num_components=2)
    
    gmm.fit(data)
    print(gmm.predict(data))
    