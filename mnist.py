"""MNIST Dataset."""

__author__ = 'Zhihui Xie <fffffarmer@gmail.com>'

import array
import gzip
import os
from os import path
import struct

import numpy as np
import urllib
from urllib import request

from utils.calculation import flatten


_BASE_URL = 'http://yann.lecun.com/exdb/mnist/'
_FILES = {
    'train_images': 'train-images-idx3-ubyte.gz', 
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz', 
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}


class MNIST():
    """MNIST dataset helper.
    
    Downloads raw data from the dataset website and then processes them into .npy files.
    
    Attributes:
        train_data: A tuple consisting of training images and training labels.
        test_data: A tuple consisting of testing images and testing labels.
    """
    
    def __init__(self, root_dir, download=False):
        """Inits MNIST dataset helper."""
        self._root_dir = root_dir
        self._data = dict()
        
        if download:
            self._download()
            
        for file_name in _FILES:
            processed_path = path.join(self._processed_dir, file_name + '.npy')
            self._data[file_name] = np.load(processed_path)
        
        print('Postpocessing...', end='')
        for file_name in _FILES:
            if file_name.split('_')[-1] == 'images':
                self._data[file_name] = flatten(self._data[file_name]) / np.float32(255.)
        
        print('Done\n')
        
        print('####### MNIST DATASET #######')
        for file_name in _FILES:
            print('{}:\t{}'.format(file_name, self._data[file_name].shape))
        
    def _download(self):
        """Downloads and processes MNIST dataset."""
        if not path.exists(self._raw_dir):
            os.makedirs(self._raw_dir)
        if not path.exists(self._processed_dir):
            os.makedirs(self._processed_dir)
            
        for file_name in _FILES:
            raw_path = path.join(self._raw_dir, _FILES[file_name])
            
            if not path.isfile(raw_path):
                print('Downloading {}...'.format(file_name), end='')
                try:
                    request.urlretrieve(_BASE_URL + _FILES[file_name], raw_path)
                    print('Done')
                except urllib.ContentTooShortError as e:
                    os.remove(raw_path)
                    print('\nDownloading stops due to unstable network conditions. Please retry later.')
                    raise e
                
        print('Processing raw data...', end='')
        for file_name in _FILES:
            raw_path = path.join(self._raw_dir, _FILES[file_name])
            processed_path = path.join(self._processed_dir, file_name)
            
            if not path.isfile(processed_path):
                with gzip.open(raw_path, 'rb') as raw_data:
                    if file_name.split('_')[-1] == 'images':
                        _, num_data, rows, cols = struct.unpack('>IIII', raw_data.read(16))
                        processed_data = np.array(
                            array.array("B", raw_data.read()),
                            dtype=np.uint8
                        ).reshape(num_data, rows, cols)
                    else:
                        _ = struct.unpack(">II", raw_data.read(8))
                        processed_data = np.array(array.array("B", raw_data.read()), dtype=np.uint8)
                        
                    np.save(file=processed_path, arr=processed_data)
        
        print('Done')
            
    @property
    def _raw_dir(self):
        return path.join(self._root_dir, 'raw')
    
    @property
    def _processed_dir(self):
        return path.join(self._root_dir, 'processed')
    
    @property
    def train_data(self):
        return self._data['train_images'], self._data['train_labels']
    
    @property
    def test_data(self):
        return self._data['test_images'], self._data['test_labels']        
        
if __name__ == '__main__':
    import time

    from numpy import random
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    
    from linear_model import *
    from non_linear_model import *
    from cluster_model import GMM
    
    mnist = MNIST('./data', download=False)
    acc = {'LR': [], 'LR_Ridge': [], 'LR_Lasso': [],  'LR_Lasso_IRLS': [], 'LR_Lasso_Kernel': [],
              'SVM_Linear': [], 'SVM_Nonlinear': [], 'LDA': [], 'GMM': []}
    train_tim = {'LR': [], 'LR_Ridge': [], 'LR_Lasso': [],  'LR_Lasso_IRLS': [], 'LR_Lasso_Kernel': [],
              'SVM_Linear': [], 'SVM_Nonlinear': [], 'LDA': [], 'GMM': []}
    test_tim = {'LR': [], 'LR_Ridge': [], 'LR_Lasso': [],  'LR_Lasso_IRLS': [], 'LR_Lasso_Kernel': [],
              'SVM_Linear': [], 'SVM_Nonlinear': [], 'LDA': [], 'GMM': []}
    
    for digit in range(10):
        # for test in range(5):
        train_images, train_labels = mnist.train_data
        test_images, test_labels = mnist.test_data
        train_labels = np.array(train_labels == digit, dtype=np.float32)
        test_labels = np.array(test_labels == digit, dtype=np.float32)

        # logistic = LogisticRegression(solver='sgd', batch_size=64, tolerance=1e-7, n_iter_no_change=100, learning_rate=1e-1)
        # start = time.time()
        # logistic.fit(train_images, train_labels)
        # train_t = time.time() - start
        # print('Training time: {}'.format(train_t))
        # start = time.time()
        # predicted_labels = logistic.predict(test_images)
        # test_t = time.time() - start
        # print('Testing time: {}'.format(train_t))
        # right_num = np.sum(predicted_labels == test_labels)
        # print(right_num)
        # acc['LR'].append(right_num)
        # train_tim['LR'].append(train_t)
        # test_tim['LR'].append(test_t)
        
        # ridge = RidgeLogisticRegression(alpha=1.0, solver='sgd', batch_size=64, tolerance=1e-7, n_iter_no_change=100, learning_rate=1e-1)
        # start = time.time()
        # ridge.fit(train_images, train_labels)
        # train_t = time.time() - start
        # print('Training time: {}'.format(train_t))
        # start = time.time()
        # predicted_labels = ridge.predict(test_images)
        # test_t = time.time() - start
        # print('Testing time: {}'.format(train_t))
        # right_num = np.sum(predicted_labels == test_labels)
        # print(right_num)
        # acc['LR_Ridge'].append(right_num)
        # train_tim['LR_Ridge'].append(train_t)
        # test_tim['LR_Ridge'].append(test_t)
                
        # lasso = LassoLogisticRegression(alpha=1.0, solver='sgd', batch_size=64, tolerance=1e-7, n_iter_no_change=100, learning_rate=1e-1)
        # start = time.time()
        # lasso.fit(train_images, train_labels)
        # train_t = time.time() - start
        # print('Training time: {}'.format(train_t))
        # start = time.time()
        # predicted_labels = lasso.predict(test_images)
        # test_t = time.time() - start
        # print('Testing time: {}'.format(train_t))
        # right_num = np.sum(predicted_labels == test_labels)
        # print(right_num)
        # acc['LR_Lasso'].append(right_num)
        # train_tim['LR_Lasso'].append(train_t)
        # test_tim['LR_Lasso'].append(test_t)
        
        # lasso_irls = LassoLogisticRegressionIRLS(alpha=1, batch_size=64)
        # start = time.time()
        # lasso_irls.fit(train_images, train_labels)
        # train_t = time.time() - start
        # print('Training time: {}'.format(train_t))
        # start = time.time()
        # predicted_labels = lasso_irls.predict(test_images)
        # test_t = time.time() - start
        # print('Testing time: {}'.format(train_t))
        # right_num = np.sum(predicted_labels == test_labels)
        # print(right_num)
        # acc['LR_Lasso_IRLS'].append(right_num)
        # train_tim['LR_Lasso_IRLS'].append(train_t)
        # test_tim['LR_Lasso_IRLS'].append(test_t)
        
        # lda = LDA()
        # start = time.time()
        # lda.fit(train_images, train_labels)
        # train_t = time.time() - start
        # print('Training time: {}'.format(train_t))
        # start = time.time()
        # predicted_labels = lda.predict(test_images)
        # test_t = time.time() - start
        # print('Testing time: {}'.format(train_t))
        # right_num = np.sum(predicted_labels == test_labels)
        # print(right_num)
        # acc['LDA'].append(right_num)
        # train_tim['LDA'].append(train_t)
        # test_tim['LDA'].append(test_t)
        
        # linear_svm = LinearSVC()
        # start = time.time()
        # linear_svm.fit(train_images, train_labels)
        # train_t = time.time() - start
        # print('Training time: {}'.format(train_t))
        # start = time.time()
        # predicted_labels = linear_svm.predict(test_images)
        # test_t = time.time() - start
        # print('Testing time: {}'.format(train_t))
        # right_num = np.sum(predicted_labels == test_labels)
        # print(right_num)
        # acc['SVM_Linear'].append(right_num)
        # train_tim['SVM_Linear'].append(train_t)
        # test_tim['SVM_Linear'].append(test_t)
        
        # nonlinear_svm = SVC()
        # start = time.time()
        # nonlinear_svm.fit(train_images, train_labels)
        # train_t = time.time() - start
        # print('Training time: {}'.format(train_t))
        # start = time.time()
        # predicted_labels = nonlinear_svm.predict(test_images)
        # test_t = time.time() - start
        # print('Testing time: {}'.format(train_t))
        # right_num = np.sum(predicted_labels == test_labels)
        # print(right_num)
        # acc['SVM_Nonlinear'].append(right_num)
        # train_tim['SVM_Nonlinear'].append(train_t)
        # test_tim['SVM_Nonlinear'].append(test_t)
        
        sampled = random.choice(60000, 6000, replace=False)
        train_images_sampled, train_labels_sampled = train_images[sampled, :], train_labels[sampled]
        
        kernel = KernelLassoLogisticRegression(alpha=0.5, solver='gd', batch_size=64, tolerance=1e-4, n_iter_no_change=100, learning_rate=1)
        start = time.time()
        kernel.fit(train_images_sampled, train_labels_sampled)
        train_t = time.time() - start
        print('Training time: {}'.format(train_t))
        start = time.time()
        predicted_labels = kernel.predict(test_images)
        test_t = time.time() - start
        print('Testing time: {}'.format(train_t))
        right_num = np.sum(predicted_labels == test_labels)
        print(right_num)
        acc['LR_Lasso_Kernel'].append(right_num)
        train_tim['LR_Lasso_Kernel'].append(train_t)
        test_tim['LR_Lasso_Kernel'].append(test_t)

    print(acc)
    print(train_tim)
    print(test_tim)
