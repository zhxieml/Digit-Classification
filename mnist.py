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
                self._data[file_name] = self._flatten(self._data[file_name]) / np.float32(255.)
        
        print('Done\n')
        
        print('####### MNIST DATASET #######')
        for file_name in _FILES:
            print('{}:\t{}'.format(file_name, self._data[file_name].shape))
        
    def _download(self):
        """Downloads MNIST dataset."""
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
            
            with gzip.open(raw_path, 'rb') as raw_data:
                if file_name.split('_')[-1] == 'images':
                    _, num_data, rows, cols = struct.unpack('>IIII', raw_data.read(16))
                    processed_data = np.array(
                        array.array("B", raw_data.read()),
                        dtype=np.uint8
                    ).reshape(num_data, rows, cols)
                else:
                    _ = struct.unpack(">II", raw_data.read(8))
                    processed_data = np.array(array.array("B", raw_data.read()), dtype=np.uint8)[:, None]
                    
                np.save(file=processed_path, arr=processed_data)
        
        print('Done')
        
    @staticmethod
    def _flatten(X):
        """Flattens all but the first dimension of a narray.
        
        Args:
            X: A 3d narray representing image data with the shape of (num, width, height). 
        """
        return np.reshape(X, (X.shape[0], -1))
            
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
    from linear_model import LogisticRegression
    
    mnist = MNIST('./data', download=False)
    train_images, train_labels = mnist.train_data
    train_labels = np.array(train_labels[:, None] == 6, dtype=np.float32)
    
    logistic = LogisticRegression()
    logistic.fit(train_images, train_labels)