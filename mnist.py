"""MNIST Dataset."""


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
    def __init__(self, root_dir, download=False):
        self.root_dir = root_dir
        self.data = dict()
        
        if download:
            self._download()
            
        for file_name in _FILES:
            processed_path = path.join(self.processed_dir, file_name + '.npy')
            self.data[file_name] = np.load(processed_path)
    
        print(self.data)
        
    def _download(self):        
        if not path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        for file_name in _FILES:
            raw_path = path.join(self.raw_dir, _FILES[file_name])
            
            if not path.isfile(raw_path):
                print('Downloading {}...'.format(file_name), end='')
                try:
                    request.urlretrieve(_BASE_URL + _FILES[file_name], raw_path)
                    print('Done')
                except urllib.ContentTooShortError as e:
                    os.remove(raw_path)
                    print('\nDownloading stops due to unstable network conditions. Please retry later.')
                    raise e
                
        # process raw data
        print('Processing...', end='')
        for file_name in _FILES:
            raw_path = path.join(self.raw_dir, _FILES[file_name])
            processed_path = path.join(self.processed_dir, file_name)
            
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
    def raw_dir(self):
        return path.join(self.root_dir, 'raw')
    
    @property
    def processed_dir(self):
        return path.join(self.root_dir, 'processed')
            
        
if __name__ == '__main__':
    test = MNIST('./data', download=True)
    