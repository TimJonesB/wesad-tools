#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:35:48 2019

@author: tim
"""

import h5py
import numpy as np
from sklearn.utils import class_weight


class DataHandler:
    def __init__(self,data_filepath):
        self.data_filepath = data_filepath
        self.data = h5py.File(data_filepath,'r')

    
    def get_data_indices(self, percent_train = 0.75):
        data_indices = [int(key.lstrip('Sample')) for key in self.data]
        n = max(data_indices)
        n_train = int(percent_train*n)
        train_indices = np.random.choice(n, n_train, replace=False)
        self.indices = {
            'train' : train_indices,
            'test' : [i for i in range(n) if i not in train_indices]
        }
    
        return self.indices
    

    def get_labels(self, indices):
        data_labels = np.zeros(len(indices))
        for i, index in enumerate(indices):
            data_labels[i] = self.data['Sample{}'.format(index)].attrs['label']
        data_labels_list = list(set(data_labels))
        normalize_label = {data_labels_list[i]: i for i in range(len(data_labels_list))}
        return [normalize_label[i] for i in data_labels]
    
    
    def get_sample(self, data_point_name, data_features):
        data_sample = np.zeros((4200,len(data_features)))
        for i,dim in enumerate(data_features):
            data_sample[:,i] = self.data[data_point_name][dim][:]
        return data_sample
    
    
    def get_data(self, indices, data_features):
        data_set = np.zeros((len(indices),4200,len(data_features)))
        for i,index in enumerate(indices):
            data_set[i, :, :] = self.get_sample('Sample{}'.format(index), data_features)
        return data_set
    
    
    def get_data_sets(self, data_features):
        """
        Data comes in the form 
        h5py.File
            -> Sample0
            -> Sample1
                Attributes
                Feature0
                Feature1
                .
                .8
                .
                FeatureN
            .
            .
            .
            ->SampleN
        """

        indices = self.get_data_indices()
        train_idx = indices['train']
        test_idx = indices['test']
    
        return  {
            'x_train' : self.get_data(train_idx, data_features),
            'y_train': self.get_labels(train_idx),
            'x_test'  : self.get_data(test_idx, data_features),
            'y_test': self.get_labels(test_idx)
        }

    def get_label_count(self):
        labels = [self.data[key].attrs['label'] for key in self.data]
        label_count = {label:0 for label in set(labels)}
        for label in labels:
            label_count[label] += 1
        return label_count

    def compute_class_weights(y_train):
        return class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)



if __name__ == '__main__':
    print('Loading Data...')

    data_features = [
                     'chestACC0',
                     #'chestACC1',
                     #'chestACC2',
                     'chestECG',
                     #'chestEDA',
                     #'chestEMG',
                     #'chestResp',
                     'chestTemp',
                     'wristACC0',
                     #'wristACC1',
                     #'wristACC2',
                     #'wristBVP',
                     #'wristEDA',
                     'wristTEMP',
                   ]

    dfile = r'../data/formatted_data.h5'
    d = DataHandler(dfile).get_data_sets(data_features)






