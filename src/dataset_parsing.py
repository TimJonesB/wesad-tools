#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29
[1] Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven. 2018. 
Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. In 2018 International 
Conference on Multimodal Interaction (ICMI â€™18), October 16â€“20, 2018, Boulder, CO, USA. ACM, New York, NY, USA, 9 pages.

0 = not defined / transient, 
1 = baseline, 
2 = stress, 
3 = amusement,
4 = meditation, 
5/6/7 = should be ignored in this dataset
"""
import os
import numpy as np
import pickle
import h5py
from datetime import datetime
from feature_editing import FeatureDesigner
import time

"""
@brief DataInfo contains configuration information about WESAD data. See paper in README for more information.
@todo Use a standalone configuration file instead of a built in class
"""
class DataInfo:
    # Index of subject. Each subject has a variable length of sample data. Some numbers skipped per data source recommendation.
    data_idx = ['2', '3', '4', '5', '6',
                '7', '8', '9', '10', '11',
                '13', '14', '15', '16', '17']
    # Sampling frequency of label data
    label_data_fs = 700
    # Sampling frequency of chest and wrist sensor data streams
    fs_hz = { #see README
        'chest' : {
            signal : 700 for signal in ['ACC', 'ECG', 'EMG',
                                        'EDA', 'Temp', 'Resp']
        },
        'wrist' : {
            'ACC'  : 32,
            'BVP'  : 64,
            'EDA'  : 4,
            'TEMP' : 4
        }
    }
    # Saves off generation datetime for reference
    def __init__(self):
        now = datetime.now()
        self.date = now.strftime("%d/%m/%Y %H:%M:%S")


"""
@brief DataProducer handles the reading and parsing of subject data streams one subject at a time
"""
class DataProducer:

    """
    @brief init method for DataProducer loads in data and creates a FeatureDesigner instance member
    @param self class instance
    @param data_filepath file path to subject datafile
    @param data_info instance of data info configuration data
    """
    def __init__(self, data_filepath, data_info):
        self.data_filepath = data_filepath
        self.data_info = data_info
        with open(data_filepath, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1' )
        self.designer = FeatureDesigner(data_info, common_hz=700, common_len=len(self.data['label']))


    """
    @brief parse calls FeatureDesigner feature-by-feature and saves off feature vectors
    @param self class instance
    @returns feature vectors for all features for the subject
    """
    def parse(self):
        feat_vecs = {}
        for component in ([('chest', feat) for feat in ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']] +
                          [('wrist', feat) for feat in ['ACC', 'BVP', 'EDA', 'TEMP']]):
            print(f'Parsing {component[0]}{component[1]} data')
            feat_vecs.update( self.designer.edit_feature(f'{component[0]}{component[1]}',
                                                   self.data['signal'][component[0]][component[1]]) )
        return feat_vecs


if __name__ == '__main__':
    output_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),
                                               '..', 
                                               'data',
                                               'formatted_data_feat.h5'))
    data_info = DataInfo()
    with h5py.File(output_path, 'w') as fout:
        fout.attrs['date'] = data_info.date
        for idx in data_info.data_idx:
            data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),
                                                    '..',
                                                    'data',
                                                    'WESAD',
                                                    f'S{idx}',
                                                    f'S{idx}.pkl'))
            print(f'Parsing {data_path}')
            t0 = time.time()
            grp = fout.create_group(f'S{idx}')
            dp = DataProducer(data_path, data_info)
            feat_vecs = dp.parse()
            for feat in feat_vecs:
                print(feat)
                grp.create_dataset(feat, data=feat_vecs[feat])
            grp.create_dataset('label', data= dp.data['label'])
            print("Time = ", time.time()-t0)
