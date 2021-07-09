#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:26:12 2019

@author: tim

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

class DataInfo:
    """
    Data information from readme.
    
    :param: data_idx = sample labels
    :param: sampling_freq_hz = sampling frequency (hz) of data signals
    """
    # data_idx = ['2', '3', '4', '5', '6',
    #             '7', '8', '9', '10', '11',
    #             '13', '14', '15', '16', '17']
    data_idx = ['2','3','4']
    label_data_fs = 700
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
    segment_duration = 10 
    valid_classes = (1,2)

    def __init__(self):
        now = datetime.now()
        self.date = now.strftime("%d/%m/%Y %H:%M:%S")


class DataProducer:
    def __init__(self, data_filepath, data_info):
        self.data_filepath = data_filepath
        self.data_info = data_info
        with open(data_filepath, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')


    def upsample_wrist_data(self, label_fs=700):
        """
        Function to upsample wrist data (see sampling_freq_hz dictionary for 
        original sampling rate) to the sampling rate of label data and chest 
        data (both at 700hz).
        
        :param: data : data from data file
        :param: label_fs : sampling rate of label and chest data
        """

        wrist_data = self.data['signal']['wrist']
        labels = self.data['label']
        upsampled_wrist_data = {}

        for signal_name in wrist_data:
            signal = wrist_data[signal_name]
            fs = self.data_info.fs_hz['wrist'][signal_name]
            time_upsamp = np.arange(0, len(labels)*(1/label_fs), 1/label_fs)
            time_signal = np.arange(0, len(signal)*(1/fs), 1/fs)

            # Able to handle multidimensional data ie 'ACC' (Accelerometer)
            sig_upsamp = np.zeros((len(labels), signal.shape[1]))
            for i, sig in enumerate(signal.T):
                sig_upsamp[:, i] = np.interp(time_upsamp, time_signal, sig)
            upsampled_wrist_data[signal_name] = sig_upsamp

        self.data['signal']['wrist_upsampled'] = upsampled_wrist_data
        return upsampled_wrist_data
    
    
    def extract_segment(self, i, steps_segment):
        """
        Extracts segment specific to certain index range [i:i+steps_segment].
        """
        segment = {}
        for sig in ['ACC', 'ECG']:#, 'EMG', 'EDA', 'Temp', 'Resp']:
            whole_series = self.data['signal']['chest'][sig]
            segment[f'chest{sig}'] = whole_series[i:(i + steps_segment)]

        for sig in []:#['ACC', 'EDA', 'TEMP']:
            whole_series = self.data['signal']['wrist_upsampled'][sig]
            segment[f'wrist{sig}'] = whole_series[i:(i + steps_segment)]

        return segment


    def is_valid_segment(self, i, steps_segment):
        """
        Checks that segment has no label/state change and that there are no
        invalid labels (ie 0, 5, 6, 7)

        0 = not defined / transient,
        1 = baseline,
        2 = stress,
        3 = amusement,
        4 = meditation,
        5/6/7 = should be ignored in this dataset
        """
        return (len(set(self.data['label'][i:i+steps_segment])) == 1 and
                self.data['label'][i:i+steps_segment][0] in self.data_info.valid_classes)


if __name__ == '__main__':
    data_info = DataInfo()
    designer = FeatureDesigner(data_info)

    steps_sample = data_info.label_data_fs * data_info.segment_duration

    output_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),
                                               '..', 
                                               'data',
                                               'formatted_data_feat.h5'))
    with h5py.File(output_path, 'w') as fout:
        fout.attrs['date'] = data_info.date
        fout.attrs['segment_duration_s'] = data_info.segment_duration
        fout.attrs['classes'] = data_info.valid_classes
        fout.attrs['sample_timesteps'] = steps_sample
        fout.attrs['sample_count'] = 0
        for idx in data_info.data_idx:
            data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),
                                                     '..',
                                                     'data',
                                                     'WESAD',
                                                     f'S{idx}',
                                                     f'S{idx}.pkl'))
            print(f'Parsing {data_path}')

            producer = DataProducer(data_path, data_info)

            producer.upsample_wrist_data()

            for i in range(0, len(producer.data['label']), steps_sample):

                if producer.is_valid_segment(i, steps_sample):
                    sample = producer.extract_segment(i, steps_sample)
                    group_name = f'Sample{fout.attrs["sample_count"]}'
                    grp = fout.create_group(group_name)

                    grp.attrs['label'] = producer.data['label'][i:i+steps_sample][0]
                    for component in sample:
                        feat_vecs = designer.edit_feature(component,
                                                          sample[component])
                        for feat in feat_vecs:
                            grp.create_dataset(feat, data=feat_vecs[feat])

                    fout.attrs['sample_count'] += 1

        print(f'Parsing complete number of samples = {fout.attrs["sample_count"]}.')
