#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 17:48:25 2020

@author: tim
"""

import numpy as np
import pandas as pd
import scipy.signal as sps
import matplotlib.pyplot as plt

class FeatureDesigner:
    """
    * Edit feature dispatches to specific designer method based on feature name.
    * {feature_name}designer returns any number of custom features.
    """
    # for feature moving window size
    other_win_duration = 60 # seconds
    chest_win_duration = 5 # seconds

    def __init__(self, cfg_info):
        self.cfg = cfg_info

    def edit_feature(self, feature_name, data):
        designer_map = {
                     'chestACC0' : self.acc_designer,
                     'chestACC1' : self.acc_designer,
                     'chestACC2' : self.acc_designer,
                     'chestECG'  : self.ecg_designer,
                     'chestEDA'  : self.eda_designer,
                     'chestEMG'  : self.emg_designer,
                     'chestResp' : self.resp_designer,
                     'chestTemp' : self.temp_designer,
                     'wristACC0' : self.wacc_designer,
                     'wristACC1' : self.wacc_designer,
                     'wristACC2' : self.wacc_designer,
                     'wristEDA'  : self.weda_designer,
                     'wristTEMP' : self.wtemp_designer,
                     }
        return designer_map[feature_name](feature_name, data)


    def acc_designer(self, feature_name, data):
        def maxf_filter(v, win_sz = 3500, skip=10):
            def dom_nonzero_freq(x):
                y = np.fft.fft(x)
                yf = np.fft.fftfreq(len(x), 1/self.cfg.fs_hz['chest']['ACC'])
                ipart = np.argpartition(y, -2)
                f1, f2 = yf[ipart[-1]], yf[ipart[-2]] # top two dominant frequencies
                return abs(f1) if abs(f1) > 0 else abs(f2)
            signal = np.array([dom_nonzero_freq(v[i:i+win_sz]) for i in range(0, len(v), skip)])
            fs = self.cfg.fs_hz['chest']['ACC'] / skip
            fsup = self.cfg.fs_hz['chest']['ACC']
            time_signal = np.arange(0, len(signal)*(1/fs), 1/fs)
            time_upsamp = np.arange(0, (fsup/fs)*len(signal)*(1/fsup), 1/fsup)
            sig_upsamp = np.interp(time_upsamp, time_signal, signal)
            return sig_upsamp
        win_sz = self.cfg.fs_hz['chest']['ACC'] * self.chest_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        std_filt = pd.Series(data).rolling(win_sz, min_periods=1).std()
        fq_filt = maxf_filter(data)
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Std'.format(feature_name)  :  std_filt,
                '{}Freq'.format(feature_name) :  fq_filt,
               }


    def wacc_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['ACC'] * self.wrist_win_duration
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Max'.format(feature_name) : max,
                '{}Min'.format(feature_name) : data,
               }

    def ecg_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['ECG'] * self.chest_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        std_filt = pd.Series(data).rolling(win_sz, min_periods=1).std()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Std'.format(feature_name)  :  std_filt  
               }


    def eda_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['EDA'] * self.chest_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Max'.format(feature_name)  : self.filt_nan(data, max_filt),
                '{}Min'.format(feature_name)  : self.filt_nan(data, min_filt),
               }


    def weda_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['EDA'] * self.wrist_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Max'.format(feature_name)  : self.filt_nan(data, max_filt),
                '{}Min'.format(feature_name)  : self.filt_nan(data, min_filt),
               }


    def emg_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['EMG'] * self.chest_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
               }


    def resp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['Resp'] * self.chest_win_duration
        win_sz = 70 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
               }


    def temp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['Temp'] * self.chest_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        return {
                '{}Mean'.format(feature_name) : mean_filt,
                '{}Max'.format(feature_name)  : data,
                '{}Min'.format(feature_name)  : data,
               }


    def wtemp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['Temp'] * self.wrist_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        return {
                '{}Mean'.format(feature_name) : mean_filt,
                '{}Max'.format(feature_name)  : data,
                '{}Min'.format(feature_name)  : data,
               }

    
