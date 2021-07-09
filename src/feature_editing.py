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
import scipy.integrate as integrate
from scipy.stats import linregress

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
                        'chestACC'  : self.acc_designer,
                        'chestECG'  : self.ecg_designer,
                        'chestEDA'  : self.eda_designer,
                        'chestEMG'  : self.emg_designer,
                        'chestResp' : self.resp_designer,
                        'chestTemp' : self.temp_designer,
                        'wristACC'  : self.wacc_designer,
                        'wristEDA'  : self.weda_designer,
                        'wristTEMP' : self.wtemp_designer,
                       }
        return designer_map[feature_name](feature_name, data)


    def acc_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['ACC'] * self.chest_win_duration
        mean_filt0 = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        std_filt0 = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).std()
        fq_filt0 = self.maxf_filter(data[:,0], self.cfg.fs_hz['chest']['ACC'])
        int_filt0 = pd.Series(data[:,0]).rolling(10, min_periods= 1).apply(integrate.trapz)
        mean_filt1 = pd.Series(data[:,1]).rolling(win_sz, min_periods=1).mean()
        std_filt1 = pd.Series(data[:,1]).rolling(win_sz, min_periods=1).std()
        fq_filt1 = self.maxf_filter(data[:,1], self.cfg.fs_hz['chest']['ACC'])
        mean_filt2 = pd.Series(data[:,2]).rolling(win_sz, min_periods=1).mean()
        std_filt2 = pd.Series(data[:,2]).rolling(win_sz, min_periods=1).std()
        fq_filt2 = self.maxf_filter(data[:,2], self.cfg.fs_hz['chest']['ACC'])
        mean_filt = mean_filt0 + mean_filt1 + mean_filt2
        std_filt = std_filt0 + std_filt1 + std_filt2
        return {
                f'{feature_name}Mean'  :  mean_filt,
                f'{feature_name}Std'   :  std_filt,
                f'{feature_name}0Mean' :  mean_filt0,
                f'{feature_name}0Std'  :  std_filt0,
                f'{feature_name}0Freq' :  fq_filt0,
               }

    # def wacc_designer(self, feature_name, data):
    #     win_sz = self.cfg.fs_hz['wrist']['ACC'] * self.chest_win_duration
    #     mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
    #     std_filt = pd.Series(data).rolling(win_sz, min_periods=1).std()
    #     fq_filt = self.maxf_filter(data)
    #     return {
    #             f'{feature_name}Mean' :  mean_filt,
    #             f'{feature_name}Std'  :  std_filt,
    #             f'{feature_name}Freq' :  fq_filt,
    #            }

    def wacc_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['ACC'] * self.wrist_win_duration
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                f'{feature_name}Mean' :  mean_filt,
                f'{feature_name}Max'  : max,
                f'{feature_name}Min'  : data,
               }

    def ecg_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['ECG'] * self.chest_win_duration
        mean_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        std_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).std()
               
        return {
                f'{feature_name}Mean' :  mean_filt,0

                f'{feature_name}Std'  :  std_filt  
               }


    def eda_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['EDA'] * self.chest_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
               
        return {
                f'{feature_name}Mean':  mean_filt,
                f'{feature_name}Max' : self.filt_nan(data, max_filt),
                f'{feature_name}Min' : self.filt_nan(data, min_filt),
               }


    def weda_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['EDA'] * self.wrist_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
               
        return {
                f'{feature_name}Mean' :  mean_filt,
                f'{feature_name}Max'  : self.filt_nan(data, max_filt),
                f'{feature_name}Min'  : self.filt_nan(data, min_filt),
               }


    def emg_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['EMG'] * self.chest_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                f'{feature_name}Mean' :  mean_filt,
               }


    def resp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['Resp'] * self.chest_win_duration
        win_sz = 70 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                f'{feature_name}Mean' :  mean_filt,
               }


    def temp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['Temp'] * self.wrist_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        dyn_filt = abs(max_filt - min_filt)
        grad_filt = pd.Series(data).rolling(win_sz, 
                                            min_period=2).apply(lambda v: linregress(np.arange(len(v)), v).slope )
        return {
                f'{feature_name}Mean' : mean_filt,
                f'{feature_name}Max'  : max_filt,
                f'{feature_name}Min'  : min_filt,
                f'{feature_name}Dyn'  : dyn_filt,
                f'{feature_name}Grad' : grad_filt,
               }


    def wtemp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['Temp'] * self.wrist_win_duration
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        dyn_filt = abs(max_filt - min_filt)
        grad_filt = pd.Series(data).rolling(win_sz, 
                                            min_period=2).apply(lambda v: linregress(np.arange(len(v)), v).slope )
        return {
                f'{feature_name}Mean' : mean_filt,
                f'{feature_name}Max'  : max_filt,
                f'{feature_name}Min'  : min_filt,
                f'{feature_name}Dyn'  : dyn_filt,
                f'{feature_name}Grad' : grad_filt,
               }


    def maxf_filter(self, v, fs, win_sz = 3500, skip=100):
        def dom_nonzero_freq(x, fs):
            y = np.fft.fft(x)
            yf = np.fft.fftfreq(len(x), 1/fs)
            ipart = np.argpartition(y, -2)
            f1, f2 = yf[ipart[-1]], yf[ipart[-2]] # top two dominant frequencies
            return abs(f1) if abs(f1) > 0 else abs(f2)
        signal = np.array([dom_nonzero_freq(v[i:i+win_sz], fs) for i in range(0, len(v), skip)])
        fst = fs / skip
        fsup = fs
        time_signal = np.arange(0, len(signal)*(1/fst), 1/fst)
        time_upsamp = np.arange(0, (fsup/fs)*len(signal)*(1/fsup), 1/fsup)
        sig_upsamp = np.interp(time_upsamp, time_signal, signal)
        return sig_upsamp