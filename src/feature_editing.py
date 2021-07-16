#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 17:48:25 2020

@author: tim
"""

import numpy as np
import pandas as pd
import scipy.signal as sps
import scipy.interpolate as spi
import scipy.integrate as integrate
from scipy.stats import linregress
import matplotlib.pyplot as plt
import biosppy
import pyhrv.tools as tools
from hrvanalysis import get_geometrical_features #tinn index method is broken in pyhrv so use this


class FeatureDesigner:
    """
    * Edit feature dispatches to specific designer method based on feature name.
    * {feature_name}designer returns any number of custom features.
    """
    # for feature moving window size
    other_win_duration = 60 # seconds
    chest_win_duration = 5 # seconds
    def __init__(self, cfg_info, common_hz=700, common_len=None):
        self.cfg = cfg_info
        self.common_hz = common_hz
        self.common_len = common_len
        self.common_ts = np.arange(0, 1/(self.common_hz)*self.common_len, 1/(self.common_hz))


    def edit_feature(self, feature_name, data):
        designer_map = {
                        'chestACC'  : self.acc_designer,
                        'chestECG'  : self.ecg_designer,
                        'chestEDA'  : self.eda_designer,
                        'chestEMG'  : self.emg_designer,
                        'chestResp' : self.resp_designer,
                        'chestTemp' : self.temp_designer,
                        'wristACC'  : self.wacc_designer,
                        'wristBVP'  : self.wbvp_designer,
                        'wristEDA'  : self.weda_designer,
                        'wristTEMP' : self.wtemp_designer,
                       }
        return designer_map[feature_name](feature_name, data)


    def acc_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['ACC'] * self.chest_win_duration

        mean_filt0 = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        std_filt0 = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).std()
        fq_filt0 = self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['ACC'], win_sz, skip=100,
                                     apply=self.dom_nonzero_freq, fs_hz=self.cfg.fs_hz['chest']['ACC'])

        mean_filt1 = pd.Series(data[:,1]).rolling(win_sz, min_periods=1).mean()
        std_filt1 = pd.Series(data[:,1]).rolling(win_sz, min_periods=1).std()
        fq_filt1 = self.rolling_skip(data[:,1], self.cfg.fs_hz['chest']['ACC'], win_sz, skip=100,
                                     apply=self.dom_nonzero_freq, fs_hz=self.cfg.fs_hz['chest']['ACC'])

        mean_filt2 = pd.Series(data[:,2]).rolling(win_sz, min_periods=1).mean()
        std_filt2 = pd.Series(data[:,2]).rolling(win_sz, min_periods=1).std()
        fq_filt2 = self.rolling_skip(data[:,2], self.cfg.fs_hz['chest']['ACC'], win_sz, skip=100,
                                     apply=self.dom_nonzero_freq, fs_hz=self.cfg.fs_hz['chest']['ACC'])

        mean_filt = mean_filt0 + mean_filt1 + mean_filt2
        std_filt = std_filt0 + std_filt1 + std_filt2
        return {
                f'{feature_name}Mean'  :  mean_filt,
                f'{feature_name}Std'   :  std_filt,
                f'{feature_name}0Mean' :  mean_filt0,
                f'{feature_name}0Std'  :  std_filt0,
                f'{feature_name}0Freq' :  fq_filt0,
                f'{feature_name}1Mean' :  mean_filt1,
                f'{feature_name}1Std'  :  std_filt1,
                f'{feature_name}1Freq' :  fq_filt1,
                f'{feature_name}2Mean' :  mean_filt2,
                f'{feature_name}2Std'  :  std_filt2,
                f'{feature_name}2Freq' :  fq_filt2,
               }


    def wacc_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['ACC'] * self.other_win_duration
        max_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).min()
        mean_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        return {
                f'{feature_name}Mean' : self.upsample_data(mean_filt, self.cfg.fs_hz['wrist']['ACC'], self.common_ts),
                f'{feature_name}Max'  : self.upsample_data(max_filt,  self.cfg.fs_hz['wrist']['ACC'], self.common_ts),
                f'{feature_name}Min'  : self.upsample_data(min_filt,  self.cfg.fs_hz['wrist']['ACC'], self.common_ts),
               }


    def ecg_designer(self, feature_name, data):

        def get_hr_metrics(x, metric, sampling_rate=700):
            if metric == 'hr_mean':
                hr = biosppy.signals.ecg.ecg(x, sampling_rate=sampling_rate, show=False)['heart_rate']
                return np.mean(hr)
            if metric == 'hrv_mean':
                rpeaks =  biosppy.signals.ecg.ecg(x, sampling_rate=sampling_rate, show=False)['rpeaks']
                hrv = tools.nn_intervals(rpeaks)
                return np.mean(hrv)
            if metric == 'hr_std':
                hr = biosppy.signals.ecg.ecg(x, sampling_rate=sampling_rate, show=False)['heart_rate']
                return np.std(hr)
            if metric == 'hrv_std':
                rpeaks =  biosppy.signals.ecg.ecg(x, sampling_rate=sampling_rate, show=False)['rpeaks']
                hrv = tools.nn_intervals(rpeaks)
                return np.std(hrv)
            if metric =='tinn':
                rpeaks =  biosppy.signals.ecg.ecg(x, sampling_rate=sampling_rate, show=False)['rpeaks']
                hrv = tools.nn_intervals(rpeaks)
                return get_geometrical_features(hrv)['triangular_index']
            if metric == 'rms':
                rpeaks =  biosppy.signals.ecg.ecg(x, sampling_rate=sampling_rate, show=False)['rpeaks']
                hrv = tools.nn_intervals(rpeaks)
                return np.sqrt( (1/len(hrv)) * sum([i**2 for i in hrv]))
            return ValueError(f"Unsupported Metric {metric}")

        win_sz =   self.cfg.fs_hz['chest']['ECG'] * self.chest_win_duration
        mean_hr =  self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['ECG'], win_sz, skip=7000,
                                     apply=get_hr_metrics, metric='hr_mean')
        std_hr =   self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['ECG'], win_sz, skip=7000,
                                     apply=get_hr_metrics, metric='hr_std')
        mean_hrv = self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['ECG'], win_sz, skip=7000,
                                     apply=get_hr_metrics, metric='hrv_mean')
        std_hrv =  self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['ECG'], win_sz, skip=7000,
                                     apply=get_hr_metrics, metric='hrv_std')
        tinn =     self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['ECG'], win_sz, skip=7000,
                                     apply=get_hr_metrics, metric='tinn')
        rms =      self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['ECG'], win_sz, skip=7000,
                                     apply=get_hr_metrics, metric='rms')

        return {
                f'{feature_name}MeanHR'  :  mean_hr,
                f'{feature_name}StdHR'   :  std_hr,
                f'{feature_name}MeanHRV' :  mean_hrv,
                f'{feature_name}StdHRV'  :  std_hrv,
                f'{feature_name}TINN'    :  tinn,
                f'{feature_name}RMS'     :  rms,
                }


    def wbvp_designer(self, feature_name, data):
        def get_bvp_metrics(x, metric, sampling_rate=64):
            if metric == 'hr_mean':
                hr = biosppy.signals.bvp.bvp(x, sampling_rate=sampling_rate, show=False)['heart_rate']
                return np.mean(hr)
            if metric == 'hrv_mean':
                onsets =  biosppy.signals.bvp.bvp(x, sampling_rate=sampling_rate, show=False)['onsets']
                hrv = tools.nn_intervals(onsets)
                return np.mean(hrv)
            if metric == 'hr_std':
                hr = biosppy.signals.bvp.bvp(x, sampling_rate=sampling_rate, show=False)['heart_rate']
                return np.std(hr)
            if metric == 'hrv_std':
                onsets =  biosppy.signals.bvp.bvp(x, sampling_rate=sampling_rate, show=False)['onsets']
                hrv = tools.nn_intervals(onsets)
                return np.std(hrv)
            if metric =='tinn':
                onsets =  biosppy.signals.bvp.bvp(x, sampling_rate=sampling_rate, show=False)['onsets']
                hrv = tools.nn_intervals(onsets)
                return get_geometrical_features(hrv)['triangular_index']
            if metric == 'rms':
                onsets =  biosppy.signals.bvp.bvp(x, sampling_rate=sampling_rate, show=False)['onsets']
                hrv = tools.nn_intervals(onsets)
                return np.sqrt( (1/len(hrv)) * sum([i**2 for i in hrv]))
            return ValueError(f"Unsupported Metric {metric}")

        win_sz = self.cfg.fs_hz['wrist']['BVP'] * self.other_win_duration

        mean_hr =  self.rolling_skip(data[:,0], self.cfg.fs_hz['wrist']['BVP'], win_sz, skip=7000, upsample=False,
                                     apply=get_bvp_metrics, metric='hr_mean')
        std_hr =   self.rolling_skip(data[:,0], self.cfg.fs_hz['wrist']['BVP'], win_sz, skip=7000, upsample=False,
                                     apply=get_bvp_metrics, metric='hr_std')
        mean_hrv = self.rolling_skip(data[:,0], self.cfg.fs_hz['wrist']['BVP'], win_sz, skip=7000, upsample=False,
                                     apply=get_bvp_metrics, metric='hrv_mean')
        std_hrv =  self.rolling_skip(data[:,0], self.cfg.fs_hz['wrist']['BVP'], win_sz, skip=7000, upsample=False,
                                     apply=get_bvp_metrics, metric='hrv_std')
        tinn =     self.rolling_skip(data[:,0], self.cfg.fs_hz['wrist']['BVP'], win_sz, skip=7000, upsample=False,
                                     apply=get_bvp_metrics, metric='tinn')
        rms =      self.rolling_skip(data[:,0], self.cfg.fs_hz['wrist']['BVP'], win_sz, skip=7000, upsample=False,
                                     apply=get_bvp_metrics, metric='rms')

        return {
                f'{feature_name}MeanHR'  :  self.upsample_data(mean_hr,
                                                               self.cfg.fs_hz['wrist']['BVP'], self.common_ts),
                f'{feature_name}StdHR'   :  self.upsample_data(std_hr,
                                                               self.cfg.fs_hz['wrist']['BVP'], self.common_ts),
                f'{feature_name}MeanHRV' :  self.upsample_data(mean_hrv,
                                                               self.cfg.fs_hz['wrist']['BVP'],self.common_ts),
                f'{feature_name}StdHRV'  :  self.upsample_data(std_hrv,
                                                               self.cfg.fs_hz['wrist']['BVP'], self.common_ts),
                f'{feature_name}TINN'    :  self.upsample_data(tinn,
                                                               self.cfg.fs_hz['wrist']['BVP'], self.common_ts),
                f'{feature_name}RMS'     :  self.upsample_data(rms,
                                                               self.cfg.fs_hz['wrist']['BVP'], self.common_ts)
                }


    def eda_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['EDA'] * self.chest_win_duration
        mean_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).min()
        dyn_filt = abs(max_filt - min_filt)

        return {
                f'{feature_name}Mean' : mean_filt,
                f'{feature_name}Max'  : max_filt,
                f'{feature_name}Min'  : min_filt,
                f'{feature_name}Dyn'  : dyn_filt,
               }


    def weda_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['EDA'] * self.chest_win_duration
        mean_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).min()
        dyn_filt = abs(max_filt - min_filt)
        return {
                f'{feature_name}Mean' : self.upsample_data(mean_filt,
                                        self.cfg.fs_hz['wrist']['EDA'], self.common_ts), 
                f'{feature_name}Max'  : self.upsample_data(max_filt,
                                        self.cfg.fs_hz['wrist']['EDA'], self.common_ts), 
                f'{feature_name}Min'  : self.upsample_data(min_filt,
                                        self.cfg.fs_hz['wrist']['EDA'], self.common_ts),
                f'{feature_name}Dyn'  : self.upsample_data(dyn_filt,
                                        self.cfg.fs_hz['wrist']['EDA'], self.common_ts),
               }


    def emg_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['EMG'] * self.chest_win_duration
        mean_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        std_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).std()
        fq_filt = self.rolling_skip(data[:,0], self.cfg.fs_hz['chest']['EMG'], win_sz, skip=100,
                                    apply=self.dom_nonzero_freq, fs_hz=self.cfg.fs_hz['chest']['EMG'])
        return {
                f'{feature_name}Mean'  :  mean_filt,
                f'{feature_name}Std'   :  std_filt,
                f'{feature_name}Fq'    :  fq_filt,
               }


    def resp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['Resp'] * self.chest_win_duration
        return {
                f'{feature_name}' : data[:,0],
               }


    def temp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['chest']['Temp'] * self.chest_win_duration
        mean_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).min()
        dyn_filt = abs(max_filt - min_filt)
        return {
                f'{feature_name}Mean' : mean_filt,
                f'{feature_name}Max'  : max_filt,
                f'{feature_name}Min'  : min_filt,
                f'{feature_name}Dyn'  : dyn_filt,
               }


    def wtemp_designer(self, feature_name, data):
        win_sz = self.cfg.fs_hz['wrist']['TEMP'] * self.other_win_duration
        mean_filt = pd.Series(data[:,0]).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data[:,0]).rolling(win_sz, min_periods=0).min()
        dyn_filt = abs(max_filt - min_filt)
        return {
                f'{feature_name}Mean' : self.upsample_data(mean_filt,
                                                           self.cfg.fs_hz['wrist']['TEMP'], self.common_ts),
                f'{feature_name}Max'  : self.upsample_data(max_filt,
                                                           self.cfg.fs_hz['wrist']['TEMP'], self.common_ts),
                f'{feature_name}Min'  : self.upsample_data(min_filt,
                                                           self.cfg.fs_hz['wrist']['TEMP'], self.common_ts),
                f'{feature_name}Dyn'  : self.upsample_data(dyn_filt,
                                                           self.cfg.fs_hz['wrist']['TEMP'], self.common_ts),
               }


    def dom_nonzero_freq(self, signal, fs_hz):
        """
        Returns most dominant non-zero frequency via fft
        """
        y = np.fft.fft(signal)
        yf = np.fft.fftfreq(len(signal), 1/fs_hz)
        ipart = np.argpartition(y, -2)
        f1, f2 = yf[ipart[-1]], yf[ipart[-2]] # top two dominant frequencies
        return abs(f1) if abs(f1) > 0 else abs(f2)


    def upsample_data(self, signal, fs, custom_ts, kind='zero'):
        """
        Upsamples data to custom ts
        """
        time_signal = np.arange(0, len(signal)*(1/fs), 1/fs)
        data_lookup = spi.interp1d(time_signal, signal, kind=kind, fill_value='extrapolate')
        return data_lookup(custom_ts)


    def rolling_skip(self, data, fs, win_sz, skip=1, upsample=True, apply=None, **kwargs):
        """
        Like pd.Rolling() but with stride
        If the apply method fails for any reason, reuse last sample
        Upsamples at the end to original times.
        """
        rolling = np.zeros(len(range(0, len(data), skip)))
        for idx, i in enumerate(range(0, len(data), skip)):
            try:
                if apply is None:
                    rolling[idx] = data[i]
                else:
                    rolling[idx] = apply(data[i:i+win_sz], **kwargs)
            except ValueError:
                rolling[idx] = rolling[idx-1]
        return (self.upsample_data(rolling, fs/skip, 
                                   np.arange(0, len(data)*(1/fs), 1/fs)) if upsample else rolling)
