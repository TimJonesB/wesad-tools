#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 17:48:25 2020

@author: tim
"""

import numpy as np
import pandas as pd


class FeatureDesigner:
    """
    * Edit feature dispatches to specific designer method based on feature name.
    * {feature_name}designer returns any number of custom features.
    """
    chest_win_size = 25
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
        win_sz = self.chest_win_size 
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Max'.format(feature_name) : max,
                '{}Min'.format(feature_name) : data,
               }


    def wacc_designer(self, feature_name, data):
        win_sz = 5 
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Max'.format(feature_name) : max,
                '{}Min'.format(feature_name) : data,
               }

    def ecg_designer(self, feature_name, data):
        win_sz = self.chest_win_size 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Std'.format(feature_name)  :  std_filt  
               }


    def eda_designer(self, feature_name, data):
        win_sz = self.chest_win_size 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Max'.format(feature_name)  : self.filt_nan(data, max_filt),
                '{}Min'.format(feature_name)  : self.filt_nan(data, min_filt),
               }


    def weda_designer(self, feature_name, data):
        win_sz = 3 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
                '{}Max'.format(feature_name)  : self.filt_nan(data, max_filt),
                '{}Min'.format(feature_name)  : self.filt_nan(data, min_filt),
               }


    def emg_designer(self, feature_name, data):
        win_sz = self.chest_win_size 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
               }


    def resp_designer(self, feature_name, data):
        win_sz = 70 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
               
        return {
                '{}Mean'.format(feature_name) :  mean_filt,
               }


    def temp_designer(self, feature_name, data):
        win_sz = self.chest_win_size
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        return {
                '{}Mean'.format(feature_name) : mean_filt,
                '{}Max'.format(feature_name)  : data,
                '{}Min'.format(feature_name)  : data,
               }


    def wtemp_designer(self, feature_name, data):
        win_sz = 8 
        mean_filt = pd.Series(data).rolling(win_sz, min_periods=1).mean()
        max_filt  = pd.Series(data).rolling(win_sz, min_periods=0).max()
        min_filt  = pd.Series(data).rolling(win_sz, min_periods=0).min()
        return {
                '{}Mean'.format(feature_name) : mean_filt,
                '{}Max'.format(feature_name)  : data,
                '{}Min'.format(feature_name)  : data,
               }

    
