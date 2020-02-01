#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 17:48:25 2020

@author: tim
"""

import numpy as np
import pandas as pd


class FeatureDesigner:
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
                     'wristACC0' : self.acc_designer,
                     'wristACC1' : self.acc_designer,
                     'wristACC2' : self.acc_designer,
                     'wristEDA'  : self.eda_designer,
                     'wristTEMP' : self.temp_designer,
                     }
        return designer_map[feature_name](feature_name, data)


    def acc_designer(self, feature_name, data):
        win_sz = 40
        return {
            '{}Mean'.format(feature_name) :  pd.Series(data).rolling(win_sz).mean(),
            '{}Std'.format(feature_name)  :  pd.Series(data).rolling(win_sz).std()
        }


    def ecg_designer(self, feature_name, data):
        win_sz = 40
        return {
            '{}Mean'.format(feature_name) :  pd.Series(data).rolling(win_sz).mean(),
            '{}Std'.format(feature_name)  :  pd.Series(data).rolling(win_sz).std()
        }


    def eda_designer(self, feature_name, data):
        win_sz = 40
        return {
            '{}Mean'.format(feature_name) :  pd.Series(data).rolling(win_sz).mean(),
            '{}Std'.format(feature_name)  :  pd.Series(data).rolling(win_sz).std()
        }


    def emg_designer(self, feature_name, data):
        win_sz = 40
        return {
            '{}Mean'.format(feature_name) :  pd.Series(data).rolling(win_sz).mean(),
            '{}Std'.format(feature_name)  :  pd.Series(data).rolling(win_sz).std()
        }


    def resp_designer(self, feature_name, data):
        win_sz = 40
        return {
            '{}Mean'.format(feature_name) :  pd.Series(data).rolling(win_sz).mean(),
            '{}Std'.format(feature_name)  :  pd.Series(data).rolling(win_sz).std()
        }


    def temp_designer(self, feature_name, data):
        win_sz = 40
        return {
            '{}Mean'.format(feature_name) :  pd.Series(data).rolling(win_sz).mean(),
            '{}Std'.format(feature_name)  :  pd.Series(data).rolling(win_sz).std()
        }