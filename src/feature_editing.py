#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 17:48:25 2020

@author: tim
"""

import numpy as np

import pandas

class FeatureDesigner:
    
                 
    def edit_feature(data,feature_name):
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
                     'wristBVP'  : self.bvp_designer,
                     'wristEDA'  : self.eda_designer,
                     'wristTEMP' : self.temp_designer,
                     }
        return designer_map[feature_name](data)

    def acc_designer(data):
        return 2

    def ecg_designer(data):
        None
        
    def eda_designer(data):
        None
    
    def emg_designer(data):
        None
    
    def resp_designer(data):
        None

    def temp_designer(data):
        None
    
    def bvp_designer(data):
        None
    
        

    