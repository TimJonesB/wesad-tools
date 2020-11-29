#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:20:26 2019

@author: tim
"""

import os
import keras as k
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import numpy as np
from data_loading import DataHandler


def config_gpu(my_config):
    gpus = my_config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                my_config.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def build_model(data_dim, timesteps, num_classes):
    layer_width = 32 
    model = Sequential()
    model.add(LSTM(layer_width, return_sequences=False,
                   input_shape=(timesteps, data_dim)))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


if __name__ == '__main__':

    # Uncomment data-labels to be trained
    data_features = [
                     'chestACC0Mean',
                    #  'chestACC0Std',
                    #  'chestACC0Max',
                    #  'chestACC0Min',
                     'chestACC1Mean',
                    #  'chestACC1Std',
                    #  'chestACC1Max',
                    #  'chestACC1Min',
                     'chestACC2Mean',
                    #  'chestACC2Std',
                    #  'chestACC2Max',
                    #  'chestACC2Min',
                    #  'chestECGMean',
                    #  'chestECGStd',
                     'chestEDAMean',
                    #  'chestEDAStd',
                    #  'chestEDAMax',
                    #  'chestEDAMin',
                    #  'chestEDADr',
                    #  'chestEMGMean',
                    #  'chestEMGStd',
                    #  'chestRespMean',
                    #  'chestRespStd',
                    #  'chestTempMean',
                    #  'chestTempStd',
                    #  'chestTempMax',
                    #  'chestTempMin',
                    #  'chestTempDr',
                     'wristACC0Mean',
                    #  'wristACC0Std',
                     'wristACC1Mean',
                    #  'wristACC1Std',
                     'wristACC2Mean',
                    #  'wristACC2Std',
                     'wristEDAMean',
                    #  'wristEDAStd',
                    #  'wristTEMPMean',
                    #  'wristTEMPStd',
                   ]

    data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                                      '..', 
                                                                      'data', 
                                                                      'formatted_data_feat.h5'))
    handler = DataHandler(data_file)

    config_gpu(tf.config.experimental)

    print('Building Model...')

    model = build_model(data_dim=len(data_features), timesteps=handler.timesteps, num_classes=2)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Generating Datasets...')

    data = handler.get_data_sets(data_features=data_features, timesteps=handler.timesteps)

    print('Fitting Model...')

    class_weight_list = handler.compute_class_weights(data['y_train'])
    class_weight_dict = {i : class_weight_list[idx] for idx,i in enumerate(set(data['y_train']))}
    print(class_weight_list)                                                        

    model.fit(data['x_train'],
              k.utils.to_categorical(data['y_train']),
              batch_size=128,
              epochs=20,
              validation_data=(data['x_test'],k.utils.to_categorical(data['y_test'])),
              class_weight=class_weight_dict)