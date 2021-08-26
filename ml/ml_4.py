"""
Hyperparameter tuning with keras-tuner.
"""
from functools import total_ordering
from textwrap import indent
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.lib.shape_base import _kron_dispatcher
import pandas as pd
from pathlib import Path
from tensorflow._api.v2 import data
from tensorflow.keras import callbacks, losses
from tensorflow.python.framework.op_def_registry import get
from tensorflow.python.keras import models
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.utils.generic_utils import default
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.python.ops.gen_math_ops import log, sign
import tensorflow as tf
import keras_tuner as kt
import tensorboard

import pickle
import time
from contextlib import redirect_stdout

import metascint.ray_tracing.python.data_analysis as dp
import metascint.ray_tracing.python.vis as dpv
import metascint.ray_tracing.python.timing_model as tm
import metascint.ray_tracing.python.circuit_signal as cs
from tensorflow.python.training.input import batch

name = 'ModB_try'

def get_dataset(filenames):

    def parse_TFR_element(element):
        """
        Just total energy, energy share %, process type
        with factors of 1e2, 1e2, 1
        """

        features = {
            "signal": tf.io.FixedLenFeature([], tf.string),
            "first_photon_time": tf.io.FixedLenFeature([], tf.string),
            "total_energy": tf.io.FixedLenFeature([], tf.string),
            "energy_share": tf.io.FixedLenFeature([], tf.string),
            "primary_gamma_pos": tf.io.FixedLenFeature([], tf.string),
            "process": tf.io.FixedLenFeature([], tf.string)
        }

        data = tf.io.parse_single_example(element, features)

        signal = tf.io.parse_tensor(data["signal"], out_type=tf.float64)
        signal = tf.reshape(signal, shape=[2510, 1])  # for 1D conv

        first_photon_time = tf.io.parse_tensor(data["first_photon_time"], out_type=tf.float64)
        first_photon_time = tf.reshape(first_photon_time, shape=[1])
        first_photon_time = first_photon_time 

        total_energy = tf.io.parse_tensor(data["total_energy"], out_type=tf.float64)
        total_energy = tf.reshape(total_energy, shape=[1])
        total_energy = total_energy * 1e2
    
        energy_share = tf.io.parse_tensor(data["energy_share"], out_type=tf.float64)  
        energy_share = tf.reshape(energy_share, shape=[3])
        energy_share = (energy_share / total_energy) * 1e2

        primary_gamma_pos = tf.io.parse_tensor(data["primary_gamma_pos"], out_type=tf.float64)
        primary_gamma_pos = tf.reshape(primary_gamma_pos, shape=[3])

        process = tf.io.parse_tensor(data["process"], out_type=tf.int64)
        process = tf.reshape(process, shape=[3])
        process = tf.cast(process, tf.float64)

        return (signal, {"total_energy": total_energy,
                        "energy_share": energy_share,
                        "process": process})
                
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_TFR_element)
    return dataset

def get_size(dataset):
        count = 0
        iterator = iter(dataset)
        while True:
            try:
                _ = next(iterator)
                count = count + 1
            except:
                break
        return count

def modB(hp):

    dropout = hp.Float('dropout', min_value=0.01, max_value=0.7, default=0.05, step=0.01)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG', default=1e-3)
    beta_1 = hp.Float('beta_1', min_value=0.8, max_value=0.95, default=0.9, step=0.05)
    beta_2 = hp.Float('beta_2', min_value=0.99, max_value=0.9999, default=0.999, step=0.0005)
    epsilon = hp.Choice('epsilon', values=[1e-5, 1e-6, 1e-7, 1e-8], default=1e-7)

    input = tf.keras.Input(shape=(2510, 1), name="input")  # (batch, 2510, 1) 

    out = tf.keras.layers.Flatten()(input)
    out = tf.keras.layers.Dense(1028, activation='relu', kernel_regularizer='l2')(out)
    out = tf.keras.layers.Dense(1028, activation='relu', kernel_regularizer='l2')(out)
    out = tf.keras.layers.Dropout(dropout)(out)
    out = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(out) 
    out = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')(out) 
    out = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')(out)

    total_energy = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer='l2', name='total_energy')(out)
    energy_share = tf.keras.layers.Dense(3, activation='relu', kernel_regularizer='l2', name='energy_share')(out)
    process = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer='l2', name='process')(out)
    model = tf.keras.Model(inputs=[input], outputs=[total_energy, energy_share, process])

    adam = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )


    model.compile(
        optimizer=adam, 
        loss = {'total_energy':tf.keras.losses.MeanSquaredError(), 
                'energy_share':tf.keras.losses.MeanSquaredError(), 
                'process':tf.keras.losses.CategoricalCrossentropy()}, 
        metrics = {"total_energy": tf.keras.metrics.MeanSquaredError(),
                    "energy_share": tf.keras.metrics.MeanSquaredError(),
                    "process": tf.keras.metrics.CategoricalAccuracy()},
        loss_weights = {'total_energy':1, 'energy_share':1, 'process':8e2}  # do weights
        )

    return model

if __name__ == '__main__':

    text_path = Path("/home/lei/leo/code/data/text")

    filenames = ["/home/lei/leo/code/data/new_train_data/new_train_metascint_type_2_2021-06-17_11:21:17.tfrecords", 
                 "/home/lei/leo/code/data/new_train_data/new_train_metascint_type_2_2021-08-11_17:23:00.tfrecords",
                 "/home/lei/leo/code/data/new_train_data/new_train_type_2_2021-08-13_21:54:24.tfrecords"]

    val_filenames = ['/home/lei/leo/code/data/val_data/train_type_2_2021-08-24_01:21:42.tfrecords']

    batchsize = 128

    dataset = get_dataset(filenames)
    data_size = get_size(dataset)
    dataset = get_dataset(filenames)
    steps_per_epoch = (data_size // batchsize) + 1

    val_dataset = get_dataset(val_filenames)
    val_data_size = get_size(val_dataset)
    val_dataset = get_dataset(val_filenames)
    val_steps_per_epcoh = (val_data_size // batchsize) + 1

    dataset = dataset.shuffle(25000).repeat().batch(batchsize)
    val_dataset = val_dataset.shuffle(10000).repeat().batch(batchsize)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    tuner = kt.Hyperband(hypermodel=modB,
                        objective='val_loss',
                        max_epochs=50,
                        hyperband_iterations=1,
                        project_name=f'/home/lei/leo/code/data/hps/{name}',
                        overwrite=True)

    print(tuner.search_space_summary())

    tuner.search(dataset,
                steps_per_epoch=steps_per_epoch,
                callbacks=[stop_early],
                validation_data=val_dataset,
                validation_steps=val_steps_per_epcoh,
                epochs=40,
                shuffle=True,
                verbose=2,
                initial_epoch=0
                )

    print('\n\n\n')
    print(tuner.results_summary())
    print('\n\n\n')
    print(tuner.get_best_hyperparameters()[0].values)
    print('\n\n\n')

    run_name = 'hps_' + name + time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()) + ".txt"

    with open(text_path / run_name, "w") as f:
        with redirect_stdout(f):
            print(f"Model Version: {name}")
            print(f"Dataset Size: {data_size}")
            print(f"Val Dataset Size: {val_data_size}")
            print(tuner.search_space_summary())
            print('\n\n\n')
            print(tuner.results_summary())
            print('\n\n\n')
            print(tuner.get_best_hyperparameters()[0].values)
            print('\n\n\n')
