from functools import total_ordering
from textwrap import indent
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.lib.shape_base import _kron_dispatcher
import pandas as pd
from pathlib import Path
from tensorflow._api.v2 import data
from tensorflow.keras import losses
from tensorflow.python.framework.op_def_registry import get
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.core import Dropout
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


def modA(hp):

    input = tf.keras.Input(shape=(2510, 1), name="input")  # (batch, 2510, 1) 

    out = tf.keras.layers.Flatten()(input)
    out = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer='l2')(out)
    out = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer='l2')(out)

    total_energy =tf.keras.layers.Dense(1, activation='relu', kernel_regularizer='l2', name='energy_share')(out)
    energy_share = tf.keras.layers.Dense(3, activation='relu', kernel_regularizer='l2', name='energy_share')(out)
    process = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer='l2', name='process')(out)

    model = tf.keras.Model(inputs=[input], outputs=[total_energy, energy_share, process])
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
        )


    model.compile(
        optimizer=adam, 
        loss = {'total_energy':tf.keras.losses.MeanSquaredError(), 
                'energy_share':tf.keras.losses.MeanSquaredError(), 
                'process':tf.keras.losses.CategoricalCrossentropy()}, 
        metrics = {"total_energy": tf.keras.metrics.MeanSquaredError(),
                    "energy_share": tf.keras.metrics.MeanSquaredError(),
                    "process": tf.keras.metrics.CategoricalAccuracy()},
        loss_weights = {'total_energy':1, 'energy_share':1, 'process':1e3}  # do weights
        )
    return model

tuner = kt.Hyperband(modA,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dirA',
                     project_name='intro_to_ktA')


