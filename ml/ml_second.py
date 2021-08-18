"""
Try to rewrite ml_first in terms of objects
"""
from leo.multiwave_metascint.ml.ml_first import get_dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.lib.shape_base import _kron_dispatcher
import pandas as pd
from pathlib import Path
from tensorflow._api.v2 import data
from tensorflow.keras import losses
from tensorflow.python.framework.op_def_registry import get
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.python.ops.gen_math_ops import log, sign
import tensorflow as tf
import tensorboard

import pickle
import time
from contextlib import redirect_stdout

import metascint.ray_tracing.python.data_analysis as dp
import metascint.ray_tracing.python.vis as dpv
import metascint.ray_tracing.python.timing_model as tm
import metascint.ray_tracing.python.circuit_signal as cs

class GetData:
    """
    Extracts the plain dataset to train on.
    Also finds dataset size.
    """
    def __init__(self, filenames):
        self.filenames = filenames

    def parse_TFR_element(self, element):
        """
        From dataset = tf.data.TFRecordDataset(filename)
        extracts the original elements.
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
    
        energy_share = tf.io.parse_tensor(data["energy_share"], out_type=tf.float64)  
        energy_share = tf.reshape(energy_share, shape=[3])  

        primary_gamma_pos = tf.io.parse_tensor(data["primary_gamma_pos"], out_type=tf.float64)
        primary_gamma_pos = tf.reshape(primary_gamma_pos, shape=[3])

        process = tf.io.parse_tensor(data["process"], out_type=tf.int64)
        process = tf.reshape(process, shape=[2])
        process = tf.cast(process, tf.float64)

        return (signal, {"first_photon_time": first_photon_time, 
                        "total_energy": total_energy,
                        "energy_share": energy_share,
                        "primary_pos": primary_gamma_pos, 
                        "process": process}) 

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self.parse_TFR_element)
        return dataset
    
    def get_size(self):
        dataset = self.get_dataset()
        count = 0
        iterator = iter(dataset)
        while True:
            try:
                _ = next(iterator)
                count = count + 1
            except:
                break
        return count


def find_output_names(model):
    """
    Helper function to extract output names
    in a model
    """
    names = []
    for output in model.output:
        name = output.name.split('/')[0]
        names.append(name)
    return names


class BaseModel():
    """
    Base Model to build train models with
    """

    labels = ["first_photon_time", "total_energy", 
               "energy_share", "primary_pos", "process"]
    losses = {"first_photon_time": tf.keras.losses.MeanSquaredError(), "total_energy": tf.keras.losses.MeanSquaredError(),
              "energy_share": tf.keras.losses.MeanSquaredError(), "primary_pos": tf.keras.losses.MeanSquaredError(), 
              "process": tf.keras.losses.CategoricalCrossentropy()} 
    metrics = {"first_photon_time": tf.keras.metrics.MeanSquaredError(),"total_energy": tf.keras.metrics.MeanSquaredError(),
               "energy_share": tf.keras.metrics.MeanSquaredError(), "primary_pos": tf.keras.metrics.MeanSquaredError(), 
               "process": tf.keras.metrics.CategoricalAccuracy()}

    def __init__(self, Model, Getdata):
        """
        Records the model's name, which target variables
        (in list) to pick

        Model is a model class, 
        """
        model = Model.model()
        self.name = Model.name
        self.targets = find_output_names(model)
        self.dataset = Getdata.get_dataset()
        self.dataset_size = Getdata.size()
        self.model = model
        self.losses = {key:self.losses[key] for key in self.losses.keys() if key in self.targets}
        self.metrics = {key:self.metrics[key] for key in self.metrics.keys() if key in self.targets}
    
    def get_train_set(self):
        """
        Selects from the plain dataset which variables
        to use.
        """
        def group_target(train_element, target_element):
            return ({"input": train_element},
                    {key:target_element[key] for key in self.targets})
        return self.dataset.map(group_target)

    def train_model(self, **kwargs):
        """
        takes in kwargs to compile and train - uses tesnroboard
        """
        
        epochs = kwargs.pop("epochs")
        batchsize = kwargs.pop("batchsize")
        adam = tf.keras.optimizers.Adam(**kwargs)

        self.model.compile(
        optimizer=adam, 
        loss = self.losses, 
        metrics = self.metrics,
        loss_weights = {"energy_share": 1, "process": 10}  # do weights
        )

        run_name = self.name + time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        logdir = log_path / run_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        history = model.fit(train_set, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])
        return history


class Mod9():

    def __init__(self):
        self.name = 'mod9_'

    def model(self):
        input = tf.keras.Input(shape=(2510, 1), name="input")  # (batch, 2510, 1) 
        out = tf.keras.layers.Flatten()(input)
        out = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer='l2')(out)
        energy_share = tf.keras.layers.Dense(3, activation='relu', kernel_regularizer='l2', name='energy_share')(out)
        process = tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer='l2', name='process')(out)


        return tf.keras.Model(inputs=[input], outputs=[energy_share, process])

    
    
    
if __name__ == '__main__':

    filenames = ["/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-06-17_11:21:17.tfrecords", 
                     "/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-08-11_17:23:00.tfrecords",
                     "/home/lei/leo/code/data/train_data/train_type_2_2021-08-13_21:54:24.tfrecords"]

    weights_path = Path("/home/lei/leo/code/data/saved_weights/mymodel.h5")  # FIXME: alter with different models
    pickle_path = Path("/home/lei/leo/code/data/misc/out_data")
    log_path = Path("/home/lei/leo/code/data/logs/fit")
    image_path = Path("/home/lei/leo/code/data/images")
    text_path = Path("/home/lei/leo/code/data/text")
