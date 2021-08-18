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

os.chdir("/home/lei/leo/code/ml")
from models import Mod9 as MyModel  # KEY: fixes which model in models to train


# Helper functions
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

def show_out(data):
    """
    Helper function.
    """
    print(f"\n\n{data}\n\n")


# Training and data
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


class BaseTrain():
    """
    Base class to train models with.
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
        self.targets = find_output_names(model)  # in the order of the model's outputs in its construction
        self.dataset = Getdata.get_dataset()
        self.dataset_size = Getdata.size()
        self.model = model
        self.losses = {key:self.losses[key] for key in self.losses.keys() if key in self.targets}
        self.metrics = {key:self.metrics[key] for key in self.metrics.keys() if key in self.targets}
    
    def get_train_set(self, train_split=0.9, seed=None, shuffle_size=25000, predict_size=20):
        """
        Selects from the plain dataset which variables
        to use and returns the dataset split.
        """
        assert train_split < 1
        def group_target(train_element, target_element):
            return ({"input": train_element},
                    {key:target_element[key] for key in self.targets})

        dataset = self.dataset.shuffle(shuffle_size, seed=seed)
        train_size = int(train_split * self.dataset_size)
        train_set = dataset.take(train_size)    
        test_set = dataset.skip(train_size)
        predict_set = test_set.take(predict_size)  

        train_set = train_set.map(group_target).repeat()
        test_set = test_set.map(group_target).batch(1)

        return train_set, test_set, predict_set

    def show_predictions(self, predict_set, space_num=45, padding=20):
        """
        Prints out target-prediction on the predict_set.
        """
        predict_list = []
        labels = self.targets

        for i in predict_set:
            print("===================")
            predict = self.model(tf.expand_dims(i[0], axis=0), training=False)
            print(" " * (padding + 2) + "target" + "   " + "-" * (space_num - 3) + "   " + "prediction")

            for j in range(len(i[1])):  # over outputs 
                index_name = labels[j] + " " * (padding - len(labels[j])) + ":"
                for k in range(i[1][j].shape[0]):  # over output dim
                    target_str = str(i[1][j][k].numpy())
                    space_mul = space_num - len(target_str)
                    predict_str = str(predict[j][0][k].numpy())
                    print(f"{index_name} index {k}: " + target_str + " " * space_mul + predict_str)
                    if k == 0:
                        index_name = " " * (padding + 1)
            print("===================")
            predict_list.append((i, predict))

        return predict_list

    def train_model(self, train_set, test_set, predict_set, **kwargs):
        """
        takes in kwargs to compile and train - uses tesnroboard.
        Feed in epochs, batchsize, weights, adam settings,
        """

        epochs = kwargs.pop("epochs")
        batchsize = kwargs.pop("batchsize")
        weights = kwargs.pop("weights")  # list
        steps_per_epoch = (self.dataset_size // batchsize) + 1
        adam = tf.keras.optimizers.Adam(**kwargs)

        train_set = train_set.batch(batchsize)

        self.model.compile(
        optimizer=adam, 
        loss = self.losses, 
        metrics = self.metrics,
        loss_weights = {self.targets[i]: weights[i] for i in range(len(weights))}  # do weights
        )

        run_name = self.name + time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        logdir = log_path / run_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        history = self.model.fit(train_set, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])
        eval_data = self.model.evaluate(test_set)
        self.model.save(weights_path)

        _ = run_name + ".txt"
        with open(text_path / _, "w") as f:
            with redirect_stdout(f):
                print(f"Model Version: {self.name}")
                print("Predictions:\n")
                _ = self.show_predictions(predict_set)
                print(f"\n\n{self.model.summary()}\n")
                print(f"\n\nHyperparameters: {kwargs}\n")
                print(f"\n\nHistory: {history.history}\n")
                print(f"\n\n Display Labels: {self.model.metrics_names}")
                print(f"\n\nEval Data: {eval_data}\n")

        return history



if __name__ == '__main__':

    filenames = ["/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-06-17_11:21:17.tfrecords", 
                     "/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-08-11_17:23:00.tfrecords",
                     "/home/lei/leo/code/data/train_data/train_type_2_2021-08-13_21:54:24.tfrecords"]

    weights_path = Path("/home/lei/leo/code/data/saved_weights/mymodel.h5")  # FIXME: alter with different models
    pickle_path = Path("/home/lei/leo/code/data/misc/out_data")
    log_path = Path("/home/lei/leo/code/data/logs/fit")
    image_path = Path("/home/lei/leo/code/data/images")
    text_path = Path("/home/lei/leo/code/data/text")

    hyper_parameters = [
        {"learning_rate":0.0005, "beta_1":0.9, "beta_2":0.999, 
        "epsilon":1e-04, "amsgrad":False, "name":"adam", "epochs":20,
        "batchsize":128, "weights":[1,1e4]}
        ]

    for i in hyper_parameters:
        Getdata = GetData(filenames)
        Model = MyModel()
        Basetrain = BaseTrain(Model, Getdata)
        train_set, test_set, predict_set = Basetrain.get_train_set()

        _ = Basetrain.train_model(train_set, test_set, predict_set, **i)
