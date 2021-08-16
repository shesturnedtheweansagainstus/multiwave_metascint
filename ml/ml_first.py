import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.lib.shape_base import _kron_dispatcher
import pandas as pd
from pathlib import Path
from tensorflow._api.v2 import data
from tensorflow.python.framework.op_def_registry import get
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.python.ops.gen_math_ops import log, sign
import tensorflow as tf
import tensorboard

import pickle
import datetime
from contextlib import redirect_stdout

import metascint.ray_tracing.python.data_analysis as dp
import metascint.ray_tracing.python.vis as dpv
import metascint.ray_tracing.python.timing_model as tm
import metascint.ray_tracing.python.circuit_signal as cs

os.chdir("/home/lei/leo/code/ml")
from models import Mod8 as MyModel  # KEY: fixes which model in models to train

"""
TensorBoard:  (command line)

rm -rf ./logs/

tensorboard --logdir logs/fit
tensorboard --logdir /home/lz1919/Documents/UNI/year_two/multiwave/code/leo/data/logs/fit


tensorboard dev upload \
  --logdir logs/fit \
  --name "(optional) My latest experiment" \
  --description "(optional) Simple comparison of several hyperparameters" \
  --one_shot

tensorboard dev delete --experiment_id EXPERIMENT_ID
"""

# get dataset
def parse_TFR_element(element):
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
    first_photon_time = first_photon_time * 1e11  # in 1e-11 sec

    total_energy = tf.io.parse_tensor(data["total_energy"], out_type=tf.float64)
    total_energy = tf.reshape(total_energy, shape=[1])
    total_energy = total_energy * 100
 
    energy_share = tf.io.parse_tensor(data["energy_share"], out_type=tf.float64)  
    energy_share = tf.reshape(energy_share, shape=[3])  
    energy_share = energy_share * 100

    primary_gamma_pos = tf.io.parse_tensor(data["primary_gamma_pos"], out_type=tf.float64)
    primary_gamma_pos = tf.reshape(primary_gamma_pos, shape=[3])
    primary_gamma_pos = tf.concat([(primary_gamma_pos[:2] / 3.2) * 100, tf.reshape(primary_gamma_pos[2], [1])], axis=-1)  # scale x,y to %

    process = tf.io.parse_tensor(data["process"], out_type=tf.int64)
    process = tf.reshape(process, shape=[2])
    process = tf.cast(process, tf.float64)


    # Write function to map over and take the relevent variables
    #return (signal, (first_photon_time, total_energy, energy_share, primary_gamma_pos, process))  # [11] 234 8910
    return (signal, (first_photon_time, total_energy, energy_share, process))  # [11] 234 8910

def get_dataset(filenames):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(parse_TFR_element)
  return dataset

def select_relevent_var(relevent_var):
    """
    Returns a function to .map over a dataset to select relevent 
    variables from ["first_photon_time", "total_energy", 
    "energy_share", "primary_gamma_pos", "process"]
    """

    var = [
        "first_photon_time", "total_energy", "energy_share",
        "primary_gamma_pos", "process"
        ]
    var_index = [var.index(i) for i in relevent_var]

    # elements are in order of relevent_var
    def mapping(element):
        return(element[0], tuple([element[1][i] for i in var_index]))

    return mapping

def set_target_shape(dataset):
    """
    REDUNDANT
    Concats target variables to one target vector
    """
    def concat_target(train_element, target_element):
        target_element = tf.concat([
            target_element[i] for i in range(len(target_element))
        ], axis=-1)
        return (train_element, target_element)
    return dataset.map(concat_target)

def group_target_shape(dataset):
    """
    Formats elements in the dataset to appear in a dictionary
    for multiple outputs.
    """
    def group_target(train_element, target_element):
        """
        return ({"input": train_element},

                {"first_photon_time": target_element[0], 
                "total_energy": target_element[1],
                "energy_share": target_element[2],
                "primary_pos": target_element[3], 
                "process": target_element[4]})
        """
        return ({"input": train_element},
                {"first_photon_time": target_element[0], 
                "total_energy": target_element[1],
                "energy_share": target_element[2],
                "process": target_element[3]})
    return dataset.map(group_target)

def split_dataset(dataset, size, train_split=0.9, shuffle=True, shuffle_size=15000, predict_size=10):  
    """
    Extracts the train/test datasets; val later
    Try to have shuffle_size > size
    """
    assert train_split <= 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        dataset = dataset.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * size)
    
    train_ds = dataset.take(train_size)    
    test_ds = dataset.skip(train_size)
    predict_ds = test_ds.take(predict_size)  # takes some elements from test set to show predictions
    
    return train_ds, test_ds, predict_ds


# training
def get_run_name(**kwargs):
    """
    Generates a run name for given hyperparameters to log on 
    TensorBoard.
    """
    name = "".join([f"{key}{kwargs[key]}_" for key in kwargs.keys()])
    return name[:-1]

def train_model(train_set, model, steps_per_epoch, run_name, **kwargs):  # FIXME: hardcoded optimizer type
    epochs = kwargs.pop("epochs")
    adam = tf.keras.optimizers.Adam(**kwargs)
    """
    model.compile(
        optimizer=adam, 
        loss = {"first_photon_time": tf.keras.losses.MeanSquaredError(), "total_energy": tf.keras.losses.MeanSquaredError(),
                "energy_share": tf.keras.losses.MeanSquaredError(), "primary_pos": tf.keras.losses.MeanSquaredError(), 
                "process": tf.keras.losses.CategoricalCrossentropy()}, 
        metrics = {"first_photon_time": tf.keras.metrics.MeanSquaredError(),"total_energy": tf.keras.metrics.MeanSquaredError(),
                    "energy_share": tf.keras.metrics.MeanSquaredError(), "primary_pos": tf.keras.metrics.MeanSquaredError(), 
                    "process": tf.keras.metrics.CategoricalAccuracy()}
        )
    """
    model.compile(
        optimizer=adam, 
        loss = {"first_photon_time": tf.keras.losses.MeanSquaredError(), "total_energy": tf.keras.losses.MeanSquaredError(),
                "energy_share": tf.keras.losses.MeanSquaredError(), "process": tf.keras.losses.CategoricalCrossentropy()}, 
        metrics = {"first_photon_time": tf.keras.metrics.MeanSquaredError(),"total_energy": tf.keras.metrics.MeanSquaredError(),
                    "energy_share": tf.keras.metrics.MeanSquaredError(), "process": tf.keras.metrics.CategoricalAccuracy()},
        loss_weights = {"first_photon_time": 1, "total_energy": 1,
                        "energy_share": 1, "process": 4}
        )
    logdir = log_path / run_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    history = model.fit(train_set, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])
    return history

def find_size_of_dataset(dataset):
    count = 0
    iterator = iter(dataset)
    while True:
        try:
            element = next(iterator)
            count = count + 1
        except:
            break
    return count

def show_out(data):
    print(f"\n\n{data}\n\n")

def train_and_test_model(dataset_paths, weights_path, pickle_path, batchsize=64, **kwargs):

    dataset = get_dataset(dataset_paths)
    size = find_size_of_dataset(dataset)  # 7181
    print(f"Dataset size is: {size}")
    dataset = get_dataset(dataset_paths)
    train_set, test_set, predict_set = split_dataset(dataset, size=size)  

    train_set = group_target_shape(train_set)
    test_set = group_target_shape(test_set)

    train_set = train_set.repeat().batch(batchsize)  # change
    test_set = test_set.batch(1)

    _ = MyModel()
    model = _.model()
    model_version = _.name
    run_name = model_version + get_run_name(**kwargs)

    steps_per_epoch = (size // batchsize) + 1
    history = train_model(train_set, model, steps_per_epoch, run_name, **kwargs)

    print("\nModel Summary: \n")
    print(model.summary())
    print("\n")
    
    try:
        eval_data = model.evaluate(test_set)
        show_out(eval_data)
    except:
        show_out(history.history)
    else:
        show_out(eval_data)
        show_out(history.history)
    
    print(f"\n\nHyperparameters: {kwargs}\n")

    # Saves Image of the input signal for the first 
    # predict_data element and text of 
    # hyperparameters etc. 
    
    _ = run_name + ".txt"
    with open(text_path / _, "w") as f:
        with redirect_stdout(f):
            print(f"Model Version: {model_version}")
            print("Predictions:\n")
            _ = show_predictions(model, predict_set)
            print(f"\n\n{model.summary()}\n")
            print(f"\n\nHyperparameters: {kwargs}\n")
            print(f"\n\nHistory: {history.history}\n")
            print(f"\n\n Display Labels: {model.metrics_names}")
            print(f"\n\nEval Data: {eval_data}\n")

    time = np.asarray([i*100e-12 for i in range(2510)])
    plt.plot(time, _[0][0][0][:, 0])
    _ = run_name + ".png"
    plt.savefig(image_path / _, dpi=1000)

    model.save(weights_path)  # use .h5 format  FIXME: alter for different hyperparameters
    pickle.dump([history.history, eval_data], open(pickle_path, "wb"))
    return (history, eval_data)

def show_predictions(model, dataset, space_num=45, padding=20):
    """
    Formats and prints out a comparision of target-predictions
    """

    predict_list = []
    #labels = ["first_photon_time", "total_energy", "energy_share", "primary_pos", "process"]
    labels = ["first_photon_time", "total_energy", "energy_share", "process"]


    for i in dataset:
        print("===================")
        predict = model(tf.expand_dims(i[0], axis=0), training=False)
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

def prediction_model(dataset_path, weights_path, predict_num=10, shuffle_size=15000, space_num=45, padding=20):
    """
    Gives comparision between target data and model predictions.

    """
    model = tf.keras.models.load_model(weights_path)

    dataset = get_dataset(dataset_path)
    predict_data = dataset.shuffle(
        shuffle_size, seed=5
        ).take(predict_num)

    predict_list = show_predictions(model, predict_data, space_num=space_num, padding=padding)

    return predict_list
    
def location_types(data_path):
    """
    Finds the distribution of primary processess in the
    dataset.
    """
    data = pd.read_csv(data_path)
    counts = {"PhotoElectric":0, "Compton":0, "RayleighScattering":0}
    list_of_events = find_list_of_events(data_path)
    for i in list_of_events.keys():
        hits = data[data["runID"] == i]
        try:
            process = hits[hits["parentID"] == 0].sort_values("time").iloc[0]["processName"]
            counts[process] = counts[process] + 1
        except:
            continue
    return counts

"""
TODO: 
    Generate more data
    Finish presentation
    Build new designs - read!
    evergy sharing
    
    https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams

"""

"""dataset_path
data_path = Path(
    "/home/lz1919/Documents/UNI/year_two/multiwave/code/metascint_gvanode/output/metascint_type_2_2021-06-17_11:21:17.csv"
    )

dataset_path = Path(
    "/home/lz1919/Documents/UNI/year_two/multiwave/code/leo/data/test_6_metascint_type_2_2021-06-17_11:21:17.tfrecords"
    )

weights_path = Path(
    "/home/lz1919/Documents/UNI/year_two/multiwave/code/leo/data/weights.tf"
    )

pickle_path = Path(
    "/home/lz1919/Documents/UNI/year_two/multiwave/code/leo/data/out_data"
)

log_path = Path(
    "/home/lz1919/Documents/UNI/year_two/multiwave/code/leo/data/logs/fit"
)
"""


if __name__ == "__main__":
    
    dataset_paths = ["/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-06-17_11:21:17.tfrecords", 
                     "/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-08-11_17:23:00.tfrecords",
                     "/home/lei/leo/code/data/train_data/train_type_2_2021-08-13_21:54:24.tfrecords"]

    weights_path = Path("/home/lei/leo/code/data/saved_weights/mymodel.h5")  # FIXME: alter with different models
    pickle_path = Path("/home/lei/leo/code/data/misc/out_data")
    log_path = Path("/home/lei/leo/code/data/logs/fit")
    image_path = Path("/home/lei/leo/code/data/images")
    text_path = Path("/home/lei/leo/code/data/text")

    
    hyper_parameters = [
        {"learning_rate":0.0005, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-04, "amsgrad":False, "name":"adam", "epochs":20}
    ]

    for i in hyper_parameters:
        _ = train_and_test_model(dataset_paths, weights_path, pickle_path, **i)
    