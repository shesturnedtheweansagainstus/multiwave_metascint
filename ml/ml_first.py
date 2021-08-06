# Set up new conda leo_env on the server
import matplotlib.pyplot as plt
#from metascint.ray_tracing.python.data_analysis import list_of_events
import numpy as np
import os
from numpy.lib.shape_base import _kron_dispatcher
import pandas as pd
from pathlib import Path
from tensorflow._api.v2 import data
from tensorflow.python.framework.op_def_registry import get
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.python.ops.gen_math_ops import log, sign
#import uproot as rt
import tensorflow as tf
import tensorboard

import pickle
import datetime
from contextlib import redirect_stdout

import metascint.ray_tracing.python.data_analysis as dp
import metascint.ray_tracing.python.vis as dpv
import metascint.ray_tracing.python.timing_model as tm
import metascint.ray_tracing.python.circuit_signal as cs

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

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_input(
    signal, first_photon_time, total_energy, energy_share, primary_gamma_pos, process
    ):

    features = {
        "signal": _bytes_feature(serialize_array(signal)),
        "first_photon_time": _bytes_feature(serialize_array(first_photon_time)),
        "total_energy": _int64_feature(total_energy),
        "energy_share": _bytes_feature(serialize_array(energy_share)),
        "primary_gamma_pos": _bytes_feature(serialize_array(primary_gamma_pos)),
        "process": _bytes_feature(serialize_array(process))
    }

    out = tf.train.Example(features=tf.train.Features(feature=features))
    return out

def sort_data_by_runID(data_path, sorted_data_path):
    """
    Sort csv by runID, have to make sure sorted_data_path is
    empty before running.
    """
    count = 0
    header = True
    for i in pd.read_csv(data_path, iterator=True, chunksize=5000):
        if count == 1:
            header = False
        sorted_events = i.sort_values("runID")
        sorted_events.to_csv(
            sorted_data_path, mode='a', index=False, header=header
            )
        count = 1
    return count

def find_list_of_events(path):
    events = {}
    for i in pd.read_csv(path, iterator=True, chunksize=5000):
        new_events = list(i["runID"].unique())
        for j in new_events:
            row_num = i[i["runID"] == j].shape[0]
            if j not in events.keys():
                events[j] = row_num
            else:
                events[j] = events[j] + row_num
    return events

def find_location_of_events(sorted_data_path, list_of_events):
    """
    Might work?? Depends on .csv being sorted by runID.
    """
    start_line = {}
    end_line = {}
    for i in pd.read_csv(sorted_data_path, iterator=True, chunksize=5000):
        new_events = list(i["runID"].unique())
        for j in new_events:
            if j not in start_line.keys():
                start_line[j] = i[i["runID"] == j].sort_index().iloc[0].name
            end_line[j] = i[i["runID"] == j].sort_index().iloc[-1].name
    
def find_naive_pulse(hits_data:dp.HitsData, bin_width=100e-12, step_num=2500):  #take sample rate/width to be approx 100ps
    """

    From the hitsdata of a given event, we generate the histogram signal from it

    """
    runID = hits_data.run_id
    data = hits_data.photon_hits
    data = data[data["posZ"] == -12.15].sort_values("time")
    data_times = data["time"] - (runID + 1)  # adjusts time such that the event starts at 0s

    back_time = np.asarray([data_times.min() + i*bin_width for i in range(step_num)])
    front_time = np.asarray([data_times.min() - i*bin_width for i in range(11, 0, -1)])  # simulates time before first photon
    time = np.concatenate([front_time, back_time])

    signal, time, _  = plt.hist(data_times.to_numpy(), bins=back_time, histtype="step")

    time = np.concatenate([front_time, back_time])[:-1]  # removes last right bin edge
    time = time - time[0]  # starts the signal at 0s
    signal = np.concatenate([np.zeros([11]), signal])
    
    return runID, time, signal  # shape (2510,)

def extract_event(runID, data_path, list_of_events, chunksize=5000):
    hits = pd.DataFrame()
    block = dpv.MetaScintillatorBlock([0.1, 0.2], 10, 3, 25)

    for i in pd.read_csv(data_path, iterator=True, chunksize=chunksize):
        hits = hits.append(i[i["runID"] == runID])
        if hits.shape[0] == list_of_events[runID]:
            return dp.HitsData(runID, 0, hits, metascint_config=block)

    if hits.shape[0] == 0:
        return None
    else:
        return dp.HitsData(runID, 0, hits, metascint_config=block)

def extract_time_energy(time, signal, parameters, circuit_path):
    """
    parameters in pF
    FIXME: butter filter??
    """
    con_parameters = {key: value * 1e-12 for key, value in parameters.items()}
    dummy, signal_b1, signal_b2 = cs.apply_ltspice_filter(
        circuit_path, time, signal, params=con_parameters
        )
    return signal_b1, signal_b2

def extract_target(hits:dp.HitsData):
    """
    We use the number of photons as a measure of energy.
    I assume all photons have exactly one interaction.

    0->Others, 1-> Plastic, 2->Crystal,

    0->PhotoElectric, 1->Compton, 2->RayleighScattering

    should package them all up then choose which ones in the tf.dataset
    """

    if hits.photon_hits.shape[0] == 0:
        return None

    first_photon_time = np.array([
        hits.photon_hits.sort_values("time").iloc[0]["time"] - (hits.run_id + 1)  # starts each event at 0s
        ])

    total_energy = hits.photon_hits["trackID"].unique().shape[0]  # ensures all photons are counted once
    energy_share = np.asarray([
        hits.photon_hits[hits.photon_hits["locationType"] == i].shape[0] 
        for i in range(3)
    ])
    energy_share = (energy_share / total_energy) * 100  # %
    #total_energy = np.array([total_energy])

    primary_gamma_pos = np.asarray(hits.hits[
        hits.hits["parentID"] == 0
    ].sort_values("time").iloc[0][["posX", "posY", "posZ"]], dtype=np.float64)  # could normalize

    processNames = ['PhotoElectric', 'Compton', 'RayleighScattering']
    process = hits.hits[hits.hits["parentID"] == 0].sort_values("time").iloc[0]["processName"]
    process = np.asarray([
        1 if process == i else 0 for i in processNames
    ])

    return first_photon_time, total_energy, energy_share, primary_gamma_pos, process

def extract_train_data(data_path, dataset_path):
    """
    Takes a .csv of gate simulations (using standard mixed block) and 
    outputs a TFrecord file of (input, target) tuples.

    For now we don't apply the circuit.
    """
    block = dpv.MetaScintillatorBlock([0.1, 0.2], 10, 3, 25)
    writer = tf.io.TFRecordWriter(dataset_path)
    count = 0

    list_of_events = find_list_of_events(data_path)
    data = pd.read_csv(data_path)  # only on server
    print(f"Number of events: {len(list_of_events)}")

    for i in list_of_events.keys():
        #hits = extract_event(i, data_path, list_of_events)
        hits = dp.HitsData(i, 0, data[data["runID"] == i], metascint_config=block)
        print([count, i])
        runID, time, signal = find_naive_pulse(hits)
        
        #first_photon_time, total_energy, energy_share, primary_gamma_pos, process = extract_target(hits)
        targets = extract_target(hits)
        if targets == None:
            continue
        else:
            first_photon_time, total_energy, energy_share, primary_gamma_pos, process = targets

        out = parse_single_input(
            signal, first_photon_time, total_energy, energy_share, primary_gamma_pos, process
            )

        writer.write(out.SerializeToString())
        count += 1
        if count == 7419:  # temp
            break

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def parse_TFR_element(element):
    """
    From dataset = tf.data.TFRecordDataset(filename)
    extracts the original elements.
    """

    features = {
        "signal": tf.io.FixedLenFeature([], tf.string),
        "first_photon_time": tf.io.FixedLenFeature([], tf.string),
        "total_energy": tf.io.FixedLenFeature([], tf.int64),
        "energy_share": tf.io.FixedLenFeature([], tf.string),
        "primary_gamma_pos": tf.io.FixedLenFeature([], tf.string),
        "process": tf.io.FixedLenFeature([], tf.string)
    }

    data = tf.io.parse_single_example(element, features)

    signal = tf.io.parse_tensor(data["signal"], out_type=tf.float64)
    signal = tf.reshape(signal, shape=[2510, 1])  # for 1D conv

    first_photon_time = tf.io.parse_tensor(data["first_photon_time"], out_type=tf.float64)
    first_photon_time = tf.reshape(first_photon_time, shape=[1])

    total_energy = data["total_energy"]
    total_energy = tf.reshape(total_energy, shape=[1])
    total_energy = tf.cast(total_energy, tf.float64)
 
    energy_share = tf.io.parse_tensor(data["energy_share"], out_type=tf.float64)
    energy_share = tf.reshape(energy_share, shape=[3])

    primary_gamma_pos = tf.io.parse_tensor(data["primary_gamma_pos"], out_type=tf.float64)
    primary_gamma_pos = tf.reshape(primary_gamma_pos, shape=[3])

    process = tf.io.parse_tensor(data["process"], out_type=tf.int64)
    process = tf.reshape(process, shape=[3])
    process = tf.cast(process, tf.float64)

    # Write function to map over and take the relevent variables
    return (signal, (first_photon_time, total_energy, energy_share, primary_gamma_pos, process))  # [11] 234 8910

def get_dataset(filename):
  dataset = tf.data.TFRecordDataset(filename)
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
        return ({"input": train_element},

                {"first_photon_time": target_element[0], 
                "total_energy": target_element[1],
                "energy_share": target_element[2],
                "primary_pos": target_element[3], 
                "process": target_element[4]})

    return dataset.map(group_target)

def split_dataset(dataset, size, train_split=0.9, shuffle=True, shuffle_size=10000, predict_size=10):  
    """
    Extracts the train/test datasets; val later
    shuffle_size > size
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

def conv_layer(input, filters, kernel_size, kernel_regularizer='l2'):
    """
    Alter to tune batch norm
    """
    out = tf.keras.layers.Conv1D(
        filters, kernel_size, kernel_regularizer=kernel_regularizer
        )(input)
    out = tf.keras.layers.BatchNormalization()(out)  #axis=-1
    out = tf.keras.layers.Activation('relu')(out)
    return out

def my_model():  # try multiple outputs?? and remove training??
    """
    For hyperparameters, return a tensorflow model
    FIXME: make filters etc variable
    """
    input = tf.keras.Input(shape=(2510, 1), name="input")  # (batch, 2510, 1) 

    stream_one = conv_layer(input, 8, 5)  # (b, 502, 8) (dim, length)
    stream_two = conv_layer(input, 8, 15)  # (b, 168, 8) 

    stream_one = tf.keras.layers.MaxPool1D(pool_size=3, strides=1)(stream_one)  # (b, 500, 8)
    stream_two = tf.keras.layers.MaxPool1D(pool_size=3, strides=1)(stream_two)  # (b, 166, 8)

    stream_one = tf.keras.layers.Dropout(0.5)(stream_one)
    stream_two = tf.keras.layers.Dropout(0.5)(stream_two)

    stream_one = conv_layer(input, 4, 5)  # (b, 100, 4) 
    stream_two = conv_layer(input, 6, 5)  # (b, 34, 6)

    stream_one = conv_layer(input, 4, 5)  # (b, 20, 4) 
    stream_two = conv_layer(input, 4, 5)  # (b, 7, 4)

    stream_one = tf.keras.layers.MaxPool1D(pool_size=5, strides=1)(stream_one)  # (b, 16, 4)
    stream_two = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(stream_two)  # (b, 6, 4)

    out = tf.concat([stream_two, stream_two], axis=-2)  # (b, 22, 4)

    #out = tf.keras.layers.GRU(40, kernel_regularizer="l2", recurrent_regularizer="l2", bias_regularizer="l1")(out)  # (b, 40)
    #out = tf.keras.layers.GRU(40, kernel_regularizer="l2", recurrent_regularizer="l2", bias_regularizer="l1")(out)  # (b, 40)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dropout(0.5)(out)

    first_photon_time = tf.keras.layers.Dense(20, activation="relu", kernel_regularizer='l2')(out)
    first_photon_time = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer='l2', name="first_photon_time")(first_photon_time)

    total_and_energy = tf.keras.layers.Dense(30, activation="relu", kernel_regularizer='l2')(out)
    total_and_energy = tf.keras.layers.Dense(15, activation="relu", kernel_regularizer='l2')(total_and_energy)
    total_and_energy = tf.keras.layers.Dense(4, activation="relu", kernel_regularizer='l2')(total_and_energy)

    total_energy = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer='l2', name="total_energy")(total_and_energy)
    energy_share = tf.keras.layers.Dense(3, activation="relu", kernel_regularizer='l2', name="energy_share")(total_and_energy)

    primary_and_process = tf.keras.layers.Dense(30, activation="relu", kernel_regularizer='l2')(out)
    primary_and_process = tf.keras.layers.Dense(15, activation="relu", kernel_regularizer='l2')(primary_and_process)

    primary_pos = tf.keras.layers.Dense(3, activation="relu", name="primary_pos")(primary_and_process)
    process = tf.keras.layers.Dense(3, activation="softmax", name="process")(primary_and_process)

    return tf.keras.Model(inputs=[input], outputs=[first_photon_time, total_energy, energy_share, primary_pos, process])

def get_run_name(**kwargs):
    """
    Generates a run name for given hyperparameters to log on 
    TensorBoard.
    """
    name = "".join([f"{key}{kwargs[key]}_" for key in kwargs.keys()])
    return name[:-1]

def train_model(train_set, model, steps_per_epoch, **kwargs):  # FIXME: hardcoded optimizer type
    adam = tf.keras.optimizers.Adam(**kwargs)
    model.compile(
        optimizer=adam, 
        loss = {"first_photon_time": tf.keras.losses.MeanSquaredError(), "total_energy": tf.keras.losses.MeanSquaredError(),
                "energy_share": tf.keras.losses.MeanSquaredError(), "primary_pos": tf.keras.losses.MeanSquaredError(), 
                "process": tf.keras.losses.CategoricalCrossentropy()}, 
        metrics = {"first_photon_time": tf.keras.metrics.MeanSquaredError(),"total_energy": tf.keras.metrics.MeanSquaredError(),
                    "energy_share": tf.keras.metrics.MeanSquaredError(), "primary_pos": tf.keras.metrics.MeanSquaredError(), 
                    "process": tf.keras.metrics.CategoricalAccuracy()}
        )
    run_name = get_run_name(**kwargs)
    logdir = log_path / run_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    history = model.fit(train_set, epochs = 10, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])
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

def train_and_test_model(dataset_path, weights_path, pickle_path, batchsize=32, **kwargs):

    dataset = get_dataset(dataset_path)
    size = find_size_of_dataset(dataset)  # 7181
    print(f"Dataset size is: {size}")
    dataset = get_dataset(dataset_path)
    train_set, test_set, predict_set = split_dataset(dataset, size=size)  

    train_set = group_target_shape(train_set)
    test_set = group_target_shape(test_set)

    train_set = train_set.repeat().batch(batchsize)  # change
    test_set = test_set.batch(1)

    model = my_model()
    
    steps_per_epoch = (size // batchsize) + 1
    history = train_model(train_set, model, steps_per_epoch, **kwargs)

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
    run_name = get_run_name(**kwargs)

    _ = run_name + ".txt"
    with open(text_path / _, "w") as f:
        with redirect_stdout(f):
            _ = show_predictions(model, predict_set)
            print(f"\n\n{model.summary()}\n")
            print(f"\n\nHyperparameters: {kwargs}\n")
            print(f"\n\nHistory: {history.history}\n")
            print(f"\n\nEval Data: {eval_data}\n")

    time = np.asarray([i*100e-12 for i in range(2510)])
    plt.plot(time, _[0][:, 0])
    _ = run_name + ".png"
    plt.savefig(image_path / _)

    model.save(weights_path)  # use .h5 format
    pickle.dump([history.history, eval_data], open(pickle_path, "wb"))
    return (history, eval_data)

def show_predictions(model, dataset, space_num=45, padding=20):
    """
    Formats and prints out a comparision of target-predictions
    """

    predict_list = []
    labels = ["first_photon_time", "total_energy", "energy_share", "primary_pos", "process"]

    for i in dataset:
        print("===================")
        predict = model(tf.expand_dims(i[0], axis=0), training=False)
        print(" " * (padding + 1) + "target" + "   " + "-" * (space_num - 2) + "   " + "prediction")

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

def prediction_model(dataset_path, weights_path, predict_num=10, shuffle_size=10000, space_num=45, padding=20):
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
FIXME: 
    Tensorboard
    Presentation
    Fix latter 
    Fix Conv + LSTM

    write model.summary(), predictions, eval, loss, accuarcy to txt

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
    # {'PhotoElectric': 2072, 'Compton': 5004, 'RayleighScattering': 342}
    data_path = Path("/home/lei/leo/metascint_gvanode/output/metascint_type_2_2021-06-17_11:21:17.csv")  
    sorted_data_path = Path("/home/lei/leo/code/data/sorted_metascint_type_2_2021-06-17_11:21:17.csv")
    dataset_path = Path("/home/lei/leo/code/data/test_7_metascint_type_2_2021-06-17_11:21:17.tfrecords")

    weights_path = Path("/home/lei/leo/code/data/saved_weights/mymodel.h5")  # FIXME: alter with 
    pickle_path = Path("/home/lei/leo/code/data/out_data")
    log_path = Path("/home/lei/leo/code/data/logs/fit")
    image_path = Path("/home/lei/leo/code/data/images")
    text_path = Path("/home/lei/leo/code/data/images/text")

    #_ = extract_train_data(data_path, str(dataset_path))

    #{"learning_rate":0.001, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-05, "amsgrad":True, "name":"adam"},
    
    hyper_parameters = [
        {"learning_rate":0.001, "beta_1":0.9, "beta_2":0.999, "epsilon":1e-05, "amsgrad":False, "name":"adam"}
    ]

    for i in hyper_parameters:
        _ = train_and_test_model(str(dataset_path), weights_path, pickle_path, **i)
    

    #_ = prediction_model(str(dataset_path), str(weights_path))
