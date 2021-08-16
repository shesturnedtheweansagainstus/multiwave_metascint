"""
Extracts input-target data from GATE .csv files
"""

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

import metascint.ray_tracing.python.data_analysis as dp
import metascint.ray_tracing.python.vis as dpv
import metascint.ray_tracing.python.timing_model as tm
import metascint.ray_tracing.python.circuit_signal as cs

# dataset
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
        "total_energy": _bytes_feature(serialize_array(total_energy)),
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
    Trying to speed up dataset writing
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

def extract_target_v1(hits:dp.HitsData):
    """
    OLD
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

def extract_target(hits:dp.HitsData):
    """
    
    We use Lei's energy sharing function. and combine Compton and Ray

    0->Others, 1-> Plastic, 2->Crystal,

    0->PhotoElectric, 1->Compton, 2->RayleighScattering

    should package them all up then choose which ones in the tf.dataset
    """

    """
    if hits.photon_hits.shape[0] == 0:
        return None
    if hits.hits[hits.hits["parentID"] == 0].shape == 0:
        return None
    """

    first_photon_time = np.array([
        hits.photon_hits.sort_values("time").iloc[0]["time"] - (hits.run_id + 1)  # starts each event at 0s
        ])

    energy = dp.ParticleAnalysis(hits).energy_share()
    total_energy = energy.sum()
    energy_share = np.zeros([3])
    for i in range(3):
        try:
            energy_share[i] = energy[i]
        except:
            continue
    #energy_share = energy_share / total_energy  # try for now with out % scale
    total_energy = np.array([total_energy])

    primary_gamma_pos = np.asarray(hits.hits[
        hits.hits["parentID"] == 0
    ].sort_values("time").iloc[0][["posX", "posY", "posZ"]], dtype=np.float64)  # could normalize

    processNames = ['PhotoElectric', 'Compton', 'RayleighScattering']
    process = hits.hits[hits.hits["parentID"] == 0].sort_values("time").iloc[0]["processName"]
    process = np.asarray([
        1 if process == i else 0 for i in processNames
    ])
    process = np.array([process[0], process[1] + process[2]])  # combines the Compton and Ray

    return first_photon_time, total_energy, energy_share, primary_gamma_pos, process

def extract_train_data(data_path, dataset_path):
    """
    Take a .csv of gate simulations (using standard mixed block) and 
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
        try:
            targets = extract_target(hits)
        except:
            continue
        
        first_photon_time, total_energy, energy_share, primary_gamma_pos, process = targets

        out = parse_single_input(
            signal, first_photon_time, total_energy, energy_share, primary_gamma_pos, process
            )

        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

if __name__ == '__main__':
    # {'PhotoElectric': 2072, 'Compton': 5004, 'RayleighScattering': 342} and proportion of compton is 0.67457535723
    data_path_0 = Path("/home/lei/leo/metascint_gvanode/output/metascint_type_2_2021-06-17_11:21:17.csv")  
    dataset_path_0 = Path("/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-06-17_11:21:17.tfrecords")

    data_path_1 = Path("/home/lei/leo/metascint_gvanode/output/metascint_type_2_2021-08-11_17:23:00.csv")
    dataset_path_1 = Path("/home/lei/leo/code/data/train_data/train_metascint_type_2_2021-08-11_17:23:00.tfrecords")

    data_path_2 = Path("/home/lei/leo/metascint_gvanode/output/metascint_type_2_2021-08-13_21:54:24.csv")
    dataset_path_2 = Path("/home/lei/leo/code/data/train_data/train_type_2_2021-08-13_21:54:24.tfrecords")

    #_ = extract_train_data(data_path_0, str(dataset_path_0))
    #_ = extract_train_data(data_path_1, str(dataset_path_1))
    #_ = extract_train_data(data_path_2, str(dataset_path_2))
