"""
Leo's code for extracting the circuit signals from the raw gate data.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
#import uproot as rt

#import scipy.signal as sg  # add filter??

#ltspice_path = Path("/home/lz1919/Documents/UNI/year_two/multiwave/code/leo/ltspice_filter")
#ltspice_path = Path("/home/lei/leo/ltspice_filter")

#os.chdir(ltspice_path)
#print(os.getcwd())
#import apply_ltspice_filter

import metascint.ray_tracing.python.apply_ltspice_filter as apply_ltspice_filter

import metascint.ray_tracing.python.data_analysis as dp
import metascint.ray_tracing.python.vis as dpv
#import metascint.ray_tracing.python.timing_model as tm

folder_to_code = Path("/home/lz1919/Documents/UNI/year_two/multiwave/code/")
block = dpv.MetaScintillatorBlock([0.1, 0.2], 10, 3, 25)

circuit_path = folder_to_code / "leo/ltspice_filter/readout/Draft1.asc"
#image_out_path = Path("/home/lz1919/Documents/UNI/year_two/multiwave/code/leo/data/image_outputs")

#circuit_path = Path("/home/lei/leo/ltspice_filter/readout/Draft1.asc")

ms = 1e-3
ns = 1e-9
ps = 1e-12

"""
def find_naive_pulse(hits_data:dp.HitsData, bin_width=50e-12, sample_num_of_bin=10):  #take sample rate/width to be approx 50ps
    runID = hits_data.run_id
    data = hits_data.photon_hits
    data = data[data["posZ"] == -12.15].sort_values("time")
    data_times = data["time"] - (runID + 1)  # adjusts time such that the signal is positioned at zero sec

    bin_num = int((data_times.max() - data_times.min()) // bin_width)
    sample_num = sample_num_of_bin * bin_num

    time =  np.linspace(data_times.min(), data_times.max(), sample_num) 
    hist_data = plt.hist(data_times.to_numpy(), bins=bin_num, histtype="step")
    counts = hist_data[0]
    width = sample_num_of_bin
    signal = np.concatenate([np.ones([width]) * i for i in counts])

    time_0 = np.linspace(0, data_times.min(), 10)  # accounts for the gap between zero and data_times.min()
    signal_0 = np.concatenate([np.array([0]), np.zeros([9])*counts[0]])

    time = np.concatenate([time_0, time])
    signal = np.concatenate([signal_0, signal])

    return runID, time, signal  
"""

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

def run_signal(axes, event_num, time, signal, parameters, circuit_path=circuit_path):

    con_parameters = {key: value * 1e-12 for key, value in parameters.items()}
    parameters = {key: str(value) + "pF" for key, value in parameters.items()}

    dummy, signal_b1, signal_b2 = apply_ltspice_filter(circuit_path, time, signal, params=con_parameters)

    axes[1].plot(dummy, signal_b1, label="e" + str(parameters))
    axes[2].plot(dummy, signal_b2, label="t" + str(parameters))

    return dummy, signal_b1, signal_b2

def test_circuit_parameters(event_num, time, signal, parameters_list, circuit_path=circuit_path, image_out_path=None, scatter=False):
    """
    event_data is list of .csv
    parameters_list should look like list of differnet configs e.g. [[10,100,10], [100,100,100]]
    """

    data = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    if scatter:
        axes[0].plot(time, signal, "o", markersize=2)
    else:
        axes[0].plot(time, signal)
    
    axes[0].set_title("input signal")
    axes[1].set_title("output energy signal")
    axes[2].set_title("output time signal")
    
    for i in parameters_list:
        parameters = {
            "C1": i[0],
            "C2": i[1],
            "C3": i[2],
            }

        dummy, signal_b1, signal_b2 = run_signal(axes, event_num, time, signal, parameters, circuit_path=circuit_path)
        data.append((dummy, signal_b1, signal_b2))

    fig.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')  # missing from saved image
    #FIXME: fig.savefig
    
    return data