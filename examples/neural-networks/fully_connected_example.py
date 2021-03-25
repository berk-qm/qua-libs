"""
fully_connected_example.py: Fully connected NN with backpropagation in QUA
Author: Ilan Mitnikov - Quantum Machines
Created: 14/03/2021
Created on QUA version: 0.8
"""
from configuration import config
from Network import *
from qm.QuantumMachinesManager import (
    QuantumMachinesManager,
    SimulationConfig,
    LoopbackInterface,
)
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt


def measure_value(var, a):
    measure("readout" * amp(a), "qe1", None, demod.full("iw", var, "out1"))


weights1 = 0.1 * np.array([[1, 2, 3], [4, 5, 6]])
weights2 = 0.1 * np.array([[1, 2], [4, 5], [7, 8]])
weights3 = 0.1 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
label = [0.1, -0.2, 0.3]
params = (0.1 * np.arange(weights1.shape[1])).tolist()

with program() as prog:
    layer1 = Dense(3, 2, activation=ReLu())
    layer2 = Dense(2, 3, activation=ReLu(), weights=weights2)
    layer3 = Dense(3, 3, initializer=Normal())
    nn = Network(
        layer1, layer2, layer3, loss=MeanSquared(), learning_rate=0.05, name="mynet"
    )

    var = declare(fixed)
    label_ = declare(fixed, value=label)
    input_ = declare(fixed, size=layer1.input_size)
    # output = declare(fixed, size=layer1.output_size)
    # result = declare_stream()

    conv = Conv(
        (3, 4),
        (2, 2),
        kernel_weights=np.array([[1, 1], [2, 2]]),
    )
    input2_ = declare(
        fixed, value=(np.array([1, 1, 2, 3, 2, 2, 3, 4, 3, 3, 4, 5]) * 0.1).tolist()
    )
    conv.forward(input2_, save_to="conv_res")
    conv.save_weights_("conv_")
    i = declare(int)
    with for_(i, 0, i < 50, i + 1):
        i = declare(int)
        a = declare(fixed)
        with for_each_((i, a), (list(range(weights1.shape[1])), params)):
            measure_value(var, a)
            assign(input_[i], var)

        # layer1.forward(input_, output)
        # layer2.forward(output, stream_or_tag="result")

        # nn.forward(input_)
        # nn.backprop(label)

        nn.training_step(input_, label_)

    nn.save_weights()
    nn.save_results()

job = QuantumMachinesManager().simulate(
    config,
    prog,
    SimulationConfig(
        100000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])
    ),
)
job.result_handles.wait_for_all_values()
# print(job.result_handles.result.fetch_all()['value'])
results = job.result_handles.mynet_results_stream.fetch_all()["value"]
# print(job.result_handles.mynet_results_stream.fetch_all()["value"])
plt.plot(results)
plt.hlines(label, xmin=0, xmax=results.shape[0], color="r", linestyles="--")
plt.ylabel("Label")
plt.figure()
plt.plot(job.result_handles.mynet_loss_stream.fetch_all()["value"])
plt.ylabel("Loss")
