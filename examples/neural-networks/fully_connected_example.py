from configuration import config
from NeuralNetwork import DenseLayer
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig, LoopbackInterface
from qm.qua import *
import numpy as np


def measure_value(var, a):
    measure("readout" * amp(a), "qe1", None, demod.full("iw", var, "out1"))


def relu(var):
    zero = declare(fixed, value=0)
    with if_(var < 0):
        assign(var, zero)


weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
weights2 = np.array([[1, 2], [4, 5], [7, 8]])

params = (0.1 * np.arange(weights1.shape[1])).tolist()
layer = DenseLayer(weights1, activation=relu)
layer2 = DenseLayer(weights2)
with program() as prog:
    var = declare(fixed)
    input_ = declare(fixed, size=weights1.shape[1])
    output = declare(fixed, size=weights1.shape[0])
    result = declare_stream()

    i = declare(int)
    a = declare(fixed)
    with for_each_((i, a), (list(range(weights1.shape[1])), params)):
        measure_value(var, a)
        assign(input_[i], var)

    layer.feed_forward(input_, output)
    layer2.feed_forward(output, save_to="result")

job = QuantumMachinesManager().simulate(config, prog, SimulationConfig(30000, simulation_interface=LoopbackInterface(
    [('con1', 1, 'con1', 1)])))
job.result_handles.wait_for_all_values()
print(job.result_handles.result.fetch_all()['value'])
