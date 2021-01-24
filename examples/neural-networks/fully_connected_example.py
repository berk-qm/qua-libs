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


input_size = 5
output_size = 7
weights = np.random.rand(output_size, input_size)
params = (0.1 * np.arange(input_size)).tolist()
layer = DenseLayer(input_size, output_size, weights, activation=relu)
layer2 = DenseLayer(output_size, input_size, np.random.rand(input_size, output_size))
with program() as prog:
    var = declare(fixed)
    input_ = declare(fixed, size=input_size)
    output = declare(fixed, size=output_size)
    result = declare_stream()

    i = declare(int)
    a = declare(fixed)
    with for_each_((i, a), (list(range(input_size)), params)):
        measure_value(var, a)
        assign(input_[i], var)

    layer.feed_forward(input_, output)
    layer2.feed_forward(output, save_to="result")

job = QuantumMachinesManager().simulate(config, prog, SimulationConfig(30000, simulation_interface=LoopbackInterface(
    [('con1', 1, 'con1', 1)])))
job.result_handles.wait_for_all_values()
print(job.result_handles.result.fetch_all()['value'])
