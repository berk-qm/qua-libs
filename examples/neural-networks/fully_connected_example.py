from configuration import config
from Network import *
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig, LoopbackInterface
from qm.qua import *
import numpy as np


def measure_value(var, a):
    measure("readout" * amp(a), "qe1", None, demod.full("iw", var, "out1"))


weights1 = 0.1 * np.array([[1, 2, 3], [4, 5, 6]])
weights2 = 0.1 * np.array([[1, 2], [4, 5], [7, 8]])
weights3 = 0.1 * np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

params = (0.1 * np.arange(weights1.shape[1])).tolist()

with program() as prog:
    layer = Dense(weights=weights1, bias=np.array([1, 2]), activation=ReLu())
    layer2 = Dense(weights=weights2)
    layer3 = Dense(weights=weights3)
    nn = Network(layer, layer2)

    var = declare(fixed)
    input_ = declare(fixed, size=weights1.shape[1])
    output = declare(fixed, size=weights1.shape[0])
    result = declare_stream()

    i = declare(int)
    a = declare(fixed)
    with for_each_((i, a), (list(range(weights1.shape[1])), params)):
        measure_value(var, a)
        assign(input_[i], var)

    layer.forward(input_, output)
    layer2.forward(output, stream_or_tag="result")

    nn.forward(input_, stream_or_tag="nn_result")

job = QuantumMachinesManager().simulate(config, prog, SimulationConfig(30000, simulation_interface=LoopbackInterface(
    [('con1', 1, 'con1', 1)])))
job.result_handles.wait_for_all_values()
print(job.result_handles.result.fetch_all()['value'])
print(job.result_handles.nn_result.fetch_all()['value'])
