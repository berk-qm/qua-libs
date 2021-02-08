from TuneDQD import TuneDQD
from configuration import config

from qm.QuantumMachinesManager import (
    QuantumMachinesManager,
    SimulationConfig,
    LoopbackInterface,
)
import numpy as np

qmm = QuantumMachinesManager()

v1_start = -0.2
v1_end = 0
v2_start = -0.1
v2_end = 0.10
step = 0.01
n_avg = 10

initial_voltages = np.array([0.04, 0.5])
patch_size = 10
sim_args = SimulationConfig(
    int(1e5),
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 1)], noisePower=0.05 ** 2
    ),
)

qm = QuantumMachinesManager().open_qm(config)
tuner = TuneDQD(
    patch_size, initial_voltages, n_avg, qm, simulate=sim_args, dry_run=True
)

gate_voltages = [(1, 2, 3)]
plunger_voltages = (v1_start, v1_end, v2_start, v2_end, step)
patches_per_diagram = 25
filename = "labeled_data"

tuner.generate_training_data(
    filename, gate_voltages, plunger_voltages, patches_per_diagram
)

tuner.train(filename + ".npz")

tuner.tune_state((1, 2))
