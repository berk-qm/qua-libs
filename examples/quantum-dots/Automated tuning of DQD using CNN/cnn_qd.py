from funcs import *
from configuration import config

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig, LoopbackInterface
import numpy as np

qmm = QuantumMachinesManager()

v1_start = -0.2
v1_end = 0
v2_start = -0.15
v2_end = 0.10
step = 0.01
n_avg = 100

sim_args = SimulationConfig(int(1e6), simulation_interface=LoopbackInterface([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 1)]))

qm = QuantumMachinesManager().open_qm(config)
a = charge_stability_patch(v1_start, v1_end, v2_start, v2_end, step, n_avg, qm,
                           simulate=sim_args, dry_run=True
                           )