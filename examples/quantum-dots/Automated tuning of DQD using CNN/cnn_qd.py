from funcs import *
from configuration import config

from qm.qua import *
from qm.QuantumMachinesManager import (
    QuantumMachinesManager,
    SimulationConfig,
    LoopbackInterface,
)
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

qmm = QuantumMachinesManager()

v1_start = -0.2
v1_end = 0
v2_start = -0.15
v2_end = 0.10
step = 0.01
n_avg = 100

# sim_args = SimulationConfig(int(1e5),
#                             simulation_interface=LoopbackInterface([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 1)],
#                                                                    noisePower=0.05 ** 2))
#
# qm = QuantumMachinesManager().open_qm(config)
# a = charge_stability_patch(v1_start, v1_end, v2_start, v2_end, step, n_avg, qm,
#                            simulate=sim_args, dry_run=True
#                            )
#

mod1 = lines_model(10)
data = np.load("labeled_patches.npz", allow_pickle=True)
images = data["arr_0"]
lines_labels = data["arr_1"]
transitions_labels = data["arr_2"]
data.close()
mod1.fit(images, lines_labels, batch_size=10)

images_transitions = []
labels_transitions = []
for i, label in enumerate(transitions_labels):
    if label is not None:
        labels_transitions.append(label)
        images_transitions.append(images[i])

mod2 = transitions_model(10)
labels_transitions = np.array(labels_transitions)
mod2.fit(
    np.array(images_transitions),
    {
        "lower_right": labels_transitions[:, 0, :],
        "upper_left": labels_transitions[:, 1, :],
        "upper_right": labels_transitions[:, 2, :],
    },
    batch_size=20,
    epochs=5,
)
