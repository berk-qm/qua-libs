from funcs import *
from configuration import config

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig, LoopbackInterface
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

sim_args = SimulationConfig(int(1e5),
                            simulation_interface=LoopbackInterface([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 1)],
                                                                   noisePower=0.05 ** 2))

qm = QuantumMachinesManager().open_qm(config)
a = charge_stability_patch(v1_start, v1_end, v2_start, v2_end, step, n_avg, qm,
                           simulate=sim_args, dry_run=True
                           )

cce_loss = tf.keras.losses.categorical_crossentropy

# define a model for recognizing whether the dots are empty
# the inputs are patches of voltages of size 20*20 with resolution 6-9 mV
# the output is a boolean(0/1)
lines = models.Sequential()
lines.add(layers.Dropout(0.05, input_shape=(20, 20, 1)))
lines.add(layers.Conv2D(48, (4, 4), activation='relu'))
lines.add(layers.Dropout(0.05))
lines.add(layers.Conv2D(12, (3, 3), activation='relu'))
lines.add(layers.Dropout(0.4))
lines.add(layers.Flatten())
lines.add(layers.Dense(50, activation='sigmoid'))
lines.add(layers.Dense(2, activation='softmax'))
lines.compile(optimizer="adam", loss=cce_loss)


# defines a model to recognize individual transition
# the inputs are patches of voltages of size 28*28 with resolution 1mV
# the outputs are 3 layers with 4 value. Each layer corresponds to one of three corners(except the lowe left),
# each of the 4 values correspond to the kind of charge transition(change in charge) occurs when moving to that corner
input_ = tf.keras.Input(shape=(28, 28, 1))
dp1 = layers.Dropout(0.2)(input_)
conv1 = layers.Conv2D(72, (6, 6), activation='relu')(dp1)
dp2 = layers.Dropout(0.1)(conv1)
conv2 = layers.Conv2D(24, (3, 3), activation='relu')(dp2)
dp3 = layers.Dropout(0.3)(conv2)
conv3 = layers.Conv2D(12, (2, 3), activation='relu')(dp3)
fl = layers.Flatten()(conv3)
d = layers.Dense(50, activation='sigmoid')(fl)
d1 = layers.Dense(4, activation='softmax')(d)
d2 = layers.Dense(4, activation='softmax')(d)
d3 = layers.Dense(4, activation='softmax')(d)
transitions = models.Model(inputs=[input_], outputs=[d1, d2, d3])
transitions.compile(optimizer="adam", loss=cce_loss)
