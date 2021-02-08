import data_processing

from qm.qua import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def scan2d(v1_start, v1_end, v2_start, v2_end, step, n_avg):
    """
    A 2D voltage scan of plunger gates using the OPX
    """
    n_v1 = int((v1_end - v1_start) / step)
    n_v2 = int((v2_end - v2_start) / step)
    with program() as prog:
        v1 = declare(fixed)
        v2 = declare(fixed)
        I = declare(fixed)
        n = declare(int)
        I_avg = declare_stream()
        with for_(n, 0, n < n_avg, n + 1):
            with for_(v1, v1_start, v1 < v1_end, v1 + step):
                with for_(v2, v2_start, v2 < v2_end, v2 + step):
                    align("PG1", "PG2", "QPC")
                    play("playOp" * amp(v1), "PG1")
                    play("playOp" * amp(v2), "PG2")
                    measure(
                        "readout", "QPC", None, integration.full("integW", I, "out1")
                    )
                    save(I, I_avg)
        with stream_processing():
            I_avg.buffer(n_v1, n_v2).average().save("current")
    return prog


def charge_stability_patch(
    v1_start, v1_end, v2_start, v2_end, step, n_avg, qm, **execute_args
):
    """
    Returns a charge stability diagram using 2d scan
    """
    job = qm.execute(
        scan2d(v1_start, v1_end, v2_start, v2_end, step, n_avg), **execute_args
    )
    job.result_handles.wait_for_all_values()
    current = job.result_handles.current.fetch_all()
    # calculate the derivative of the current for the charge stability diagram
    charge_stability = np.mean(
        np.array(np.gradient(current, step)[0] + np.gradient(current, step)[1])
    )

    return charge_stability


def set_gate_voltages(*v):
    """
    Sets the voltage of different gates using external devices
    """
    pass


def generate_diagrams(gate_voltages, plunger_voltages, n_avg, qm, **execute_args):
    """
    Generates charge stability diagrams for different gate configurations and given plunger voltages
    :param gate_voltages: a list of gate voltages configurations to scan
    :param plunger_voltages: v1_start, v1_end, v2_start, v2_end, step - the start and end values of the plunger gates voltages
    :param n_avg: the number of measurements to average upon
    :param qm: a QuantumMachine object
    :param execute_args: Qua execution arguments
    """
    diagrams = []
    v1_start, v1_end, v2_start, v2_end, step = plunger_voltages
    for v in gate_voltages:
        set_gate_voltages(*v)
        diagrams.append(
            charge_stability_patch(
                v1_start, v1_end, v2_start, v2_end, step, n_avg, qm, **execute_args
            )
        )

    return diagrams


def labeled_patch(diagram, size):
    """
    A returns a random labeled patch of a charge-stability diagram
    """
    x, y = np.random.randint(0, diagram.shape[0] - size), np.random.randint(
        0, diagram.shape[0] - size
    )

    # for simulation purposes
    label_line = (1, 0)
    label_transition = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])

    return diagram[x : x + size, y : y + size], label_line, label_transition


def get_random_patches(diagrams, size, count):
    """
    Gets random patches from charge stability diagrams, and assigns labels to each
    :param diagrams: the charge-stability diagrams to sample
    :param size: the patch size to sample
    :param count: the number of patches per charge-stability diagram
    """

    # when using real data and having a labeling function or person

    # patches = []
    # line_labels = []
    # transition_labels = []
    # for d in diagrams:
    #     for i in range(count):
    #         patch, line_label, transition_label = labeled_patch(d, size)
    #         patches.append(patch)
    #         line_labels.append(line_label)
    #         transition_labels.append(transition_label)
    # return np.array(patches), np.array(line_labels), np.array(transition_labels)

    # when using simulated data from QFlow
    return data_processing.get_data_and_labels(size, count)


cce_loss = tf.keras.losses.categorical_crossentropy


def lines_model(input_size):
    """
    define a model for recognizing whether the dots are empty
    the inputs are patches of voltages
    the output is a boolean(0/1)
    """
    lines = models.Sequential()
    lines.add(layers.Dropout(0.05, input_shape=(input_size, input_size, 1)))
    lines.add(layers.Conv2D(48, (4, 4), activation="relu"))
    lines.add(layers.Dropout(0.05))
    lines.add(layers.Conv2D(12, (3, 3), activation="relu"))
    lines.add(layers.Dropout(0.4))
    lines.add(layers.Flatten())
    lines.add(layers.Dense(50, activation="sigmoid"))
    lines.add(layers.Dense(2, activation="softmax"))
    lines.compile(optimizer="adam", loss=cce_loss, metrics=["accuracy"])
    return lines


def transitions_model(input_size):
    """
    defines a model to recognize individual transition
    the inputs are patches of voltages
    the outputs are 3 layers with 4 value. Each layer corresponds to one of three corners(except the lowe left),
    each of the 4 values correspond to the kind of charge transition(change in charge) occurs when moving to that corner
    """
    input_ = tf.keras.Input(shape=(input_size, input_size, 1))
    dp1 = layers.Dropout(0.2)(input_)
    conv1 = layers.Conv2D(72, (6, 6), activation="relu")(dp1)
    dp2 = layers.Dropout(0.1)(conv1)
    conv2 = layers.Conv2D(24, (3, 3), activation="relu")(dp2)
    dp3 = layers.Dropout(0.3)(conv2)
    conv3 = layers.Conv2D(12, (2, 3), activation="relu")(dp3)
    fl = layers.Flatten()(conv3)
    d = layers.Dense(50, activation="sigmoid")(fl)
    d1 = layers.Dense(4, activation="softmax", name="lower_right")(d)
    d2 = layers.Dense(4, activation="softmax", name="upper_left")(d)
    d3 = layers.Dense(4, activation="softmax", name="upper_right")(d)
    transitions = models.Model(inputs=[input_], outputs=[d1, d2, d3])
    transitions.compile(optimizer="adam", loss=cce_loss, metrics=["accuracy"])

    return transitions
