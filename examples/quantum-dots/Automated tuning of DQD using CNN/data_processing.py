import numpy as np

import glob
import os
import tensorflow as tf


def get_charge_state(filename):
    files = glob.glob(filename)
    dat = np.load(files[0], allow_pickle=True).item()

    N_v = 100

    charge_vec_2 = np.array([np.array(x["charge"]) for x in dat["output"]]).reshape(
        N_v, N_v
    )
    state_vec = np.array([x["state"] for x in dat["output"]]).reshape(N_v, N_v)
    charge_vec = np.array(
        [np.sum(np.array(x["charge"])) for x in dat["output"]]
    ).reshape(N_v, N_v)

    return charge_vec_2, state_vec, charge_vec


def charge_stability(charge):
    return np.mean(np.array(np.gradient(charge)), axis=0)


map_diff = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


def get_random_patch(state, charge2, patch_size):
    i = 0
    while i < 1000:
        x = np.random.randint(0, charge2.shape[0] - patch_size)
        y = np.random.randint(0, charge2.shape[0] - patch_size)
        patch = (x, y)
        if is_DD(state, x, y, patch_size):
            if (
                np.array_equal(charge2[x, y], charge2[x + patch_size, y])
                and np.array_equal(
                    charge2[x, y + patch_size], charge2[x + patch_size, y + patch_size]
                )
                and np.array_equal(
                    charge2[x + patch_size, y], charge2[x, y + patch_size]
                )
            ):
                lines = (1, 0)
                lr, ul, ur = 0, 0, 0
                labels = (lines, np.array(tf.one_hot((lr, ul, ur), 4)))
            else:
                lines = (0, 1)
                try:
                    lr = map_diff[tuple(charge2[x + patch_size, y] - charge2[x, y])]
                    ul = map_diff[tuple(charge2[x, y + patch_size] - charge2[x, y])]
                    ur = map_diff[
                        tuple(charge2[x + patch_size, y + patch_size] - charge2[x, y])
                    ]
                    labels = (lines, np.array(tf.one_hot((lr, ul, ur), 4)))
                except KeyError:
                    labels = (lines, None)
            return patch, labels
        else:
            i += 1
    print("Try a smaller patch size")
    return None


def is_DD(state, x, y, patch_size):
    return (
        (state[x, y] == 2)
        and (state[x + patch_size, y] == 2)
        and state[x, y + patch_size] == 2
        and state[x + patch_size, y + patch_size] == 2
    )


def get_patch(charge, x, y, patch_size):
    return charge[x : x + patch_size, y : y + patch_size]


def get_data_and_labels(size, count):
    images = []
    labels_lines = []
    labels_transitions = []
    folder_path = os.getcwd() + "/raw_data/"
    directory = os.fsencode(folder_path)
    for i, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            print(str(i) + " -- " + os.path.join(folder_path, filename))
            charge2, state, charge = get_charge_state(file_path)
            for _ in range(count):
                patch, label = get_random_patch(state, charge2, size)
                patch_im = get_patch(charge_stability(charge), *patch, size)
                images.append(patch_im)
                labels_lines.append(label[0])
                labels_transitions.append(label[1])
    return np.array(images), np.array(labels_lines), np.array(labels_transitions)
