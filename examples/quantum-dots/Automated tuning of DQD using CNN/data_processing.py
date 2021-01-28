import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import glob
import os

import QFlow_class

qf = QFlow_class.QFlow()

files = glob.glob(os.getcwd() + "/raw_data/" + "20171118-123151236063.npy")
dat = np.load(files[0], allow_pickle=True).item()

V_P1 = -dat['V_P1_vec']
V_P2 = -dat['V_P2_vec']
X, Y = np.meshgrid(V_P1, V_P2)
N_v = 100

current_vec = np.array([x['current'] for x in dat['output']]).reshape(N_v, N_v)
charge_vec = np.array([np.sum(x['charge']) for x in dat['output']]).reshape(N_v, N_v)
charge_vec_2 = np.array([np.array(x['charge']) for x in dat['output']]).reshape(N_v, N_v)

state_vec = np.array([x['state'] for x in dat['output']]).reshape(N_v, N_v)
sensor_vec = np.array([x['sensor'] for x in dat['output']]).reshape(N_v, N_v, -1)[:, :, 0]

matplotlib.rcParams.update({'font.size': 12})

fig, axarr = plt.subplots(2, 2, figsize=(11, 10))
fig.tight_layout(w_pad=7.0, h_pad=6.0)
plt.yticks(np.arange(0.0, 0.5, 0.1))

cd = axarr[0, 0].pcolor(X, Y, current_vec, vmax=1e-4, cmap=cm.summer)
axarr[0, 0].set_title('Current data')
axarr[0, 0].set_yticks(np.arange(0.0, 0.5, 0.1))
fig.colorbar(cd, ax=axarr[0, 0], fraction=0.045)

tcd = axarr[0, 1].pcolor(X, Y, charge_vec, cmap=plt.cm.get_cmap('summer', 13))
axarr[0, 1].set_title('Total charge number')
axarr[0, 1].set_yticks(np.arange(0.0, 0.5, 0.1))
fig.colorbar(tcd, ax=axarr[0, 1], fraction=0.045, ticks=np.arange(0, 13, 1))
tcd.set_clim(-0.5, 12.5)

csd = axarr[1, 0].pcolor(X, Y, sensor_vec, cmap=cm.summer)
axarr[1, 0].set_title('Charge sensor data')
axarr[1, 0].set_yticks(np.arange(0.0, 0.5, 0.1))
fig.colorbar(csd, ax=axarr[1, 0], fraction=0.045)

sd = axarr[1, 1].pcolor(X, Y, state_vec, cmap=plt.cm.get_cmap('summer', 4))
axarr[1, 1].set_title('State labels')
axarr[1, 1].set_yticks(np.arange(0.0, 0.5, 0.1))
sd.set_clim(-1.5, 2.5)
sd_bar = fig.colorbar(sd, fraction=0.045, ticks=[-1, 0, 1, 2])
sd_bar.ax.set_yticklabels(['SC(-1)', 'QPC(0)', 'SD(1)', 'DD(2)'])

plt.show()

map_diff = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


def get_random_patch(charge, patch_size):
    x = np.random.randint(0, charge.shape[0] - patch_size)
    y = np.random.randint(0, charge.shape[0] - patch_size)
    patch = (x, y)
    if charge[x, y] == charge[x + patch_size, y] == charge[x, y + patch_size] == charge[x + patch_size, y + patch_size]:
        lines = 0
        lr, ul, ur = 0, 0, 0
    else:
        lines = 1
        try:
            lr = map_diff[tuple(charge[x + patch_size, y] - charge[x, y])]
            ul = map_diff[tuple(charge[x, y + patch_size] - charge[x, y])]
            ur = map_diff[tuple(charge[x + patch_size, y + patch_size] - charge[x, y])]
        except KeyError:
            # in case there's a difference of more than 1 charge per quantum dot
            return None
    labels = (lines, (lr, ul, ur))
    return patch, labels
