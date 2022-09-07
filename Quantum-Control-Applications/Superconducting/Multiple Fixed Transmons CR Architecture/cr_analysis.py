import numpy as np
import matplotlib.pyplot as plt
Q = np.load("CR_echo_HT_Q.npz")
Qt = Q.f.arr_0
t_min = 4
t_max = 125
dt = 1
ts = np.arange(t_min, t_max, dt)
plt.subplot(311)
plt.cla()
plt.title("<z> - Q")
plt.plot(ts * 4, Qt[0])
plt.plot(ts * 4, Qt[1])
plt.subplot(312)
plt.cla()
plt.title("<y> - Q")
plt.plot(ts * 4, Qt[2])
plt.plot(ts * 4, Qt[3])
plt.subplot(313)
plt.cla()
plt.title("<x> - Q")
plt.plot(ts * 4, Qt[4])
plt.plot(ts * 4, Qt[5])