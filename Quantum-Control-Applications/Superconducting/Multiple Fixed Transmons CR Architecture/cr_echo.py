"""
amp_rabi_vs_freq.py: performs 2D scans to all qubits of the driving amplitude vs the driving frequency.
The goal of this protocol is to find the resonance frequencies of the qubits.
"""
# todo: axis and axis labels
import matplotlib
matplotlib.use('TKAgg')
from state_and_config import build_config, state
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qualang_tools.units import unit

###################
# The QUA program #
###################

control = 1
target = 0

n_avg = 10000
u = unit()
cooldown_time = 50 * u.us // 4

t_min = 4
t_max = 125
dt = 1
ts = np.arange(t_min, t_max, dt)

with program() as cr_echo:

    n = declare(int)
    t = declare(int)
    It = declare(fixed)
    Qt = declare(fixed)
    It_st = declare_stream()
    Qt_st = declare_stream()

    frame_rotation_2pi(0.125, f"cr_c{control}t{target}")

    with for_(n, 0, n < n_avg, n + 1):
        with for_(t, t_min, t < t_max, t + dt):

            for a in [0.0, 1.0]:

                play("x180"*amp(a), f"q{control}")
                align()
                play("cw"*amp(1.0), f"cr_c{control}t{target}", duration=t)
                align()
                play("x180", f"q{control}")
                align()
                wait(5, f"cr_c{control}t{target}")
                play("cw"*amp(-1.0), f"cr_c{control}t{target}", duration=t)
                align()
                wait(4, f"rr{target}")
                measure("readout"*amp(0.7), f"rr{target}", None,
                    dual_demod.full("cos", "out1", "sin", "out2", It),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Qt))

                wait(cooldown_time, f"rr{target}")
                save(It, It_st)
                save(Qt, Qt_st)

                align()

            for a in [0.0, 1.0]:

                play("x180" * amp(a), f"q{control}")
                align()
                play("cw" * amp(1.0), f"cr_c{control}t{target}", duration=t)
                align()
                play("x180", f"q{control}")
                align()
                wait(5, f"cr_c{control}t{target}")
                play("cw" * amp(-1.0), f"cr_c{control}t{target}", duration=t)
                align()
                play("x90", f"q{target}")
                wait(4, f"rr{target}")
                measure("readout" * amp(0.7), f"rr{target}", None,
                        dual_demod.full("cos", "out1", "sin", "out2", It),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Qt))

                wait(cooldown_time, f"rr{target}")
                save(It, It_st)
                save(Qt, Qt_st)

                align()

            for a in [0.0, 1.0]:
                play("x180" * amp(a), f"q{control}")
                align()
                play("cw" * amp(1.0), f"cr_c{control}t{target}", duration=t)
                align()
                play("x180", f"q{control}")
                align()
                wait(5, f"cr_c{control}t{target}")
                play("cw" * amp(-1.0), f"cr_c{control}t{target}", duration=t)
                align()
                play("y90", f"q{target}")
                wait(4, f"rr{target}")
                measure("readout" * amp(0.7), f"rr{target}", None,
                        dual_demod.full("cos", "out1", "sin", "out2", It),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Qt))

                wait(cooldown_time, f"rr{target}")
                save(It, It_st)
                save(Qt, Qt_st)

                align()

    with stream_processing():
        It_st.buffer(len(ts), 6).average().save(f"It")
        Qt_st.buffer(len(ts), 6).average().save(f"Qt")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(port=9510)

#######################
# Simulate or execute #
#######################

simulate = False
config = build_config(state)

if simulate:
    simulation_config = SimulationConfig(duration=30000)
    job = qmm.simulate(config, cr_echo, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(cr_echo)

# Get results from QUA program
res_handles = job.result_handles
# res_handles.wait_for_all_values()

job.result_handles.get(f"It").wait_for_values(1)
job.result_handles.get(f"Qt").wait_for_values(1)

while job.result_handles.is_processing():

    plt.clf()
    # It = res_handles.get(f"It").fetch_all()
    Qt = res_handles.get(f"Qt").fetch_all().T

    # Plot results
    plt.subplot(311)
    plt.cla()
    plt.title("<z> - Q")
    plt.plot(ts*4, Qt[0])
    plt.plot(ts*4, Qt[1])
    plt.subplot(312)
    plt.cla()
    plt.title("<y> - Q")
    plt.plot(ts*4, Qt[2])
    plt.plot(ts*4, Qt[3])
    plt.subplot(313)
    plt.cla()
    plt.title("<x> - Q")
    plt.plot(ts*4, Qt[4])
    plt.plot(ts*4, Qt[5])

    plt.tight_layout()
    plt.pause(1)
    np.savez("CR_echo_HT_Q", Qt)
