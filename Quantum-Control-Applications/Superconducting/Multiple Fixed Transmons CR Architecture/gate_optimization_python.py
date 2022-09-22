"""
gate_optimization_realtime.py: Optimizes the gate amplitude using an error amplification sequence and computing the
cost function on the FPGA.
"""

from state_and_config import build_config, state
from qm.simulate import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import time
from qualang_tools.units import unit
from qualang_tools.results import fetching_tool

u = unit()
###################
# The QUA program #
###################
gate = "x90"        # gate under study (ex: "x90" for pi/2 pulse along X)
goal = 0.5          # reference transition probability (ex: 0.5 for pi/2 pulse optimization)
exit_cond = 0.03    # value of the cost function that exits the loop
nb_of_pulses = 5    # number of pi/2 pulses defined as (X/2)^nb_of_pulses - must be odd
qubits = [0, 1]      # list of the qubits under study

n_avg_power_of_2 = 10   # number of averages to derive the transition probability defined as 2**n_avg_power_of_2
ge_threshold = 0.07958  # threshold used for state discrimination (excited state if I > ge_threshold)

cooldown_time = 10 * u.us // 4  # wait time between each iteration to let the qubit and resonator decay
# Amplitude scan, reduce da to increase the resolution
a_min = 0.1
a_max = 1.0
da = 0.1
amps = np.arange(a_min, a_max + da / 2, da)

with program() as gate_optimization:

    n = declare(int)
    n_p = declare(int)
    f = declare(int)
    a = declare(fixed)
    I = declare(fixed)
    res = declare(bool)
    probability = declare(fixed)
    cost = declare(fixed)
    probability_st =  [declare_stream() for _ in qubits]
    cost_st =  [declare_stream() for _ in qubits]
    for q in range(len(qubits)):
        # ge_threshold = state["readout_resonators"][q]["ge_threshold"]
        assign(a, a_min)
        assign(cost, 0.1)
        # Exit the loop if the cost function reached the exit threshold or if the amplitude scan is finished
        with while_((cost > exit_cond) & (a < a_max + da / 2)):
            assign(probability, 0)
            with for_(n, 0, n < 2 ** n_avg_power_of_2, n + 1):
                with for_(n_p, 0, n_p < nb_of_pulses, n_p + 1):
                    play(gate * amp(a), f"q{qubits[q]}")
                align()
                wait(
                    4, f"rr{qubits[q]}"
                )  # to prevent simultaneous driving and readout
                measure(
                    "readout",
                    f"rr{qubits[q]}",
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                )
                assign(res, I > ge_threshold)
                # Sum the results and divide by the number of iterations to get the average on the fly
                assign(
                    probability,
                    probability + (Cast.to_fixed(res) >> n_avg_power_of_2),
                )
                wait(cooldown_time, f"rr{qubits[q]}")
            # Assign the probability to IO1 to be accessible in python
            assign(IO1, probability)
            # Wait until python resume()
            pause()
            # Get the cost function evaluation from python
            assign(cost, IO2)
            assign(a, a + da)
        # Condition to indicate python that the loop is over (note that probability cannot be -2.0)
        assign(IO1, -2.0)
        pause()

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=state["network"]["ip"], port=state["network"]["port"])

#######################
# Simulate or execute #
#######################

simulate = False
config = build_config(state)

if simulate:
    simulation_config = SimulationConfig(duration=30000)
    job = qmm.simulate(config, gate_optimization, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Cost function
    def cost_function(value, goal):
        return (value - goal)**2

    qm = qmm.open_qm(config)
    job = qm.execute(gate_optimization)
    cost = 0.1          #  Initial value > 0
    probability = 0.0   #  Initial value
    probability_vec, cost_vec = [], []      # output data vectors (one dimension for each qubit)
    probability_temp, cost_temp = [], []    # temporary vector for computation over the individual qubits
    q = 0
    plt.figure()
    plt.suptitle(f"Qubit {q}")
    while q < len(qubits):
        while not job.is_paused():
            time.sleep(1e-3)
        probability = qm.get_io1_value()["fixed_value"]
        if probability != -2.0:
            probability_temp.append(probability)
            cost = cost_function(probability, goal)
            cost_temp.append(cost)
            qm.set_io2_value(cost)
            plt.subplot(121)
            plt.cla()
            plt.plot(amps[:len(probability_temp)], probability_temp, 'o')
            plt.title("probability")
            plt.xlabel("Amplitude pre-factor")
            plt.subplot(122)
            plt.cla()
            plt.plot(amps[:len(cost_temp)], cost_temp, 'o')
            plt.title("Cost value")
            plt.xlabel("Amplitude pre-factor")
            plt.tight_layout()
            plt.pause(0.001)
        elif q < len(qubits)-1:
            probability_vec.append(probability_temp)
            cost_vec.append(cost_temp)
            probability_temp = []
            cost_temp = []
            q += 1
            plt.figure()
            plt.suptitle(f"Qubit {q}")
        else:
            q += 1

        job.resume()
