"""
resonator_spec.py: performs 1D resonator spectroscopy for multiple qubits
"""
import matplotlib

matplotlib.use("TKagg")
from state_and_config import build_config, quam
from qm.simulate.credentials import create_credentials
from qm.simulate import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qualang_tools.units import unit
from qualang_tools.results import fetching_tool

###################
# The QUA program #
###################
u = unit()

qubits = [0, 1]
n_avg = 4
cooldown_time = 16 * u.ns // 4

f_center = [round(quam["readout_resonators"][q]["f_res"] - quam["readout_lines"][0]["lo_freq"]) for q in qubits]
f_span = 1 * u.MHz
df = 1 * u.MHz
n_points = (2 * f_span + df) // df

with program() as resonator_spec:
    n = declare(int)
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = [declare_stream() for q in qubits]
    Q_st = [declare_stream() for q in qubits]

    idx = 0
    for q in qubits:
        align()

        with for_(n, 0, n < n_avg, n + 1):
            # Notice it's <= to include f_max (This is only for integers!)
            with for_(f, f_center[q] - f_span, f <= f_center[q] + f_span, f + df):
                update_frequency(f"rr{q}", f)
                measure(
                    "readout",
                    f"rr{q}",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                wait(cooldown_time, f"rr{q}")
                save(I, I_st[idx])
                save(Q, Q_st[idx])

        idx += 1

    with stream_processing():
        for idx in range(len(qubits)):
            I_st[idx].buffer(n_points).average().save(f"I_q{qubits[idx]}")
            Q_st[idx].buffer(n_points).average().save(f"Q_q{qubits[idx]}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(
    host="nord-quantique-d14d58b1.quantum-machines.co",
    port=443,
    credentials=create_credentials(),
)

#######################
# Simulate or execute #
#######################

simulate = True
config = build_config(quam)

if simulate:
    simulation_config = SimulationConfig(duration=20000)
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)

# Get results from QUA program
data_list = sum([[f"I_q{q}", f"Q_q{q}"] for q in range(len(qubits))], [])
results = fetching_tool(job, data_list, mode="live")

# Plot results
fig, axs = plt.subplots(2, len(qubits))
plotted_data = []
x_axis = []
for q in range(len(qubits)):
    ax = axs[0, q]
    ax.set_title(f"rr{q} spectroscopy amplitude")
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
    plotted_data.append(None)
    ax = axs[1, q]
    ax.set_title(f"rr{q} spectroscopy phase")
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("Phase [rad]")
    plotted_data.append(None)
    # Get frequency vector for each qubit
    x_axis.append(np.arange(f_center[q] - f_span, f_center[q] + f_span + 0.1, df))

# Live plot
while results.is_processing():
    # Fetch results
    data = results.fetch_all()
    for q in range(len(qubits)):
        # Get I and Q data for each qubit
        I, Q = data[2 * q], data[2 * q + 1]
        phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        # Plot results
        ax = axs[0, q]
        if plotted_data[2 * q] is not None:
            plotted_data[2 * q].remove()
        (plotted_data[2 * q],) = ax.plot(x_axis[q] / u.MHz, np.sqrt(I**2 + Q**2), ".")
        ax = axs[1, q]
        if plotted_data[2 * q + 1] is not None:
            plotted_data[2 * q + 1].remove()
        (plotted_data[2 * q + 1],) = ax.plot(x_axis[q] / u.MHz, phase, ".")
    plt.tight_layout()
    plt.pause(1.0)
