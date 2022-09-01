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
from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close

###################
# The QUA program #
###################
u = unit()

qubits = [0, 1]
n_avg = 4
cooldown_time = 16 * u.ns // 4

f_center = [round(quam["readout_resonators"][q]["f_res"] - quam["readout_lines"][0]["lo_freq"]) for q in qubits]
f_span = 2 * u.MHz
df = 1 * u.MHz
n_points = (2 * f_span + df) // df
f_vec = [np.arange(f_center[q]-f_span, f_center[q]+f_span+0.1, df) for q in qubits]

with program() as resonator_spec:
    n = declare(int)
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    qubits_qua = declare(int, value=qubits)
    qb_index = declare(int)
    f_vec_qua = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(qb_index, qubits_qua):
            with switch_(qb_index):
                for j in qubits:
                    with case_(j):
                        with for_(*from_array(f, f_vec[j])):
                            update_frequency(f"rr{j}", f)
                            measure(
                                "readout",
                                f"rr{j}",
                                None,
                                dual_demod.full("cos", "out1", "sin", "out2", I),
                                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                            )
                            wait(cooldown_time, f"rr{j}")
                            save(I, I_st)
                            save(Q, Q_st)


    with stream_processing():
        I_st.buffer(n_points).buffer(len(qubits)).average().save("I")
        Q_st.buffer(n_points).buffer(len(qubits)).average().save("Q")

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
results = fetching_tool(job, ["I", "Q"], mode="live")

# Plot results
fig, axs = plt.subplots(2, len(qubits))
interrupt_on_close(fig, job)
# Live plot
while results.is_processing():
    # Fetch results
    I, Q = results.fetch_all()

    for q in range(len(qubits)):
        # Get I and Q data for each qubit
        I_q, Q_q = I[q], Q[q]
        phase = signal.detrend(np.unwrap(np.angle(I_q + 1j * Q_q)))
        # Plot results
        plt.subplot(2, len(qubits), q+1)
        plt.cla()
        plt.plot(f_vec[q] / u.MHz, np.sqrt(I_q**2 + Q_q**2), "-")
        plt.title(f"rr{q} spectroscopy amplitude")
        plt.xlabel("frequency [MHz]")
        plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
        plt.subplot(2, len(qubits), q + 1 + len(qubits))
        plt.cla()
        plt.plot(f_vec[q] / u.MHz, phase, "-")
        plt.title(f"rr{q} spectroscopy phase")
        plt.xlabel("frequency [MHz]")
        plt.ylabel("Phase [rad]")
    plt.tight_layout()
    plt.pause(0.01)
