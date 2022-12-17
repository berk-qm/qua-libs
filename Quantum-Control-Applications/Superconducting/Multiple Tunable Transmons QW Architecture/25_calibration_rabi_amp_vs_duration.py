"""
performs 2D Rabi chevron gate amplitude versus gate duration
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from macros import *

##################
# State and QuAM #
##################
experiment = "2D_Rabi_chevron_amp_vs_length"
debug = True
simulate = False
charge_lines = [0, 1]
injector_list = [0, 1]
digital = [1, 2, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
qubit_list = [0, 1, 2, 3, 4, 5]  # you can shuffle the order at which you perform the experiment

gate_params = []
for i, q in enumerate(qubit_list):
    gate_params.append((machine.get_qubit_gate(q, gate_shape).length, machine.get_qubit_gate(q, gate_shape).angle2volt.deg180))
    machine.get_qubit_gate(q, gate_shape).length = 16e-9
    machine.get_qubit_gate(q, gate_shape).angle2volt.deg180 = 0.4

config = machine.build_config(digital, qubit_list, injector_list, charge_lines, gate_shape)

for i, q in enumerate(qubit_list):
    machine.get_qubit_gate(q, gate_shape).length = gate_params[i][0]
    machine.get_qubit_gate(q, gate_shape).angle2volt.deg180 = gate_params[i][1]


###################
# The QUA program #
###################
n_avg = 4e1

# Frequency scan
length_min = 16 // 4
length_max = 2000 // 4
dlength = 8 // 4
lengths = np.arange(length_min, length_max, dlength)
# Flux bias scan
a_min = 0
a_max = 1
da = 0.05
amplitudes = np.arange(a_min, a_max + da / 2, da)


# QUA program
with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    t = declare(int)
    a = declare(fixed)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        for j, z in enumerate(qubit_and_charge_relation):
            if q == z:
                set_dc_offset(machine.qubits[q].name + "_charge", "single",
                              machine.get_charge_bias_point(j, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(a, amplitudes)):
                with for_(*from_array(t, lengths)):
                    play("x180" * amp(a), machine.qubits[q].name, duration=t)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("cos", I[i], "out1"),
                        demod.full("sin", Q[i], "out1"),
                    )
                    wait_cooldown_time_fivet1(q, machine, simulate)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i,q in enumerate(qubit_list):
            I_st[i].buffer(len(lengths)).buffer(len(amplitudes)).average().save(f"I{q}")
            Q_st[i].buffer(len(lengths)).buffer(len(amplitudes)).average().save(f"Q{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=20000)
    job = qmm.simulate(config, rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(rabi)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for i,q in enumerate(qubit_list):
        print("Qubit " + str(q))
        qubit_data[i]["iteration"] = 0
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[i]["I"] = data[0]
            qubit_data[i]["Q"] = data[1]
            qubit_data[i]["iteration"] = data[2]
            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
            if debug:
                plot_demodulated_data_2d(
                    lengths * 4,
                    # amplitudes * machine.get_qubit_gate(q, gate_shape).angle2volt.deg180,
                    amplitudes * 0.4,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Gate length [ns]",
                    "Gate amplitude [V]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
