"""
Perform the 2D qubit spec as func of amp and flux
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


##################
# State and QuAM #
##################
experiment = "2D_qubit_spectroscopy_amp_vs_flux"
debug = True
simulate = False
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
wait_time = 200
flux_point = "readout" ""
# machine.get_qubit_gate(0, gate_shape).length = 1e-6
machine.get_sequence_state(0, "qubit_spectroscopy").length = (
    machine.get_qubit_gate(0, gate_shape).length + wait_time * 4e-9
)
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 4e3

# Qubit pulse amplitude scan
a_min = 0.01
a_max = 1.99
na = 51
amps = np.linspace(a_min, a_max, na)
# Flux bias scan
bias_min = [-0.1]
bias_max = [0.05]
dbias = 0.005
bias = [np.arange(bias_min[i], bias_max[i] + dbias / 2, dbias) for i in range(len(qubit_list))]
# Ensure that flux biases remain in the [-0.5, 0.5) range
for i in qubit_list:
    assert np.all(bias[i] + machine.get_flux_bias_point(i, "zero_frequency_point").value < 0.5)
    assert np.all(bias[i] + machine.get_flux_bias_point(i, "zero_frequency_point").value >= -0.5)

with program() as qubit_spec:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    a = declare(fixed)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    b = declare(fixed)

    for q in range(len(qubit_list)):
        if not simulate:
            cooldown_time = 5 * int(machine.qubits[q].t1 * 1e9) // 4
        else:
            cooldown_time = 16
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, flux_point).value)
        # Pre-factors to apply in order to get the bias scan
        pre_factors = bias[q] / machine.get_sequence_state(0, "qubit_spectroscopy").amplitude

        with for_(n[q], 0, n[q] < n_avg, n[q] + 1):
            with for_(*from_array(b, pre_factors)):
                with for_(*from_array(a, amps)):
                    play("qubit_spectroscopy" * amp(b), machine.qubits[q].name + "_flux")
                    wait(wait_time, machine.qubits[q].name)
                    play("x180" * amp(a), machine.qubits[q].name)
                    align()
                    wait(16, machine.readout_resonators[q].name)
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I[q]),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q[q]),
                    )
                    wait(cooldown_time, machine.readout_resonators[q].name)
                    save(I[q], I_st[q])
                    save(Q[q], Q_st[q])
            save(n[q], n_st[q])

        align()

    with stream_processing():
        for q in range(len(qubit_list)):
            I_st[q].buffer(len(amps)).buffer(len(bias[q])).average().save(f"I{q}")
            Q_st[q].buffer(len(amps)).buffer(len(bias[q])).average().save(f"Q{q}")
            n_st[q].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=10000)
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for q in range(len(qubit_list)):
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["I"] = data[0]
            qubit_data[q]["Q"] = data[1]
            qubit_data[q]["iteration"] = data[2]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
            if debug:
                plot_demodulated_data_2d(
                    amps * machine.get_qubit_gate(q, gate_shape).angle2volt.deg180,
                    bias[q] + machine.get_flux_bias_point(q, flux_point).value,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "Microwave drive amplitude [V]",
                    "Flux bias [V]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
