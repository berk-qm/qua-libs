"""
DRAG_cal.py: performs power drag calibration
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from datetime import datetime

##################
# State and QuAM #
##################
experiment = "drag_cal"
debug = True
simulate = False
fit_data = True
qubit_list = [0, 1]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

machine.qubits[0].driving.drag_cosine.angle2volt.deg180 = 0.4
machine.qubits[1].driving.drag_cosine.angle2volt.deg180 = 0.4
config = machine.build_config(digital, qubit_list, gate_shape)

# set the drag_coef in the configuration
drag_coef = 1

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3

cooldown_time = 5 * u.us // 4

a_min = 0.2
a_max = 1
da = 0.01

amps = np.arange(a_min, a_max + da / 2, da)

iter_min = 0
iter_max = 25
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)

with program() as drag_cal:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    a = declare(fixed)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    b = declare(fixed)
    it = declare(int)
    pulses = declare(int)
    state = [declare(bool) for _ in range(len(qubit_list))]
    state_st = [declare_stream() for _ in range(len(qubit_list))]

    for i in range(len(qubit_list)):
        # bring other qubits to zero frequency
        machine.nullify_qubits(True, qubit_list, i)
        set_dc_offset(
            machine.qubits[i].name + "_flux", "single", machine.get_flux_bias_point(i, "near_anti_crossing").value
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(a, amps)):
                with for_(*from_array(it, iters)):
                    with for_(pulses, iter_min, pulses <= it, pulses + d):
                        play("x180" * amp(1, 0, 0, a), machine.qubits[i].name)
                        play("x180" * amp(-1, 0, 0, -a), machine.qubits[i].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[i].name,
                        None,
                        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I[i]),
                        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q[i]),
                    )
                    wait(cooldown_time, machine.readout_resonators[i].name)
                    assign(state[i], I[i] > machine.readout_resonators[i].ge_threshold)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    save(state[i], state_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(len(qubit_list)):
            I_st[i].buffer(len(iters)).buffer(len(amps)).average().save(f"I{i}")
            Q_st[i].buffer(len(iters)).buffer(len(amps)).average().save(f"Q{i}")
            state_st[i].boolean_to_int().buffer(len(iters)).buffer(len(amps)).average().save(f"state{i}")
            n_st[i].save(f"iteration{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, drag_cal, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(drag_cal)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]

    # Create the fitting object
    Fit = fitting.Fit()

    for q in range(len(qubit_list)):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"state{i}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["I"] = data[0]
            qubit_data[q]["Q"] = data[1]
            qubit_data[q]["state"] = data[2]
            qubit_data[q]["iteration"] = data[3]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
        #     if debug and not fit_data:
        #         plot_demodulated_data_1d(
        #             amps * machine.qubits[q].driving.drag_cosine.angle2volt.deg180,
        #             qubit_data[q]["I"],
        #             qubit_data[q]["Q"],
        #             "x180 amplitude [V]",
        #             f"Power rabi {q}",
        #             amp_and_phase=False,
        #             fig=fig,
        #             plot_options={"marker": "."},
        #         )
        #
        # # Update state with new resonance frequency
        # if fit_data:
        #     print(f"Previous x180 amplitude: {machine.qubits[q].driving.drag_cosine.angle2volt.deg180:.1f} V")
        #     machine.qubits[q].driving.drag_cosine.angle2volt.deg180 = np.round(fit_I["amp"][0])
        #     print(f"New x180 amplitude: {machine.qubits[q].driving.drag_cosine.angle2volt.deg180:.1f} V")

machine.save("./labnotebook/state_after_" + experiment + "_" + now + ".json")
machine.save("latest_quam.json")
