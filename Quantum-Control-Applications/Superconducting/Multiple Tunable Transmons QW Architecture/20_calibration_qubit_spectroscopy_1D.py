"""
performs 1D qubit spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from macros import *


from scipy import signal
# import warnings
# from scipy.optimize import OptimizeWarning
# warnings.simplefilter("ignore", RuntimeWarning)
# warnings.simplefilter("ignore", OptimizeWarning)
def plot_demodulated_data_1d(
    x: np.ndarray,
    I: np.ndarray,
    Q: np.ndarray,
    x_label: str = None,
    title: str = None,
    amp_and_phase: bool = True,
    fig=None,
    plot_options: dict = None,
):
    """
    Plots 1D graphs of either 'I' and 'Q' or the corresponding amplitude and phase in a single figure.

    :param x: 1D numpy array representing the x-axis.
    :param I: 1D numpy array representing the 'I' quadrature.
    :param Q: 1D numpy array representing the 'Q' quadrature.
    :param x_label: name for the x-axis label.
    :param title: title of the plot.
    :param amp_and_phase: if `True` (default value), plots the amplitude [np.sqrt(I**2 + Q**2)] and phase [signal.detrend(np.unwrap(np.angle(I + 1j * Q)))] instead of I and Q.
    :param fig: a matplotlib figure. If `none` (default), will create a new one.
    :param plot_options: dictionary containing various plotting options. Defaults are ['fontsize': 14, 'color': 'b', 'marker': 'o', 'linewidth': 0, 'figsize': None].
    :return: the matplotlib figure object.
    """
    _plot_options = {
        "fontsize": 14,
        "color": "b",
        "marker": "o",
        "linewidth": 0,
        "figsize": None,
    }
    if plot_options is not None:
        for key in plot_options:
            if key in _plot_options.keys():
                _plot_options[key] = plot_options[key]
            else:
                raise ValueError(
                    f"The key '{key}' in 'plot_options' doesn't exists. The available options are {[option for option in _plot_options.keys()]}"
                )

    if amp_and_phase:
        y1 = np.sqrt(I**2 + Q**2)
        y2 = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        y1_label = "Amplitude [a.u.]"
        y2_label = "Phase [rad]"
    else:
        y1 = I
        y2 = Q
        y1_label = "I [a.u.]"
        y2_label = "Q [a.u.]"

    if fig is None:
        fig = plt.figure(figsize=_plot_options["figsize"])
    plt.suptitle(title, fontsize=_plot_options["fontsize"] + 2)
    ax1 = plt.subplot(211)
    plt.cla()
    plt.plot(
        x,
        y1,
        color=_plot_options["color"],
        marker=_plot_options["marker"],
        linewidth=_plot_options["linewidth"],
    )
    plt.ylabel(y1_label, fontsize=_plot_options["fontsize"])
    plt.xticks(fontsize=_plot_options["fontsize"])
    plt.yticks(fontsize=_plot_options["fontsize"])
    ax2 = plt.subplot(212, sharex=ax1)
    plt.cla()
    plt.plot(
        x,
        y2,
        color=_plot_options["color"],
        marker=_plot_options["marker"],
        linewidth=_plot_options["linewidth"],
    )
    plt.xlabel(x_label, fontsize=_plot_options["fontsize"])
    plt.ylabel(y2_label, fontsize=_plot_options["fontsize"])
    plt.xticks(fontsize=_plot_options["fontsize"])
    plt.yticks(fontsize=_plot_options["fontsize"])
    plt.pause(0.001)
    plt.tight_layout()
    return fig, [ax1, ax2]


def perform_fit(x_data, y_data, fitting_function, plot=None):
    try:
        fit = fitting_function(x_data,y_data)
        if plot is not None:
            plot[0].sca(plot[1])
            fitting_function(x_data,y_data,plot=True)
            plt.pause(0.01)
        return fit
    except (Exception,):
        return None

##################
# State and QuAM #
##################
experiment = "1D_qubit_spectroscopy"
debug = True
simulate = False
fit_data = True
live = True
qubit_list = [1,0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
machine.network.qop_ip = "172.16.2.103"
machine.network.port = 82
machine.get_qubit_gate(0, gate_shape).angle2volt.deg180 = 0.2
machine.get_qubit_gate(0, gate_shape).length = 200e-9
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 4e2

# Frequency scan
freq_span = 10e6
df = 0.1e6
freq = [
    np.arange(machine.get_qubit_IF(i) - freq_span, machine.get_qubit_IF(i) + freq_span + df / 2, df) for i in qubit_list
]

with program() as qubit_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    f = declare(int)
    b = declare(fixed)

    for i,q in enumerate(qubit_list):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "readout").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(f, freq[i])):
                update_frequency(machine.qubits[q].name, f)
                play("x180", machine.qubits[q].name)
                align()
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                )
                wait_cooldown_time(5 * machine.qubits[q].t1, simulate)
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i,q in enumerate(qubit_list):
            I_st[i].buffer(len(freq[i])).average().save(f"I{q}")
            Q_st[i].buffer(len(freq[i])).average().save(f"Q{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open quantum machine
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(qubit_spec)
    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    # Create the fitting object
    if fit_data:
        Fit = fitting.Fit()
        fit = None

    # Loop over the qubits
    for i,q in enumerate(qubit_list):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        else:
            fig = None
        print("Qubit " + str(q))
        qubit_data[i]["iteration"] = 0
        qubit_data[i]["scan_axis_1"] = freq[i] + machine.drive_lines[q].lo_freq
        # Get results from QUA program
        if live:
            my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="live")
            while my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:
                # Fetch results
                data = my_results.fetch_all()
                qubit_data[i]["I"] = data[0]
                qubit_data[i]["Q"] = data[1]
                qubit_data[i]["amplitude"] = np.sqrt(qubit_data[i]["I"]**2 + qubit_data[i]["Q"]**2)
                qubit_data[i]["phase"] = signal.detrend(np.unwrap(np.angle(qubit_data[i]["I"] + 1j * qubit_data[i]["Q"])))
                qubit_data[i]["iteration"] = data[2]
                # Progress bar
                progress_counter(qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time)
                # live plot
                if debug:
                    fig, axes = plot_demodulated_data_1d(
                        qubit_data[i]["scan_axis_1"],
                        qubit_data[i]["I"],
                        qubit_data[i]["Q"],
                        "Microwave drive frequency [Hz]",
                        f"{experiment}_qubit{q}",
                        amp_and_phase=True,
                        fig=fig,
                        plot_options={"marker": "."},
                    )
                    if fit_data:
                        fit0 = perform_fit(x_data=qubit_data[i]["scan_axis_1"], y_data=qubit_data[i]["amplitude"],
                                    fitting_function=Fit.reflection_resonator_spectroscopy, plot=(fig, axes[0]))
                        if fit0 is not None:
                            fit = fit0
                elif fit_data:
                    fit0 = perform_fit(x_data=qubit_data[i]["scan_axis_1"], y_data=qubit_data[i]["amplitude"],
                                       fitting_function=Fit.reflection_resonator_spectroscopy)
                    if fit0 is not None:
                        fit = fit0
        else:
            my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="wait_for_all")
            data = my_results.fetch_all()
            qubit_data[i]["I"] = data[0]
            qubit_data[i]["Q"] = data[1]
            qubit_data[i]["amplitude"] = np.sqrt(qubit_data[i]["I"] ** 2 + qubit_data[i]["Q"] ** 2)
            qubit_data[i]["phase"] = signal.detrend(np.unwrap(np.angle(qubit_data[i]["I"] + 1j * qubit_data[i]["Q"])))
            qubit_data[i]["iteration"] = data[2]
            if debug:
                fig, axes = plot_demodulated_data_1d(
                    qubit_data[i]["scan_axis_1"],
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Microwave drive frequency [Hz]",
                    f"{experiment}_qubit{q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"marker": "."},
                )
            if fit_data:
                fit0 = perform_fit(x_data=qubit_data[i]["scan_axis_1"], y_data=qubit_data[i]["amplitude"],
                                   fitting_function=Fit.reflection_resonator_spectroscopy)
                if fit0 is not None:
                    fit = fit0
        # Update state with new resonance frequency
        if fit_data:
            print(f"Qubit frequency: {machine.qubits[q].f_01 * 1e-9:.6f} GHz")
            machine.qubits[q].f_01 = np.round(fit["f"][0])
            print(f"New qubit frequency: {machine.qubits[q].f_01 * 1e-9:.6f} GHz")
            print(f"New resonance IF frequency: {machine.get_qubit_IF(q) * 1e-6:.3f} MHz")
        # Exit qubit for loop if Execution stopped by user
        if live:
            if my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:
                break

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
