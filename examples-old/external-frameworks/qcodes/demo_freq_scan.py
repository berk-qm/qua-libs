import os
from plottr.apps import inspectr
import IPython.lib.backgroundjobs as bg
import qcodes as qc
from qcodes import (
    load_or_create_experiment,
    initialise_or_create_database_at,
)
from qcodes.utils.dataset.doNd import do0d, do1d

from configuration import *
from opx_refelectometry_freq_scan import *

##########################################
# Experiment and database initialization #
##########################################

db_name = "QM_demo_reflectometry.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "reflectometry"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

######################
# Program parameters #
######################

opx_freq_scan = OPXSpectrumScan(config, host=qop_ip, port=qop_port)
opx_freq_scan.readout_element("resonator")  # readout_element
opx_freq_scan.readout_operation("readout")  # gate_operation
opx_freq_scan.f_start(200.85e6)
opx_freq_scan.f_stop(309.88e6)
opx_freq_scan.n_f(21)
opx_freq_scan.a_start(-1)
opx_freq_scan.a_stop(1)
opx_freq_scan.n_a(51)
opx_freq_scan.n_avg(100000)
opx_freq_scan.amp(1.0)
opx_freq_scan.readout_pulse_length(readout_pulse_length)

## Change something in the config
config["waveforms"]["reflect_wf"]["sample"] = 0.3  # Change amplitude
opx_freq_scan.set_config(config)

full_data = OPXSpectrumParameters(
    opx_freq_scan,
    ["I", "Q", "R", "Phi"],
    "Spectrum",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
    # The following two lines tell it that it will return a vector of items!
    #shapes=((opx_freq_scan.n_f(), opx_freq_scan.n_a(),),) * 4,
    #setpoints=((opx_freq_scan.freq_axis(), (opx_freq_scan.amp_axis() if opx_freq_scan.n_a()>1 else None)),) * 4,
)

station = qc.Station()
station.add_component(opx_freq_scan)

#####################
# Program execution #
#####################
simulate = False

if simulate:
    # Simulate program
    opx_freq_scan.sim_time(10000)
    opx_freq_scan.simulate_exp()
    opx_freq_scan.plot_simulated_wf()
else:
    # Execute and plot
    opx_freq_scan.live_plot = True
    opx_freq_scan.live_in_python = True
    if opx_freq_scan.live_plot:
        if opx_freq_scan.live_in_python:
            # Live plot the results using matplotlib
            do0d(opx_freq_scan.run_exp, full_data, exp=experiment, do_plot=False)
        else:
            # 'Live' plot the results using plottr
            COUNTER = MyCounter(name="counter")
            jobs = bg.BackgroundJobManager()
            jobs.new(inspectr.main, db_file_path)
            do1d(
                COUNTER,
                1,
                40,
                40,
                0.01,
                full_data,
                enter_actions=[opx_freq_scan.run_exp],
                exit_actions=[opx_freq_scan.halt],
                write_period=0.1,
                exp=experiment,
                do_plot=False,
            )
    else:
        jobs = bg.BackgroundJobManager()
        jobs.new(inspectr.main, db_file_path)
        do0d(opx_freq_scan.run_exp, full_data, exp=experiment, do_plot=False)
## Close connection (to start different experiment)
station.remove_component(opx_freq_scan.name)
