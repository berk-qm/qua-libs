import os

# QCodes
import qcodes as qc
from qcodes import (
    initialise_or_create_database_at,
    load_or_create_experiment,
)
from qcodes.utils.dataset.doNd import do0d, do1d

# Plottr
import IPython.lib.backgroundjobs as bg
from plottr.apps import inspectr

# OPX drivers
from configuration import *
from opx_spiral_scan import *


##########################################
# Experiment and database initialization #
##########################################
db_name = "QM_demo_spiral.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "2d_spiral_scan_opx_single"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

######################
# Program parameters #
######################
#
opx_spiral = OPXSpiralScan(config, host=qop_ip, port=qop_port)
opx_spiral.readout_element("resonator")  # readout_element
opx_spiral.x_element("G2")  # x_element
opx_spiral.y_element("G1")  # y_element
opx_spiral.gate_operation("jump")  # gate_operation
opx_spiral.readout_operation("readout")  # gate_operation
opx_spiral.Vx_center(0.0)  # Center of the spiral scan
opx_spiral.Vx_span(0.04)  # Scan defined as center +/- span/2
opx_spiral.Vy_center(0.250)  # Center of the spiral scan
opx_spiral.Vy_span(0.1)  # Scan defined as center +/- span/2
opx_spiral.n_points(51)  # Must be odd !
opx_spiral.wait_time(100)  # Wait time before measuring
opx_spiral.n_avg(10000)  # Number of averages
# Result parameters after demodulation
full_data = QMDemodParameters(
    opx_spiral,
    ["I", "Q", "R", "Phi"],
    "spiral_scan",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
    shapes=((opx_spiral.n_points(), opx_spiral.n_points()),) * 4,
    setpoints=((opx_spiral.Vx_axis(), opx_spiral.Vy_axis()),) * 4,
)
# Add the instrument opx_spiral
station = qc.Station()
station.add_component(opx_spiral)

#####################
# Program execution #
#####################
simulate = False

if simulate:
    # Simulate program
    opx_spiral.sim_time(10000)
    opx_spiral.simulate_exp()
    opx_spiral.plot_simulated_wf()
else:
    # Execute and plot
    opx_spiral.live_plot = True
    opx_spiral.live_in_python = True
    if opx_spiral.live_plot:
        if opx_spiral.live_in_python:
            # Live plot the results using matplotlib
            do0d(opx_spiral.run_exp, full_data, exp=experiment, do_plot=False)
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
                enter_actions=[opx_spiral.run_exp],
                exit_actions=[opx_spiral.halt],
                write_period=0.1,
                exp=experiment,
                do_plot=False,
            )
    else:
        jobs = bg.BackgroundJobManager()
        jobs.new(inspectr.main, db_file_path)
        do0d(opx_spiral.run_exp, full_data, exp=experiment, do_plot=False)
