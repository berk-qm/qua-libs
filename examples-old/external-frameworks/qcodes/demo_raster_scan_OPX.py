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
from opx_raster_scan import *


##########################################
# Experiment and database initialization #
##########################################
db_name = "QM_demo_raster.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "2d_raster_scan_opx_single"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

######################
# Program parameters #
######################
opx_raster = OPXRasterScan(config, host=qop_ip, port=qop_port)
opx_raster.readout_element("resonator")  # readout_element
opx_raster.x_element("G2")  # x_element
opx_raster.y_element("G1")  # y_element
opx_raster.gate_operation("jump")  # gate_operation
opx_raster.readout_operation("readout")  # gate_operation
opx_raster.Vx_min(-0.02)
opx_raster.Vx_max(0.02)
opx_raster.Nx(100)
opx_raster.Vy_min(0.233)
opx_raster.Vy_max(0.248)
opx_raster.Ny(120)
opx_raster.wait_time(100)  # Wait time before measuring [ns]
opx_raster.n_avg(10000)  # Number of averages

full_data = QMDemodParameters(
    opx_raster,
    ["I", "Q", "R", "Phi"],
    "Raster_scan",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
    shapes=((opx_raster.Ny(), opx_raster.Nx()),) * 4,
    setpoints=((opx_raster.Vy_axis(), opx_raster.Vx_axis()),) * 4,
)
station = qc.Station()
station.add_component(opx_raster)
#####################
# Program execution #
#####################
simulate = False

if simulate:
    # Simulate program
    opx_raster.sim_time(10000)
    opx_raster.simulate_exp()
    opx_raster.plot_simulated_wf()
else:
    # Execute and plot
    opx_raster.live_plot = True
    opx_raster.live_in_python = True
    if opx_raster.live_plot:
        if opx_raster.live_in_python:
            # Live plot the results using matplotlib
            do0d(opx_raster.run_exp, full_data, exp=experiment, do_plot=False)
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
                enter_actions=[opx_raster.run_exp],
                exit_actions=[opx_raster.halt],
                write_period=0.1,
                exp=experiment,
                do_plot=False,
            )
    else:
        jobs = bg.BackgroundJobManager()
        jobs.new(inspectr.main, db_file_path)
        do0d(opx_raster.run_exp, full_data, exp=experiment, do_plot=False)
