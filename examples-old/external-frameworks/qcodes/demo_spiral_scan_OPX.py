import os

import qcodes as qc
from qcodes import initialise_or_create_database_at, load_or_create_experiment
from qcodes import (Measurement,
                    experiments,
                    initialise_database,
                    initialise_or_create_database_at,
                    load_by_guid,
                    load_by_run_spec,
                    load_experiment,
                    load_last_experiment,
                    load_or_create_experiment,
                    new_experiment,
                    ManualParameter)
from qcodes.dataset.plotting import plot_dataset
from opx_spiral_scan import *
from qcodes.utils.dataset.doNd import do2d, do0d, do1d
from configuration import *
import matplotlib.pyplot as plt
import IPython.lib.backgroundjobs as bg
from plottr.apps import inspectr


db_name = "QM_demo_spiral.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "2d_spiral_scan_opx_single"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

opx_spiral = OPXSpiralScan(config, host="172.16.2.103", port=85)
opx_spiral.Vx_center(0.0)
opx_spiral.Vx_span(0.04)  # Scan defined as center +/- span/2
opx_spiral.Vy_center(0.250)
opx_spiral.Vy_span(0.1)   # Scan defined as center +/- span/2
opx_spiral.N_points(51)  # Must be odd !
opx_spiral.wait_time(100)
opx_spiral.n_avg(10000)

full_data = QMDemodParameters(
    opx_spiral,
    ["I", "Q", "R", "Phi"],
    "spiral_scan",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
    shapes=((opx_spiral.N_points(), opx_spiral.N_points()), ) * 4,
    setpoints=((opx_spiral.Vx_axis(), opx_spiral.Vy_axis()),) * 4,
)
station = qc.Station()
station.add_component(opx_spiral)

simulate = False
COUNTER = MyCounter(name="counter")
if simulate:
    # Simulate program
    do0d(opx_spiral.simulate_exp(10000), full_data)
else:
    # Execute and plot
    opx_spiral.live_plot = True
    opx_spiral.live_in_python = True
    if opx_spiral.live_plot:
        if opx_spiral.live_in_python:
            do0d(opx_spiral.run_exp, full_data, exp=experiment, do_plot=False)
        else:
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
                do_plot=False,
            )
    else:
        jobs = bg.BackgroundJobManager()
        jobs.new(inspectr.main, db_file_path)
        do0d(opx_spiral.run_exp, full_data, exp=experiment, do_plot=False)