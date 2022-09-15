import os

import qcodes as qc
from qcodes import initialise_or_create_database_at, load_or_create_experiment

from opx_single_point_readout import *
from qcodes.utils.dataset.doNd import do2d
from configuration import *

db_name = "QM_demo_reflectometry.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "2d_scan_opx_single"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)
VP1 = MyCounter(name="vp1")
VP2 = MyCounter(name="vp2")
opx_single_point = OPXSinglePointReadout(config, host = "172.16.2.103", port=85)
opx_single_point.f(300.8509e6)
opx_single_point.t_meas(0.010)
opx_single_point.n_avg(1)
opx_single_point.amp(1.0)
opx_single_point.readout_pulse_length(readout_pulse_length)
full_data = QMDemodParameters(
    opx_single_point,
    ["I", "Q", "R", "Phi"],
    "Spectrum",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
)
opx_single_point.live_plot = False
opx_single_point.live_in_python = False
station = qc.Station()
station.add_component(opx_single_point)
do2d(
    VP1,
    10,
    20,
    10,
    1,
    VP2,
    10,
    20,
    10,
    0.0135,
    opx_single_point.resume,
    full_data,
    enter_actions=[opx_single_point.run_exp],
    exit_actions=[opx_single_point.halt],
    show_progress=True,
)
