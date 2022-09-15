import os
from plottr.apps import inspectr
import IPython.lib.backgroundjobs as bg
import qcodes as qc
from qcodes import (
    load_or_create_experiment,
    initialise_or_create_database_at,
)
from qcodes.utils.dataset.doNd import do0d

from configuration import *
from opx_refelectometry_freq_scan import *

db_name = "QM_demo_reflectometry.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "reflectometry"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

station = qc.Station()
opx_freq_scan = OPXSpectrumScan(config, host = "172.16.2.103", port=85)
station.add_component(opx_freq_scan)

opx_freq_scan.f_start(200.85e6)
opx_freq_scan.f_stop(309.88e6)
opx_freq_scan.n_points(200)
opx_freq_scan.t_meas(0.00010)
opx_freq_scan.n_avg(1000)
opx_freq_scan.amp(1.0)
opx_freq_scan.readout_pulse_length(readout_pulse_length)

full_data = QMDemodParameters(
    opx_freq_scan,
    ["I", "Q", "R", "Phi"],
    "Spectrum",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
    # The following two lines tell it that it will return a vector of items!
    shapes=((opx_freq_scan.n_points(),),) * 4,
    setpoints=((opx_freq_scan.freq_axis(),),) * 4,
)
opx_freq_scan.live_plot = True
opx_freq_scan.live_in_python = True
# jobs = bg.BackgroundJobManager()
# jobs.new(inspectr.main, db_file_path)
do0d(opx_freq_scan.run_exp, full_data)

## Change something in the config
config["waveforms"]["reflect_wf"]["sample"] = 0.3  # Change amplitude
opx_freq_scan.set_config(config)

## Close connection (to start different experiment)
station.remove_component(opx_freq_scan.name)
