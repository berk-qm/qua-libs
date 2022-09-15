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
from opx_raster_scan import *
from qcodes.utils.dataset.doNd import do2d, do0d
from configuration import *
import matplotlib.pyplot as plt
import IPython.lib.backgroundjobs as bg
from plottr.apps import inspectr


db_name = "QM_demo_raster.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "2d_raster_scan_opx_single"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)



opx_raster = OPXRasterScan(config, host="172.16.2.103", port=85)
opx_raster.Vx_min(-0.02)
opx_raster.Vx_max(0.02)
opx_raster.Nx(100)
opx_raster.Vy_min(0.233)
opx_raster.Vy_max(0.248)
opx_raster.Ny(120)
opx_raster.n_avg(20)

full_data = QMDemodParameters(
    opx_raster,
    ["I", "Q", "R", "Phi", "Vx", "Vy"],
    "Raster_scan",
    names=["I", "Q", "R", "Phi", "Vx", "Vy"],
    units=["V", "V", "V", "°", "V", "V"],
    shapes=((opx_raster.Ny(),opx_raster.Nx()), )* 6,
    setpoints=((opx_raster.Vy_axis(),opx_raster.Vx_axis()),) * 6,
)
station = qc.Station()
station.add_component(opx_raster)

do0d(opx_raster.simulate_exp(50000), full_data)


# jobs = bg.BackgroundJobManager()
# jobs.new(inspectr.main, db_file_path)
# do0d(opx_raster.run_exp(), full_data, exp=experiment, do_plot=False)
# station.remove_component("OPX")
from qcodes.dataset.plotting import plot_dataset
# ax, cb = plot_dataset(full_data)
# plt.pcolor(full_data.get()[2])