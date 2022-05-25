import qua_simple_tableau as qst
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
from qm import LoopbackInterface
import numpy as np
from configuration import *
from qm.simulate.credentials import create_credentials  # only when simulating

with program() as test:
    mat1 = declare(int, value=33825)
    mat2 = declare(int, value=33825)
    mat3 = declare(int, value=0)
    col_mask = declare(int, value=4369)
    save(mat3, 'mat3')

    qst.mat_mul_qua(mat1, mat2, mat3)
    save(mat3, 'mat3')


qmm = QuantumMachinesManager(host='yuval-2539185a.dev.quantum-machines.co',port =443,credentials=create_credentials())
job_sim = qmm.simulate(config, test, SimulationConfig(1000))

res = job_sim.result_handles
print(res.mat3.fetch_all())