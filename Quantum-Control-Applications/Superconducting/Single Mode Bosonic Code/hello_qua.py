from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from qm.qua import *
from qm.simulate import SimulationConfig
from configuration import config

qmm = QuantumMachinesManager(host="nord-quantique-d14d58b1.quantum-machines.co", port=443, credentials=create_credentials())

with program() as hello_qua:

    adc_st = declare_stream(adc_trace=True)
    I = declare(fixed)
    I_st = declare_stream()

    play("const", "qubit")
    play("saturation", "qubit")
    play("gauss", "qubit")
    play("x180", "qubit")
    play("x90", "qubit")
    align()
    play("const", "bmode")
    align()
    play("const", "rr")
    measure("readout", "rr", adc_st, demod.full("cos", I))
    save(I, I_st)

    with stream_processing():
        adc_st.input1().save_all('adc')
        I_st.save_all('I')

job = qmm.simulate(config, hello_qua, SimulationConfig(2000))
job.get_simulated_samples().con1.plot()