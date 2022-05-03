from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

tmax = 300  # clock cycles
dt = 1  # clock cycles
tvec = np.arange(0, tmax, dt)
sig1 = 'out1'
sig2 = 'out1'

config['elements']['resonator_aux'] = config['elements']['resonator']
config['elements']['resonator_aux']['operations']['zero'] = 'zero_pulse'
config['integration_weights']['cosine_weights']['cosine'] = [(10000.0, tmax * 4)]
config['integration_weights']['cosine_weights']['sine'] = [(0.0, tmax * 4)]
config['pulses']['zero_pulse'] = {'operation': 'measurement',
                                  'length': 16,
                                  'waveforms': {'I': 'zero_wf', 'Q': 'zero_wf'},
                                  'integration_weights': {'cos': 'cosine_weights'},
                                  'digital_marker': 'ON'}

with program() as correlations:

    n = declare(int)
    t = declare(int)
    vec = [declare(fixed, size=len(tvec)) for i in range(2)]
    corr_st = declare_stream()
    corr_ = declare(fixed)

    with for_(n, 0, n < 100, n + 1):
        with for_(t, 0, t < len(tvec), t + dt):

            wait(4, "resonator")
            measure("readout", "resonator", None, integration.sliced("cos", vec[0], dt, sig1))
            wait(4 + t, "resonator_aux")
            measure("zero", "resonator_aux", None, integration.sliced("cos", vec[1], dt, sig2))
            assign(corr_, Math.dot(vec[0], vec[1]))
            save(corr_, corr_st)

    with stream_processing():
        corr_st.buffer(len(tvec)).average().save('corr')


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

qm = qmm.open_qm(config)

job = qm.execute(correlations)
res_handles = job.result_handles
corr_handle = res_handles.get("corr")
res_handles.wait_for_all_values()
corr = corr_handle.fetch_all()
t = tvec * 4  # transform to ns
plt.plot(t, corr)
