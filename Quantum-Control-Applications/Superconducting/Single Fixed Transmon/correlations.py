from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np


# ==== parameters ==== #
M = 4  # number of auxiliary elements
tmax = 480 // 4;dt = 1; tvec = np.arange(0, tmax, dt)  # clock cycles
print(len(tvec)/M)  # we need len(tvec)%M==0 if we don't want to deal with edge issues
sig1 = 'out1'; sig2 = 'out2'  # between which outputs of the resonator to perform the correlation measurement
readout_pules_length = 2000
tof = 248


# ==== editing the configuration ==== #
config['pulses']['readout_pulse']['length'] = readout_pules_length
config['elements']['resonator']['time_of_flight'] = tof
for i in range(M):
    config['elements'][f'resonator_aux_{i+1}'] = config['elements']['resonator']
    config['elements'][f'resonator_aux_{i+1}']['operations']['zero'] = 'zero_pulse'

config['integration_weights']['cosine_weights']['cosine'] = [(10, tmax * 4)]
config['integration_weights']['cosine_weights']['sine'] = [(0.0, tmax * 4)]
config['pulses']['zero_pulse'] = {'operation': 'measurement',
                                  'length': 16,
                                  'waveforms': {'I': 'zero_wf', 'Q': 'zero_wf'},
                                  'integration_weights': {'cos': 'cosine_weights'},
                                  'digital_marker': 'ON'}


# ==== the correlation program ===== #
with program() as correlations:

    n = declare(int)
    m = declare(int)
    t = declare(int)
    vec = [declare(fixed, size=len(tvec)) for i in range(1+M)]
    corr_st = declare_stream()
    corr_ = [declare(fixed) for i in range(M)]
    corr_vec = declare(fixed, size=len(tvec))
    idx = declare(int)

    update_frequency("resonator", int(121e6))
    with for_(m, 0, m < 2, m+1):
        with for_(n, 0, n < 100, n + 1):
            with for_(t, 0, t < len(tvec), t + dt*M):

                # state preparation
                wait(1000)
                align()

                # sliced measurements
                wait(4, "resonator")
                measure("readout", "resonator", None, integration.sliced("cos", vec[0], dt, sig1))
                for i in range(M):
                    wait(4 + t + dt * i, f"resonator_aux_{i+1}")
                    measure("zero", f"resonator_aux_{i+1}", None, integration.sliced("cos", vec[1+i], dt, sig2))
                    assign(corr_[i], Math.dot(vec[i+1], vec[0]))
                    assign(corr_vec[idx], corr_vec[idx] + corr_[i])
                    assign(idx, idx+1)
            assign(idx, 0)

        with for_(idx, 0, idx < len(tvec), idx+1):
            save(corr_vec[idx], corr_st)
            assign(corr_vec[idx], 0.0)

    with stream_processing():
        corr_st.buffer(len(tvec)).average().save('corr')


# ==== execution and analysis ==== #
qop_ip = "172.16.2.103"
qop_port = 80
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)
qm = qmm.open_qm(config)
job = qm.execute(correlations)
res_handles = job.result_handles
corr_handle = res_handles.get("corr")
res_handles.wait_for_all_values()
corr = corr_handle.fetch_all()
t = tvec * 4  # transform to ns

plt.plot(t, corr, '-*')
plt.xlabel('time in [ns]')
plt.ylabel('correlation [a.u]')