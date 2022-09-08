from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from macros import qubit_frequency_tracking
from configuration import *
import matplotlib.pyplot as plt
import time
from qualang_tools.results import fetching_tool, progress_counter


######################################
#  Open Communication with the QOP  #
######################################
qmm = QuantumMachinesManager(qop_ip, port=85)

# Open quantum machine
qm = qmm.open_qm(config)

# Initialize object
freq_track_obj = qubit_frequency_tracking("qubit", "resonator", qubit_IF, ge_threshold)


########################
#  Time domain Ramsey  #
########################

n_avg = 20
tau_vec = np.arange(4, 50_000, 50)
print(f"Initial frequency: {freq_track_obj.f_res:.0f} Hz")

# Repeat the measurement twice, without and with correction of the frequency
for arg in ["Pe_initial", "Pe_corrected"]:
    # The QUA program
    with program() as prog:
        freq_track_obj.qua_declaration()
        freq_track_obj.time_domain_ramsey_full_sweep(n_avg, f_det=int(0.06e6), tau_vec=tau_vec)

        with stream_processing():

            freq_track_obj.state_estimation_st[0].buffer(len(tau_vec)).average().save(arg)
    # Execute the program
    job = qm.execute(prog)
    # Wait until processing is done before fetching results
    job.result_handles.wait_for_all_values()
    # Plot raw data + fit
    plt.figure(arg)
    freq_track_obj.time_domain_ramsey_full_sweep_analysis(job.result_handles, stream_name=arg)
    # Prepare to apply a correction on the next iteration
    print(f"Correct frequency: {freq_track_obj.f_res:.0f} Hz")

#############################
#  Frequency domain Ramsey  #
#############################
n_avg = 20
f_min = freq_track_obj.f_res - 2 * freq_track_obj.f_det
f_max = freq_track_obj.f_res + 2 * freq_track_obj.f_det
d_f = 2 * u.kHz
f_vec = np.arange(f_min, f_max, d_f)
oscillation = 1

# The QUA program
with program() as prog:
    freq_track_obj.qua_declaration()
    freq_track_obj.freq_domain_ramsey_full_sweep(n_avg, f_vec, oscillation)

    with stream_processing():
        freq_track_obj.state_estimation_st[0].buffer(len(f_vec)).average().save("Pe_fd")

# Execute the program
job = qm.execute(prog)
# Wait until processing is done before fetching results
job.result_handles.wait_for_all_values()
# Plot raw data + fit
plt.figure("Pe_fd")
freq_track_obj.freq_domain_ramsey_full_sweep_analysis(job.result_handles, "Pe_fd")

#########################
#  Real-time correction #
#########################
n_repetitions = 10000
tau_vec = np.arange(4, 50_000, 200)
with program() as prog:

    i = declare(int)
    i_st = declare_stream()
    with for_(i, 0, i < n_repetitions, i + 1):

        freq_track_obj.time_domain_ramsey_full_sweep(n_avg, freq_track_obj.f_det, tau_vec, False)
        freq_track_obj.two_points_ramsey()
        freq_track_obj.time_domain_ramsey_full_sweep(n_avg, freq_track_obj.f_det, tau_vec, True)
        save(i, i_st)
    with stream_processing():
        freq_track_obj.state_estimation_st[0].buffer(n_avg, len(tau_vec)).average().save("Pe_td_ref" )
        freq_track_obj.state_estimation_st[1].buffer(n_avg, len(tau_vec)).average().save("Pe_td_corr")
        i_st.save("iteration")

# Execute the program
job = qm.execute(prog)
# Handle results
results = fetching_tool(job, ["Pe_td_ref", "Pe_td_corr", "iteration"], mode="live")

# Starting time
t0 = time.time()

hours = 2
t_ = t0
cond = (t_ - t0) / 3600 < hours
fig, (ax1, ax2) = plt.subplots(1, 2)
Pe_td_ref = []
Pe_td_corr = []
t = []

while cond:
    # Fetch results
    Pe_td_ref_, Pe_td_corr_, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_repetitions, start_time=t0)
    # Get current time
    t_ = time.time()
    # Update while loop condition
    cond = (t_ - t0) / 3600 < hours
    # Update time vector and results
    t.append((t_ - t0) / 3600)
    Pe_td_ref.append(Pe_td_ref_)
    Pe_td_corr.append(Pe_td_corr_)
    # Plot results
    ax1.pcolormesh(freq_track_obj.tau_vec, t, Pe_td_ref)
    ax1.title.set_text("TD Ramsey feedback off")
    ax1.set_xlabel("tau [ns]")
    ax1.set_ylabel("time [hours]")
    ax2.pcolormesh(freq_track_obj.tau_vec, t, Pe_td_corr)
    ax2.title.set_text("TD Ramsey feedback on")
    ax2.set_xlabel("tau [ns]")
    ax2.set_ylabel("time [hours]")
    plt.pause(10)


##################################################
# verify correction from 2point WRT ramsey in TD #
##################################################
# freq_track_obj.fres = freq_track_obj.fres - 0
# points = 50
#
# with program() as prog:
#
#     freq_track_obj.qua_declarations()
#     i = declare(int)
#     with for_(i, 0, i < points, i+1):
#         freq_track_obj.time_domain_ramesy_full_sweep(n_avg, freq_track_obj.f_ref, 4, 50000, 200, 'td')
#         freq_track_obj.two_points_ramsey()
#
#     with stream_processing():
#         freq_track_obj.state_estimation_st[0].buffer(n_avg, len(freq_track_obj.tau_vec)).map(FUNCTIONS.average()).save_all('td')
#         freq_track_obj.corr_st.save_all('2point')
#
# job = qm.execute(prog)
# job.result_handles.wait_for_all_values()
# Pe = job.result_handles.get('td').fetch_all()['value']
# td_shifts = []; two_point_sifts = []
# for i in range(points):
#     t = np.array(freq_track_obj.tau_vec)*4
#     out = qubit_frequency_tracking._fit_ramsey(freq_track_obj, t, Pe[i])  # in [ns]
#     td_shifts.append(out['f'] * 1e9 - freq_track_obj.f_ref)
#     print(f"td shift: {out['f'] * 1e9 - freq_track_obj.f_ref}")
#     two_point_sifts.append(job.result_handles.get('2point').fetch_all()['value'][i])
#     print(job.result_handles.get('2point').fetch_all()['value'][i])
#
# plt.figure('td ramsey corr VS two point ramsey corr'); plt.plot(td_shifts, two_point_sifts, '.')
# plt.figure('td ramsey & two-point ramsey vs time'); plt.plot(td_shifts); plt.plot(two_point_sifts)


###################
# goal experiment #
###################

