"""
This file contains useful QUA macros meant to simplify and ease QUA programs.
All the macros below have been written and tested with the basic configuration. If you modify this configuration
(elements, operations, integration weights...) these macros will need to be modified accordingly.
"""

from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from qualang_tools.loops import from_array

##############
# QUA macros #
##############

# Single shot readout macro
def readout_macro(threshold=None, state=None, I=None, Q=None):
    """
    A macro for performing the readout, with the ability to perform state discrimination.
    If `threshold` is given, the information in the `I` quadrature will be compared against the threshold and `state`
    would be `True` if `I > threshold`.
    Note that it is assumed that the results are rotated such that all the information is in the `I` quadrature.

    :param threshold: Optional. The threshold to compare `I` against
    :param state: A QUA variable for the state information, only used when a threshold is given.
        Should be of type `bool`. If not given, a new variable will be created
    :param I: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :param Q: A QUA variable for the information in the `Q` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :return: Three QUA variables populated with the results of the readout: (`state`, `I`, `Q`)
    """
    if I is None:
        I = declare(fixed)
    if Q is None:
        Q = declare(fixed)
    if threshold is not None and state is None:
        state = declare(bool)
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
    )
    if threshold is not None:
        assign(state, I > threshold)
    return state, I, Q


# Macro for resetting the qubit state
def reset_qubit(method, **kwargs):
    """
    Macro to reset the qubit state.

    If method is 'cooldown', then the variable cooldown_time (in clock cycles) must be provided as a python integer > 4.

    **Example**: reset_qubit('cooldown', cooldown_time=500)

    If method is 'active', then 3 parameters are available as listed below.

    **Example**: reset_qubit('active', threshold=-0.003, max_tries=3)

    :param method: Method the reset the qubit state. Can be either 'cooldown' or 'active'.
    :type method: str
    :key cooldown_time: qubit relaxation time in clock cycle, needed if method is 'cooldown'. Must be an integer > 4.
    :key threshold: threshold to discriminate between the ground and excited state, needed if method is 'active'.
    :key max_tries: python integer for the maximum number of tries used to perform active reset,
        needed if method is 'active'. Must be an integer > 0 and default value is 1.
    :key Ig: A QUA variable for the information in the `I` quadrature used for active reset. If not given, a new
        variable will be created. Must be of type `Fixed`.
    :return:
    """
    if method == "cooldown":
        # Check cooldown_time
        cooldown_time = kwargs.get("cooldown_time", None)
        if (cooldown_time is None) or (cooldown_time < 4):
            raise Exception("'cooldown_time' must be an integer > 4 clock cycles")
        # Reset qubit state
        wait(cooldown_time, "qubit")
    elif method == "active":
        # Check threshold
        threshold = kwargs.get("threshold", None)
        if threshold is None:
            raise Exception("'threshold' must be specified for active reset.")
        # Check max_tries
        max_tries = kwargs.get("max_tries", 1)
        if (max_tries is None) or (not float(max_tries).is_integer()) or (max_tries < 1):
            raise Exception("'max_tries' must be an integer > 0.")
        # Check Ig
        Ig = kwargs.get("Ig", None)
        # Reset qubit state
        return active_reset(threshold, max_tries=max_tries, Ig=Ig)


# Macro for performing active reset until successful for a given number of tries.
def active_reset(threshold, max_tries=1, Ig=None):
    """Macro for performing active reset until succesfull for a gicen number of tries.

    :param threshold: threshold for the 'I' quadrature discriminating between ground and excited state.
    :param max_tries: python integer for the maximum number of tries used to perform active reset. Must >= 1.
    :param Ig: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :return: A QUA variable for the information in the `I` quadrature and the number of tries after success.
    """
    if Ig is None:
        Ig = declare(fixed)
    if (max_tries < 1) or (not float(max_tries).is_integer()):
        raise Exception("max_count must be an integer >= 1.")
    # Initialize Ig to be > threshold
    assign(Ig, threshold + 2**-28)
    # Number of tries for active reset
    counter = declare(int)
    # Reset the number of tries
    assign(counter, 0)

    # Perform active feedback
    align("qubit", "resonator")
    # Use a while loop and counter for other protocols and tests
    with while_((Ig > threshold) & (counter < max_tries)):
        # Measure the resonator
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", Ig),
        )
        # Play a pi pulse to get back to the ground state
        play("pi", "qubit", condition=(Ig > threshold))
        # Increment the number of tries
        assign(counter, counter + 1)
    return Ig, counter


# Frequency tracking class
class qubit_frequency_tracking:
    def __init__(self, qubit, rr, f_res, ge_threshold):

        self.qubit = qubit
        self.rr = rr
        self.f_res = f_res
        self.ge_threshold = ge_threshold
        self.t2 = None
        self.tau0 = None
        self.phase = None
        self.tau_vec = None
        self.f_det = None
        self.f_vec = None
        self.delta = None
        self.frequency_sweep_amp = None

        self.I = declare(fixed)
        self.Q = declare(fixed)
        self.state_estimation = declare(fixed)
        self.state_estimation_st = [declare_stream() for i in range(10)]  # TODO Why 10 and not 2?
        self.state_estimation_st_idx = 0

        self.res = declare(bool)

        self.n = declare(int)
        self.tau = declare(int)

        self.m = declare(int)
        self.f = declare(int)

        self.p = declare(int)
        self.if_total = declare(int, value=0)
        self.se_vec = declare(fixed, size=3)
        self.idx = declare(int)
        self.fres_corr = declare(int, value=int(self.f_res + 0.5)) # TODO why +0.5?
        self.fres_corr_st = declare_stream()
        self.corr = declare(int, value=0)
        self.corr_st = declare_stream()

    def _fit_ramsey(self, x, y):

        w = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(x))
        new_w = w[1 : len(freqs // 2)]
        new_f = freqs[1 : len(freqs // 2)]

        ind = new_f > 0
        new_f = new_f[ind]
        new_w = new_w[ind]

        yy = np.abs(new_w)
        first_read_data_ind = np.where(yy[1:] - yy[:-1] > 0)[0][0]  # away from the DC peak

        new_f = new_f[first_read_data_ind:]
        new_w = new_w[first_read_data_ind:]

        out_freq = new_f[np.argmax(np.abs(new_w))]
        new_w_arg = new_w[np.argmax(np.abs(new_w))]

        omega = out_freq * 2 * np.pi / (x[1] - x[0])  # get gauss for frequency #here

        cycle = int(np.ceil(1 / out_freq))
        peaks = np.array([np.std(y[i * cycle : (i + 1) * cycle]) for i in range(int(len(y) / cycle))]) * np.sqrt(2) * 2

        initial_offset = np.mean(y[:cycle])
        cycles_wait = np.where(peaks > peaks[0] * 0.37)[0][-1]

        post_decay_mean = np.mean(y[-cycle:])

        decay_gauss = (
            np.log(peaks[0] / peaks[cycles_wait]) / (cycles_wait * cycle) / (x[1] - x[0])
        )  # get gauss for decay #here

        fit_type = lambda x, a: post_decay_mean * a[4] * (1 - np.exp(-x * decay_gauss * a[1])) + peaks[0] / 2 * a[2] * (
            np.exp(-x * decay_gauss * a[1])
            * (a[5] * initial_offset / peaks[0] * 2 + np.cos(2 * np.pi * a[0] * omega / (2 * np.pi) * x + a[3]))
        )  # here problem, removed the 1+

        def curve_fit3(f, x, y, a0):
            def opt(x, y, a):
                return np.sum(np.abs(f(x, a) - y) ** 2)

            out = optimize.minimize(lambda a: opt(x, y, a), a0)
            return out["x"]

        angle0 = np.angle(new_w_arg) - omega * x[0]

        popt = curve_fit3(
            fit_type,
            x,
            y,
            [1, 1, 1, angle0, 1, 1, 1],
        )

        print(
            f"f = {popt[0] * omega / (2 * np.pi)}, phase = {popt[3] % (2 * np.pi)}, tau = {1 / (decay_gauss * popt[1])}, amp = {peaks[0] * popt[2]}, uncertainty population = {post_decay_mean * popt[4]},initial offset = {popt[5] * initial_offset}"
        )
        out = {
            "fit_func": lambda x: fit_type(x, popt),
            "f": popt[0] * omega / (2 * np.pi),
            "phase": popt[3] % (2 * np.pi),
            "tau": 1 / (decay_gauss * popt[1]),
            "amp": peaks[0] * popt[2],
            "uncertainty_population": post_decay_mean * popt[4],
            "initial_offset": popt[5] * initial_offset,
        }

        plt.plot(x, fit_type(x, [1, 1, 1, angle0, 1, 1, 1]), "--r", linewidth=1)
        return out

    def time_domain_ramsey_full_sweep(self, n_avg, f_det, tau_vec, correct=False):
        """QUA program to perform a time-domain Ramsey sequence with `n_avg` averages and scanning the idle time over `tau_vec`.

        :param int n_avg: python integer for the number of averaging loops
        :param int f_det: python integer for the detuning to apply in Hz
        :param tau_vec: numpy array of integers for the idle times to be scanned in clock cycles (4ns)
        :param bool correct: boolean for
        :return: None
        """
        self.f_det = f_det
        self.tau_vec = tau_vec

        if correct:
            update_frequency(self.qubit, self.f_res + self.f_det - self.corr)
        else:
            update_frequency(self.qubit, self.f_res + self.f_det)

        with for_(self.n, 0, self.n < n_avg, self.n + 1):
            with for_(*from_array(self.tau, tau_vec)):
                # Qubit initialization
                reset_qubit("cooldown", cooldown_time=1000)
                # Ramsey sequence (time-domain)
                play("x90", self.qubit)
                wait(self.tau, self.qubit)
                play("x90", self.qubit)

                align(self.qubit, self.rr)
                # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
                # the QUA variable "self.res". True for the qubit in the excited. ##################################
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", self.I),
                )
                assign(self.res, self.I > self.ge_threshold)
                ####################################################################################################
                # Convert bool to fixed to perform the average
                assign(self.state_estimation, Cast.to_fixed(self.res))
                save(
                    self.state_estimation,
                    self.state_estimation_st[self.state_estimation_st_idx],
                )

        self.state_estimation_st_idx = self.state_estimation_st_idx + 1

    def time_domain_ramsey_full_sweep_analysis(self, result_handles, stream_name):
        # Get the average excited population
        Pe = result_handles.get(stream_name).fetch_all()
        # Get the idle time vector in ns
        t = np.array(self.tau_vec) * 4
        # Plot raw data
        plt.plot(t, Pe, ".", label="Experimental data")
        # Fit data
        out = qubit_frequency_tracking._fit_ramsey(self, t, Pe)  # in [ns]
        # Plot fit
        plt.plot(t, out["fit_func"](t), "m", label="Fit")
        plt.xlabel("time[ns]")
        plt.ylabel("P(|e>)")
        # New intermediate frequency: f_res - (delta - f_det)
        self.f_res = self.f_res - (out["f"] * 1e9 - self.f_det)
        print(f"shifting by {out['f'] * 1e9 - self.f_det:.0f} Hz, and now f_res = {self.f_res} Hz")

        self.t2 = out["tau"]
        self.phase = out["phase"]
        self.tau0 = int(1 / self.f_det / 4e-9)
        plt.plot(
            self.tau0 * 4,
            out["fit_func"](self.tau0 * 4),
            "r*",
            label="Ideal first peak location",
        )
        plt.legend()

    def freq_domain_ramsey_full_sweep(self, n_avg, f_vec, oscillation_number=1, correct=False):
        """QUA program to perform a frequency-domain Ramsey sequence with `n_avg` averages and scanning the frequency over `f_vec`.

        :param int n_avg: python integer for the number of averaging loops
        :param f_vec: numpy array of integers for the qubit detuning to be scanned in Hz
        :param oscillation_number: number of oscillations to capture used to define the idle time as .
        :param correct:
        :return:
        """
        self.f_vec = f_vec

        self.tau0 = oscillation_number * int(1 / (2 * max(f_vec)) / 4e-9)
        self.delta = 1 / (self.tau0 * 4e-9) / 4  # the last 4 is for 1/4 of a cycle

        with for_(self.n, 0, self.n < n_avg, self.n + 1):
            with for_(*from_array(self.f, f_vec)):
                # Qubit initialization
                # Note: if you are using active reset, you might want to do it with the new corrected frequency
                reset_qubit("cooldown", cooldown_time=1000)
                ####################################################################################################
                # Update the frequency
                if correct:
                    update_frequency(self.qubit, self.f + self.corr)
                else:
                    update_frequency(self.qubit, self.f)
                # Ramsey sequence
                play("x90", self.qubit)
                wait(self.tau0, self.qubit)
                play("x90", self.qubit)

                align(self.qubit, self.rr)
                # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
                # the QUA variable "self.res". True for the qubit in the excited. ##################################
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", self.I),
                )
                assign(self.res, self.I > self.ge_threshold)
                ####################################################################################################
                # Convert bool to fixed to perform the average
                assign(self.state_estimation, Cast.to_fixed(self.res))
                save(self.state_estimation, self.state_estimation_st[self.state_estimation_st_idx])

        self.state_estimation_st_idx = self.state_estimation_st_idx + 1

    def freq_domain_ramsey_full_sweep_analysis(self, result_handles, stream_name):
        # Get the average excited population
        Pe = result_handles.get(stream_name).fetch_all()
        # Get the qubit detuning vector
        f = np.array(self.f_vec)
        # Plot raw data
        plt.plot(f - self.f_res, Pe, '.', label='Experimental data')
        # Fit data
        out = qubit_frequency_tracking._fit_ramsey(self, f - self.f_res, Pe)  # in Hz
        self.frequency_sweep_amp = out["amp"]
        # Plot fit
        plt.plot(f - self.f_res, out["fit_func"](f - self.f_res), "m", label='fit')
        # Plot specific points at half the central fringe
        plt.plot(
            [-self.delta, self.delta],
            out["fit_func"](np.array([-self.delta, self.delta])),
            "r*",
        )
        plt.xlabel("Detuning from resonance [Hz]")
        plt.ylabel("P(|e>)")
        plt.legend()

    def two_points_ramsey(self):

        c = int(1 / (2 * np.pi * self.tau0 * 4e-9 * self.frequency_sweep_amp))
        print(f"c = {c}")
        assign(self.se_vec[0], 0)
        assign(self.se_vec[1], 0)
        # TODO what does this 2**15 mean? biggest?
        with for_(self.p, 0, self.p < 32768, self.p + 1):
            # Go to the left side of the central fringe
            assign(self.f, self.f_res - self.delta)
            # Alternate between left and right sides
            with for_(self.idx, 0, self.idx < 2, self.idx + 1):
                # Qubit initialization
                # Note: if you are using active reset, you might want to do it with the new corrected frequency
                reset_qubit("cooldown", cooldown_time=1000)
                ####################################################################################################
                # Set qubit frequency
                update_frequency(self.qubit, self.f)
                # Ramsey sequence
                play("pi2", self.qubit)
                wait(self.tau0, self.qubit)
                play("pi2", self.qubit)

                align(self.qubit, self.rr)
                # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
                # the QUA variable "self.res". True for the qubit in the excited. ##################################
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", self.I),
                )
                assign(self.res, self.I > self.ge_threshold)
                ####################################################################################################
                # Sum the results and divide by the number of iterations to get the average on the fly
                assign(self.se_vec[self.idx], self.se_vec[self.idx] + (Cast.to_fixed(self.res) >> 15))
                # Go to the right side of the central fringe
                assign(self.f, self.f + 2 * self.delta)

        # Derive the frequency shift
        assign(self.corr, Cast.mul_int_by_fixed(c, (self.se_vec[0] - self.se_vec[1])))
        # To keep track of the qubit frequency over time
        assign(self.fres_corr, self.fres_corr - self.corr)

        save(self.fres_corr, self.fres_corr_st)
        save(self.corr, self.corr_st)


# class qubit_frequency_tracking:
#     def __init__(self, qubit, rr, f_res):
#
#         self.qubit = qubit
#         self.rr = rr
#         self.fres = f_res
#         self.t2 = None
#         self.tau0 = None
#         self.phase = None
#         self.tau_vec = None
#         self.f_det = None
#         self.fvec = None
#         self.delta = None
#         self.frequency_sweep_amp = None
#
#     def qua_declarations(self):
#
#         self.I = declare(fixed)
#         self.Q = declare(fixed)
#         self.state_estimation = declare(fixed)
#         self.state_estimation_st = [declare_stream() for i in range(10)]
#         self.state_estimation_st_idx = 0
#
#         self.res = declare(bool)
#
#         self.n = declare(int)
#         self.tau = declare(int)
#
#         self.m = declare(int)
#         self.f = declare(int)
#
#         self.p = declare(int)
#         self.if_total = declare(int, value=0)
#         self.se_vec = declare(fixed, size=3)
#         self.idx = declare(int)
#         self.fres_corr = declare(int, value=int(self.fres + 0.5))
#         self.fres_corr_st = declare_stream()
#         self.corr = declare(int, value=0)
#         self.corr_st = declare_stream()
#
#     def _fit_ramsey(self, x, y):
#
#         w = np.fft.fft(y)
#         freqs = np.fft.fftfreq(len(x))
#         new_w = w[1 : len(freqs // 2)]
#         new_f = freqs[1 : len(freqs // 2)]
#
#         ind = new_f > 0
#         new_f = new_f[ind]
#         new_w = new_w[ind]
#
#         yy = np.abs(new_w)
#         first_read_data_ind = np.where(yy[1:] - yy[:-1] > 0)[0][0]  # away from the DC peak
#
#         new_f = new_f[first_read_data_ind:]
#         new_w = new_w[first_read_data_ind:]
#
#         out_freq = new_f[np.argmax(np.abs(new_w))]
#         new_w_arg = new_w[np.argmax(np.abs(new_w))]
#
#         omega = out_freq * 2 * np.pi / (x[1] - x[0])  # get gauss for frequency #here
#
#         cycle = int(np.ceil(1 / out_freq))
#         peaks = np.array([np.std(y[i * cycle : (i + 1) * cycle]) for i in range(int(len(y) / cycle))]) * np.sqrt(2) * 2
#
#         initial_offset = np.mean(y[:cycle])
#         cycles_wait = np.where(peaks > peaks[0] * 0.37)[0][-1]
#
#         post_decay_mean = np.mean(y[-cycle:])
#
#         decay_gauss = (
#             np.log(peaks[0] / peaks[cycles_wait]) / (cycles_wait * cycle) / (x[1] - x[0])
#         )  # get gauss for decay #here
#
#         fit_type = lambda x, a: post_decay_mean * a[4] * (1 - np.exp(-x * decay_gauss * a[1])) + peaks[0] / 2 * a[2] * (
#             np.exp(-x * decay_gauss * a[1])
#             * (a[5] * initial_offset / peaks[0] * 2 + np.cos(2 * np.pi * a[0] * omega / (2 * np.pi) * x + a[3]))
#         )  # here problem, removed the 1+
#
#         def curve_fit3(f, x, y, a0):
#             def opt(x, y, a):
#                 return np.sum(np.abs(f(x, a) - y) ** 2)
#
#             out = optimize.minimize(lambda a: opt(x, y, a), a0)
#             return out["x"]
#
#         angle0 = np.angle(new_w_arg) - omega * x[0]
#
#         popt = curve_fit3(
#             fit_type,
#             x,
#             y,
#             [1, 1, 1, angle0, 1, 1, 1],
#         )
#
#         print(
#             f"f = {popt[0] * omega / (2 * np.pi)}, phase = {popt[3] % (2 * np.pi)}, tau = {1 / (decay_gauss * popt[1])}, amp = {peaks[0] * popt[2]}, uncertainty population = {post_decay_mean * popt[4]},initial offset = {popt[5] * initial_offset}"
#         )
#         out = {
#             "fit_func": lambda x: fit_type(x, popt),
#             "f": popt[0] * omega / (2 * np.pi),
#             "phase": popt[3] % (2 * np.pi),
#             "tau": 1 / (decay_gauss * popt[1]),
#             "amp": peaks[0] * popt[2],
#             "uncertainty_population": post_decay_mean * popt[4],
#             "initial_offset": popt[5] * initial_offset,
#         }
#
#         plt.plot(x, fit_type(x, [1, 1, 1, angle0, 1, 1, 1]), "--r", linewidth=1)
#         return out
#
#     def time_domain_ramsey_full_sweep(self, n_avg, f_ref, tau_min, tau_max, dtau, stream_name, correct=False):
#
#         self.f_det = f_ref
#         self.tau_vec = np.arange(tau_min, tau_max, dtau).astype(int).tolist()
#
#         if correct:
#             update_frequency(self.qubit, self.fres + self.f_det - self.corr)
#         else:
#             update_frequency(self.qubit, self.fres + self.f_det)
#         with for_(self.n, 0, self.n < n_avg, self.n + 1):
#             with for_(self.tau, tau_min, self.tau < tau_max, self.tau + dtau):
#                 # Should be replaced by the initialization procedure of the qubit to the ground state #
#                 wait(10000, "qubit")
#                 #######################################################################################
#
#                 play("x90", self.qubit)
#                 wait(self.tau, self.qubit)
#                 play("x90", self.qubit)
#
#                 align(self.qubit, self.rr)
#
#                 # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
#                 # the QUA variable "self.res". True for the qubit in the excited. ##################################
#                 measure(
#                     "readout",
#                     "resonator",
#                     None,
#                     dual_demod.full("cos", "out1", "sin", "out2", self.I),
#                 )
#                 assign(self.res, self.I > 0)
#                 ####################################################################################################
#
#                 assign(self.state_estimation, Cast.to_fixed(self.res))
#                 save(
#                     self.state_estimation,
#                     self.state_estimation_st[self.state_estimation_st_idx],
#                 )
#
#         self.state_estimation_st_idx = self.state_estimation_st_idx + 1
#
#     def time_domain_ramsey_full_sweep_analysis(self, result_handles, stream_name):
#
#         Pe = result_handles.get(stream_name).fetch_all()
#         t = np.array(self.tau_vec) * 4
#         plt.plot(t, Pe)
#         out = qubit_frequency_tracking._fit_ramsey(self, t, Pe)  # in [ns]
#         plt.plot(t, out["fit_func"](t), "m")
#         plt.xlabel("time[ns]")
#         plt.ylabel("P(|e>)")
#
#         self.fres = self.fres - (out["f"] * 1e9 - self.f_det)  # Intermediate frequency [Hz]
#         print(f"shifting by {out['f'] * 1e9 - self.f_det}, and now f_res = {self.fres}")
#
#         self.t2 = out["tau"]
#         self.phase = out["phase"]
#         self.tau0 = int(1 / self.f_det / 4e-9)
#         plt.plot(
#             self.tau0 * 4,
#             out["fit_func"](self.tau0 * 4),
#             "r*",
#             label="ideal first peak location",
#         )
#         plt.legend()
#
#     def freq_domain_ramsey_full_sweep(self, n_avg, fmin, fmax, df, stream_name, oscillation_number=1, correct=False):
#         self.tau0 = oscillation_number * int(1 / self.f_det / 4e-9)
#         self.delta = 1 / (self.tau0 * 4e-9) / 4  # the last 4 is for 1/4 of a cycle
#         self.fvec = np.arange(fmin, fmax, df).astype(int).tolist()
#
#         with for_(self.m, 0, self.m < n_avg, self.m + 1):
#             with for_(self.f, fmin, self.f < fmax, self.f + df):
#
#                 # Should be replaced by the initialization procedure of the qubit to the ground state #
#                 wait(10000, "qubit")
#                 # Note: if you are using active reset, you might want to do it with the new corrected
#                 # frequency
#                 #######################################################################################
#
#                 if correct:
#                     update_frequency(self.qubit, self.f + self.corr)
#                 else:
#                     update_frequency(self.qubit, self.f + self.corr)
#                 play("x90", self.qubit)
#                 wait(self.tau0, self.qubit)
#                 play("x90", self.qubit)
#
#                 align(self.qubit, self.rr)
#
#                 # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
#                 # the QUA variable "self.res". True for the qubit in the excited. ##################################
#                 measure(
#                     "readout",
#                     "resonator",
#                     None,
#                     dual_demod.full("cos", "out1", "sin", "out2", self.I),
#                 )
#                 assign(self.res, self.I > 0)
#                 ####################################################################################################
#
#                 assign(self.state_estimation, Cast.to_fixed(self.res))
#                 save(
#                     self.state_estimation,
#                     self.state_estimation_st[self.state_estimation_st_idx],
#                 )
#
#         self.state_estimation_st_idx = self.state_estimation_st_idx + 1
#
#     def freq_domain_ramsey_full_sweep_analysis(self, result_handles, stream_name):
#         Pe = result_handles.get(stream_name).fetch_all()
#         f = np.array(self.fvec)
#         plt.plot(f - self.fres, Pe)
#         out = qubit_frequency_tracking._fit_ramsey(self, f - self.fres, Pe)  # in Hz
#         self.frequency_sweep_amp = out["amp"]
#         plt.plot(f - self.fres, out["fit_func"](f - self.fres), "m")
#         plt.plot(
#             [-self.delta, self.delta],
#             out["fit_func"](np.array([-self.delta, self.delta])),
#             "r*",
#         )
#         plt.xlabel("detuning from resonance[Hz]")
#         plt.ylabel("P(|e>)")
#
#     def two_points_ramsey(self):
#
#         c = int(1 / (2 * np.pi * self.tau0 * 4e-9 * self.frequency_sweep_amp))
#         print(f"c = {c}")
#         assign(self.se_vec[0], 0)
#         assign(self.se_vec[1], 0)
#
#         with for_(self.p, 0, self.p < 32768, self.p + 1):
#             assign(self.f, self.fres - self.delta)
#
#             with for_(self.idx, 0, self.idx < 2, self.idx + 1):
#                 # Should be replaced by the initialization procedure of the qubit to the ground state #
#                 wait(10000, "qubit")
#                 # Note: if you are using active reset, you might want to do it with the new corrected
#                 # frequency
#                 #######################################################################################
#
#                 update_frequency(self.qubit, self.f)
#                 play("pi2", self.qubit)
#                 wait(self.tau0, self.qubit)
#                 play("pi2", self.qubit)
#
#                 align(self.qubit, self.rr)
#
#                 # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
#                 # the QUA variable "self.res". True for the qubit in the excited. ##################################
#                 measure(
#                     "readout",
#                     "resonator",
#                     None,
#                     dual_demod.full("cos", "out1", "sin", "out2", self.I),
#                 )
#                 assign(self.res, self.I > 0)
#                 ####################################################################################################
#
#                 assign(
#                     self.se_vec[self.idx],
#                     self.se_vec[self.idx] + (Cast.to_fixed(self.res) >> 15),
#                 )
#                 assign(self.f, self.f + 2 * self.delta)
#
#         assign(self.corr, Cast.mul_int_by_fixed(c, (self.se_vec[0] - self.se_vec[1])))
#         assign(self.fres_corr, self.fres_corr - self.corr)
#         # update_frequency(self.qubit, self.fres_corr)
#
#         save(self.fres_corr, self.fres_corr_st)
#         save(self.corr, self.corr_st)
