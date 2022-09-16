from qcodes.utils.validators import Arrays
from opx_driver import *
from qm.qua import *


# noinspection PyAbstractClass
class OPXSpectrumScan(OPX):
    def __init__(self, config: Dict, name: str = "OPX", host=None, port=None, **kwargs):
        super().__init__(config, name, host=host, port=port, **kwargs)
        self.add_parameter(
            "f_start",
            initial_value=30e6,
            unit="Hz",
            label="f start",
            vals=Numbers(-400e6, 400e6),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "f_stop",
            initial_value=70e6,
            unit="Hz",
            label="f stop",
            vals=Numbers(-400e6, 400e6),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "n_points",
            initial_value=100,
            unit="",
            vals=Numbers(
                1,
            ),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "n_avg",
            unit="",
            initial_value=1,
            vals=Numbers(
                1,
            ),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "freq_axis",
            unit="Hz",
            label="Freq Axis",
            parameter_class=GeneratedSetPoints,
            startparam=self.f_start,
            stopparam=self.f_stop,
            numpointsparam=self.n_points,
            vals=Arrays(shape=(self.n_points.get_latest,)),
        )
        self.add_parameter(
            "amp",
            unit="",
            initial_value=1,
            vals=Numbers(-2, 2 - 2**-28),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "readout_pulse_length",
            unit="ns",
            vals=Numbers(16, 1e7),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "readout_element",
            unit="",
            initial_value="resonator",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "readout_operation",
            unit="",
            initial_value="readout",
            get_cmd=None,
            set_cmd=None,
        )

    def get_prog(self):
        df = round((self.f_stop() - self.f_start()) / self.n_points())
        with program() as prog:
            n = declare(int)
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            n_st = declare_stream()
            I_st = declare_stream()
            Q_st = declare_stream()
            with for_(n, 0, n < self.n_avg(), n + 1):
                with for_(f, self.f_start(), f < self.f_stop(), f + df):
                    update_frequency(self.readout_element(), f)
                    measure(
                        self.readout_operation() * amp(self.amp()),
                        self.readout_element(),
                        None,
                        demod.full("cos", I, "out1"),
                        demod.full("sin", Q, "out1"),
                    )
                    save(I, I_st)
                    save(Q, Q_st)
                save(n, n_st)
            with stream_processing():
                I_st.buffer(self.n_points()).average().save("I")
                Q_st.buffer(self.n_points()).average().save("Q")
                n_st.save("iteration")

        return prog

    def run_exp(self):
        self.execute_prog(self.get_prog())

    # def get_res(self):
    #     if self.result_handles is None:
    #         n = self.n_points()
    #         return {"I": (0,) * n, "Q": (0,) * n, "R": (0,) * n, "Phi": (0,) * n}
    #     else:
    #         print("get res")
    #         self.result_handles.wait_for_all_values()
    #         I = self.result_handles.get("I").fetch_all() / self.readout_pulse_length() * 2**12 * 2
    #         Q = self.result_handles.get("Q").fetch_all() / self.readout_pulse_length() * 2**12 * 2
    #         R = np.sqrt(I**2 + Q**2)
    #         phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
    #         return {"I": I, "Q": Q, "R": R, "Phi": phase}

    def get_res(self):

        if self.result_handles is None:
            n = self.n_points()
            return {"I": [[0] * n] * n, "Q": [[0] * n] * n, "R": [[0] * n] * n, "Phi": [[0] * n] * n}
        else:
            I = 0
            Q = 0
            R = 0
            phase = 0
            if self.live_plot:
                if self.live_in_python:
                    results = fetching_tool(self.job, ["I", "Q", "iteration"], mode="live")
                    fig = plt.figure()
                    interrupt_on_close(fig, self.job)
                    while results.is_processing():
                        I, Q, iteration = results.fetch_all()
                        progress_counter(iteration, self.n_avg(), start_time=results.start_time)

                        I = I / self.config["pulses"]["readout_pulse"]["length"] * 2**12
                        Q = Q / self.config["pulses"]["readout_pulse"]["length"] * 2**12
                        R = np.sqrt(I**2 + Q**2)
                        phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
                        plt.subplot(221)
                        plt.cla()
                        plt.title("I [V]")
                        plt.plot(self.freq_axis(), I)
                        plt.xlabel("Frequency [Hz]")
                        plt.subplot(222)
                        plt.cla()
                        plt.title("Q [V]")
                        plt.plot(self.freq_axis(), Q)
                        plt.xlabel("Frequency [Hz]")
                        plt.subplot(223)
                        plt.cla()
                        plt.title("R [V]")
                        plt.plot(self.freq_axis(), R)
                        plt.xlabel("Frequency [Hz]")
                        plt.subplot(224)
                        plt.cla()
                        plt.title("phase [deg]")
                        plt.plot(self.freq_axis(), phase)
                        plt.xlabel("Frequency [Hz]")
                        plt.tight_layout()
                        plt.pause(0.1)

                else:
                    self.result_handles.get("I").wait_for_values(1)
                    self.result_handles.get("Q").wait_for_values(1)
                    self.result_handles.get("iteration").wait_for_values(1)

                    I = (
                        self.result_handles.get("I").fetch_all()
                        / self.config["pulses"]["readout_pulse"]["length"]
                        * 2**12
                    )
                    Q = (
                        self.result_handles.get("Q").fetch_all()
                        / self.config["pulses"]["readout_pulse"]["length"]
                        * 2**12
                    )
                    R = np.sqrt(I**2 + Q**2)
                    phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
                    iteration = self.result_handles.get("iteration").fetch_all()
                    progress_counter(iteration, self.n_avg())

            else:
                self.result_handles.wait_for_all_values()
                I = (
                    self.result_handles.get("I").fetch_all()
                    / self.config["pulses"]["readout_pulse"]["length"]
                    * 2**12
                )
                Q = (
                    self.result_handles.get("Q").fetch_all()
                    / self.config["pulses"]["readout_pulse"]["length"]
                    * 2**12
                )
                R = np.sqrt(I**2 + Q**2)
                phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
        return {"I": I, "Q": Q, "R": R, "Phi": phase}
