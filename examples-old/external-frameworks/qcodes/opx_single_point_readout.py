from opx_driver import *
from qm.qua import *


# noinspection PyAbstractClass
class OPXSinglePointReadout(OPX):
    def __init__(self, config: Dict, name: str = "OPX", host=None, port=None, **kwargs):
        super().__init__(config, name, host=host, port=port, **kwargs)
        self.counter = 0
        self.add_parameter(
            "f",
            initial_value=300e6,
            unit="Hz",
            label="freq",
            vals=Numbers(-400e6, 400e6),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "t_meas",
            unit="s",
            initial_value=0.013,
            vals=Numbers(0, 1),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "n_avg",
            unit="",
            initial_value=1,
            vals=Numbers(1, 1e9),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "amp",
            unit="",
            initial_value=1,
            vals=Numbers(-2, 2 - 2**-16),
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

    def get_prog(self):
        # n_avg = round(self.t_meas() * 1e9 / self.readout_pulse_length())
        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            update_frequency("resonator", round(self.f()))
            with infinite_loop_():
                pause()
                with for_(n, 0, n < self.n_avg(), n + 1):
                    measure(
                        "cw_reflectometry" * amp(self.amp()),
                        "resonator",
                        None,
                        demod.full("cos", I, "out1"),
                        demod.full("sin", Q, "out1"),
                    )
                    save(I, I_st)
                    save(Q, Q_st)

            with stream_processing():
                I_st.buffer(self.n_avg()).map(FUNCTIONS.average()).save_all("I")
                Q_st.buffer(self.n_avg()).map(FUNCTIONS.average()).save_all("Q")

        return prog

    def run_exp(self):
        self.execute_prog(self.get_prog())
        self.counter = 0

    def resume(self):
        self.job.resume()
        self.counter += 1

    def get_res(self):
        if self.result_handles is None:
            return {"I": 0, "Q": 0, "R": 0, "Phi": 0}
        else:
            self.result_handles.wait_for_all_values()
            I = self.result_handles.get("I").fetch_all() / self.config["pulses"]["readout_pulse"][
                "length"] * 2 ** 12
            Q = self.result_handles.get("Q").fetch_all() / self.config["pulses"]["readout_pulse"][
                "length"] * 2 ** 12
            R = np.sqrt(I ** 2 + Q ** 2)
            phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
        return {"I": I, "Q": Q, "R": R, "Phi": phase}


