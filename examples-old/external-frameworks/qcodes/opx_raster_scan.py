from qcodes.utils.validators import Arrays
from opx_driver import *
from qm.qua import *


# noinspection PyAbstractClass
class OPXRasterScan(OPX):
    def __init__(self, config: Dict, name: str = "OPX", host=None, port=None, **kwargs):
        super().__init__(config, name, host=host, port=port, **kwargs)
        self.config = config
        self.counter = 0

        self.add_parameter(
            "Vx_min",
            unit="",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vx_max",
            unit="",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Nx",
            unit="",
            vals=Numbers(1, 1e9),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vy_min",
            unit="",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vy_max",
            unit="",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Ny",
            unit="",
            vals=Numbers(1, 1e9),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vx_axis",
            unit="V",
            label="Vx Axis",
            parameter_class=GeneratedSetPoints,
            startparam=self.Vx_min,
            stopparam=self.Vx_max,
            numpointsparam=self.Nx,
            vals=Arrays(shape=(self.Nx.get_latest,)),
        )
        self.add_parameter(
            "Vy_axis",
            unit="V",
            label="Vy Axis",
            parameter_class=GeneratedSetPoints,
            startparam=self.Vy_min,
            stopparam=self.Vy_max,
            numpointsparam=self.Ny,
            vals=Arrays(shape=(self.Ny.get_latest,)),
        )
        self.add_parameter(
            "n_avg",
            unit="",
            vals=Numbers(1, 1e9),
            get_cmd=None,
            set_cmd=None,
        )

    def get_prog(self):
        readout_element = "resonator"
        x_element = "G2"
        y_element = "G1"
        self.qm.set_output_dc_offset_by_element(y_element, "single", self.Vy_min())
        self.qm.set_output_dc_offset_by_element(x_element, "single", self.Vx_min())
        # n_avg = round(self.t_meas() * 1e9 / self.readout_pulse_length())
        dy = (self.Vy_max() - self.Vy_min()) / ((self.Ny() - 1) * self.config["waveforms"]["jump_wf"]["sample"])
        dx = (self.Vx_max() - self.Vx_min()) / ((self.Nx() - 1) * self.config["waveforms"]["jump_wf"]["sample"])
        print(f"dx = {dx}")
        print(f"dy = {dy}")
        with program() as prog:
            n = declare(int)
            x = declare(int)
            y = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            Vx = declare(fixed, value=self.Vx_min())
            Vy = declare(fixed, value=self.Vy_min())
            Vx_st = declare_stream()
            Vy_st = declare_stream()
            I_st = declare_stream()
            Q_st = declare_stream()
            # play("const", "G2")
            with for_(n, 0, n < self.n_avg(), n + 1):
                ramp_to_zero(x_element, duration=4)
                ramp_to_zero(y_element, duration=4)
                assign(Vx, self.Vx_min())
                assign(Vy, self.Vy_min())
                with for_(y, 0, y < self.Ny(), y + 1):
                    with if_(y > 0):
                        play("jump" * amp(dy), y_element)
                        assign(Vy, Vy + dy * self.config["waveforms"]["jump_wf"]["sample"])
                    ramp_to_zero(x_element, duration=4)
                    assign(Vx, self.Vx_min())
                    with for_(x, 0, x < self.Nx(), x + 1):
                        with if_(x > 0):
                            play("jump" * amp(dx), x_element)
                            assign(Vx, Vx + dx * self.config["waveforms"]["jump_wf"]["sample"])
                        align()
                        wait(100, readout_element)
                        measure(
                            "readout",
                            readout_element,
                            None,
                            demod.full("cos", I, "out1"),
                            demod.full("sin", Q, "out1"),
                        )
                        save(Vx, Vx_st)
                        save(Vy, Vy_st)
                        save(I, I_st)
                        save(Q, Q_st)
            with stream_processing():
                I_st.buffer(self.Nx()).buffer(self.Ny()).average().save("I")
                Q_st.buffer(self.Nx()).buffer(self.Ny()).average().save("Q")
                Vx_st.buffer(self.Nx()).buffer(self.Ny()).save("Vx")
                Vy_st.buffer(self.Nx()).buffer(self.Ny()).save("Vy")

        return prog

    def run_exp(self):
        self.execute_prog(self.get_prog())
        self.counter = 0

    def simulate_exp(self, duration):
        self.simulate_prog(self.get_prog(), duration=duration)
        self.counter = 0

    def resume(self):
        self.job.resume()
        self.counter += 1

    def get_res(self):
        if self.result_handles is None:
            nx = self.Nx()
            ny = self.Ny()
            return {
                "I": [[0] * nx] * ny,
                "Q": [[0] * nx] * ny,
                "R": [[0] * nx] * ny,
                "Phi": [[0] * nx] * ny,
                "Vx": [[0] * nx] * ny,
                "Vy": [[0] * nx] * ny,
            }
        else:
            self.result_handles.wait_for_all_values()
            I = self.result_handles.get("I").fetch_all() / self.config["pulses"]["readout_pulse"]["length"] * 2**12
            Q = self.result_handles.get("Q").fetch_all() / self.config["pulses"]["readout_pulse"]["length"] * 2**12
            R = np.sqrt(I**2 + Q**2)
            phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
            Vx = self.result_handles.get("Vx").fetch_all()
            Vy = self.result_handles.get("Vy").fetch_all()
            return {"I": I, "Q": Q, "R": R, "Phi": phase, "Vx": Vx, "Vy": Vy}
