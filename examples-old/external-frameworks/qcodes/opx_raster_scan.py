from qcodes.utils.validators import Arrays
from opx_driver import *
from qm.qua import *


# noinspection PyAbstractClass
class OPXRasterScan(OPX):
    def __init__(self, config: Dict, name: str = "OPX", host=None, port=None, **kwargs):
        super().__init__(config, name, host=host, port=port, **kwargs)
        self.config = config

        self.add_parameter(
            "wait_time",
            unit="ns",
            vals=Numbers(4, 1e7),
            get_cmd=None,
            set_cmd=None,
        )
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
        self.add_parameter(
            "readout_element",
            unit="",
            initial_value="resonator",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "x_element",
            unit="",
            initial_value="G2",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "y_element",
            unit="",
            initial_value="G1",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "gate_operation",
            unit="",
            initial_value="jump",
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
        self.qm.set_output_dc_offset_by_element(self.y_element(), "single", self.Vy_min())
        self.qm.set_output_dc_offset_by_element(self.x_element(), "single", self.Vx_min())
        pulse = self.config["elements"][self.x_element()]["operations"][self.gate_operation()]
        wf = self.config["pulses"][pulse]["waveforms"]["single"]
        dy = (self.Vy_max() - self.Vy_min()) / ((self.Ny() - 1) * self.config["waveforms"][wf].get("sample"))
        dx = (self.Vx_max() - self.Vx_min()) / ((self.Nx() - 1) * self.config["waveforms"][wf].get("sample"))
        print(
            f"X scan from {self.Vx_min() * 1000:.2f} mV "
            f"to {self.Vx_max() * 1000:.2f} mV "
            f"in {self.Nx()} steps "
            f"of {dx * self.config['waveforms'][wf].get('sample') * 1000:.2f} mV."
        )
        print(
            f"Y scan from {self.Vy_min() * 1000:.2f} mV "
            f"to {self.Vy_max() * 1000:.2f} mV "
            f"in {self.Ny()} steps "
            f"of {dy * self.config['waveforms'][wf].get('sample') * 1000:.2f} mV."
        )
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
            n_st = declare_stream()
            with for_(n, 0, n < self.n_avg(), n + 1):
                ramp_to_zero(self.x_element(), duration=4)
                ramp_to_zero(self.y_element(), duration=4)
                assign(Vx, self.Vx_min())
                assign(Vy, self.Vy_min())
                with for_(y, 0, y < self.Ny(), y + 1):
                    with if_(y > 0):
                        play(self.gate_operation() * amp(dy), self.y_element())
                        assign(Vy, Vy + dy * self.config["waveforms"][wf].get("sample"))
                    ramp_to_zero(self.x_element(), duration=4)
                    assign(Vx, self.Vx_min())
                    with for_(x, 0, x < self.Nx(), x + 1):
                        with if_(x > 0):
                            play(self.gate_operation() * amp(dx), self.x_element())
                            assign(Vx, Vx + dx * self.config["waveforms"][wf].get("sample"))
                        align()
                        wait(self.wait_time(), self.readout_element())
                        measure(
                            self.readout_operation(),
                            self.readout_element(),
                            None,
                            demod.full("cos", I, "out1"),
                            demod.full("sin", Q, "out1"),
                        )
                        save(Vx, Vx_st)
                        save(Vy, Vy_st)
                        save(I, I_st)
                        save(Q, Q_st)
                save(n, n_st)
            with stream_processing():
                I_st.buffer(self.Nx()).buffer(self.Ny()).average().save("I")
                Q_st.buffer(self.Nx()).buffer(self.Ny()).average().save("Q")
                # Vx_st.buffer(self.Nx()).buffer(self.Ny()).save("Vx")
                # Vy_st.buffer(self.Nx()).buffer(self.Ny()).save("Vy")
                n_st.save("iteration")

        return prog

    def run_exp(self):
        self.execute_prog(self.get_prog())

    def simulate_exp(self, get_results=False):
        if get_results:
            self.simulate_and_read(self.get_prog())
        else:
            self.simulate_prog(self.get_prog())

    def get_res(self):

        if self.result_handles is None:
            nx = self.Nx()
            ny = self.Ny()
            return {
                "I": [[0] * nx] * ny,
                "Q": [[0] * nx] * ny,
                "R": [[0] * nx] * ny,
                "Phi": [[0] * nx] * ny,
            }
        else:
            I = 0
            Q = 0
            R = 0
            phase = 0
            if self.live_plot:
                if self.live_in_python:
                    # Live plot the results using matplotlib
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
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), I)
                        plt.xlabel("Vx [V]")
                        plt.ylabel("Vy [V]")
                        plt.colorbar()
                        plt.subplot(222)
                        plt.cla()
                        plt.title("Q [V]")
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), Q)
                        plt.xlabel("Vx [V]")
                        plt.ylabel("Vy [V]")
                        plt.colorbar()
                        plt.subplot(223)
                        plt.cla()
                        plt.title("R [V]")
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), R)
                        plt.xlabel("Vx [V]")
                        plt.ylabel("Vy [V]")
                        plt.colorbar()
                        plt.subplot(224)
                        plt.cla()
                        plt.title("phase [deg]")
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), phase)
                        plt.xlabel("Vx [V]")
                        plt.ylabel("Vy [V]")
                        plt.colorbar()
                        plt.tight_layout()
                        plt.pause(0.1)

                else:
                    # Live plot the results using plottr
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
                # Fetch all results at the end of the program
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
