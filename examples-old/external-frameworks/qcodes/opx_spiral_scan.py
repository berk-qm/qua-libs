from qcodes.utils.validators import Arrays
from opx_driver import *
from qm.qua import *
from macros import round_to_fixed, measurement_macro, spiral
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter

# noinspection PyAbstractClass
class OPXSpiralScan(OPX):
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
            "Vx_span",
            unit="V",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vx_center",
            unit="V",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vy_span",
            unit="V",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vy_center",
            unit="V",
            vals=Numbers(-0.5, 0.5),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "n_points",
            unit="",
            vals=Numbers(1, 1e9),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Vx_axis",
            unit="V",
            label="Vx Axis",
            parameter_class=GeneratedSetPointsSpan,
            spanparam=self.Vx_span,
            centerparam=self.Vx_center,
            numpointsparam=self.n_points,
            vals=Arrays(shape=(self.n_points.get_latest,)),
        )

        self.add_parameter(
            "Vy_axis",
            unit="V",
            label="Vy Axis",
            parameter_class=GeneratedSetPointsSpan,
            spanparam=self.Vy_span,
            centerparam=self.Vy_center,
            numpointsparam=self.n_points,
            vals=Arrays(shape=(self.n_points.get_latest,)),
        )
        self.add_parameter(
            "n_avg",
            unit="",
            vals=Numbers(
                1,
            ),
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
        # Check that the resolution is odd to form the spiral
        assert self.n_points() % 2 == 1, "the parameter 'n_points' must be odd {}".format(self.n_points())
        # Set the dc offset to the center of the spiral (these must be set to the slow voltage source)
        self.qm.set_output_dc_offset_by_element(self.y_element(), "single", self.Vy_center())
        self.qm.set_output_dc_offset_by_element(self.x_element(), "single", self.Vx_center())
        # Get the gate pulse amplitude and derive the voltage step
        pulse = self.config["elements"][self.x_element()]["operations"][self.gate_operation()]
        wf = self.config["pulses"][pulse]["waveforms"]["single"]
        dx = round_to_fixed(self.Vx_span() / ((self.n_points() - 1) * self.config["waveforms"][wf].get("sample")))
        dy = round_to_fixed(self.Vy_span() / ((self.n_points() - 1) * self.config["waveforms"][wf].get("sample")))
        print(
            f"X scan from {(self.Vx_center()-self.Vx_span()/2)*1000:.2f} mV "
            f"to {(self.Vx_center()+self.Vx_span()/2)*1000:.2f} mV "
            f"in {self.n_points() - 1} steps "
            f"of {dx* self.config['waveforms'][wf].get('sample')*1000:.2f} mV."
        )
        print(
            f"Y scan from {(self.Vy_center()-self.Vy_span()/2)*1000:.2f} mV "
            f"to {(self.Vy_center()+self.Vy_span()/2)*1000:.2f} mV "
            f"in {self.n_points() - 1} steps "
            f"of {dy* self.config['waveforms'][wf].get('sample')*1000:.2f} mV."
        )
        with program() as prog:
            i = declare(int)  # an index variable for the x index
            j = declare(int)  # an index variable for the y index

            Vx = declare(fixed)  # a variable to keep track of the Vx coordinate
            Vy = declare(fixed)  # a variable to keep track of the Vy coordinate
            Vx_st = declare_stream()
            Vy_st = declare_stream()
            average = declare(int)  # an index variable for the average
            n_st = declare_stream()
            moves_per_edge = declare(int)  # the number of moves per edge [1, self.n_points()]
            completed_moves = declare(int)  # the number of completed move [0, self.n_points() ** 2]
            movement_direction = declare(fixed)  # which direction to move {-1., 1.}

            # declaring the measured variables and their streams
            I, Q = declare(fixed), declare(fixed)
            I_st, Q_st = declare_stream(), declare_stream()

            with for_(average, 0, average < self.n_avg(), average + 1):
                # initialising variables
                assign(moves_per_edge, 1)
                assign(completed_moves, 0)
                assign(movement_direction, +1)
                assign(Vx, 0.0)
                assign(Vy, 0.0)

                ramp_to_zero(self.x_element(), duration=4)
                ramp_to_zero(self.y_element(), duration=4)
                align(self.x_element(), self.y_element(), self.readout_element())
                # for the first pixel it is unnecessary to move before measuring
                measurement_macro(measured_element=self.readout_element(), I=I, I_stream=I_st, Q=Q, Q_stream=Q_st)
                save(Vx, Vx_st)
                save(Vy, Vy_st)

                with while_(completed_moves < self.n_points() * (self.n_points() - 1)):
                    # for_ loop to move the required number of moves in the x direction
                    with for_(i, 0, i < moves_per_edge, i + 1):
                        assign(Vx, Vx + movement_direction * dx * self.config["waveforms"][wf].get("sample"))
                        # if the x coordinate should be 0, ramp to zero to remove fixed point arithmetic errors accumulating
                        with if_(Vx == 0.0):
                            ramp_to_zero(self.x_element(), duration=4)
                        # playing the constant pulse to move to the next pixel
                        with else_():
                            play(self.gate_operation() * amp(movement_direction * dx), self.x_element())

                        # Make sure that we measure after the pulse has settled
                        align(self.x_element(), self.y_element(), self.readout_element())
                        if self.wait_time() >= 4:  # if logic to enable wait_time = 0 without error
                            wait(self.wait_time() // 4, self.readout_element())
                        # Measurement
                        measurement_macro(
                            measured_element=self.readout_element(), I=I, I_stream=I_st, Q=Q, Q_stream=Q_st
                        )
                        save(Vx, Vx_st)
                        save(Vy, Vy_st)
                    # for_ loop to move the required number of moves in the y direction
                    with for_(j, 0, j < moves_per_edge, j + 1):
                        assign(Vy, Vy + movement_direction * dy * self.config["waveforms"][wf].get("sample"))
                        # if the y coordinate should be 0, ramp to zero to remove fixed point arithmetic errors accumulating
                        with if_(Vy == 0.0):
                            ramp_to_zero(self.y_element(), duration=4)
                        # playing the constant pulse to move to the next pixel
                        with else_():
                            play(self.gate_operation() * amp(movement_direction * dy), self.y_element())

                        # Make sure that we measure after the pulse has settled
                        align(self.x_element(), self.y_element(), self.readout_element())
                        if self.wait_time() >= 4:  # if logic to enable wait_time = 0 without error
                            wait(self.wait_time() // 4, self.readout_element())
                        # Measurement
                        measurement_macro(
                            measured_element=self.readout_element(), I=I, I_stream=I_st, Q=Q, Q_stream=Q_st
                        )
                        save(Vx, Vx_st)
                        save(Vy, Vy_st)
                    # updating the variables
                    assign(completed_moves, completed_moves + 2 * moves_per_edge)  # * 2 because moves in both x and y
                    # *-1 as subsequent steps in the opposite direction
                    assign(movement_direction, movement_direction * -1)
                    # moving one row/column out so need one more move_per_edge
                    assign(moves_per_edge, moves_per_edge + 1)

                # filling in the final x row, which was not covered by the previous for_ loop
                with for_(i, 0, i < moves_per_edge - 1, i + 1):
                    assign(Vx, Vx + movement_direction * dx * self.config["waveforms"][wf].get("sample"))

                    # if the x coordinate should be 0, ramp to zero to remove fixed point arithmetic errors accumulating
                    with if_(Vx == 0.0):
                        ramp_to_zero(self.x_element(), duration=4)
                    # playing the constant pulse to move to the next pixel
                    with else_():
                        play(self.gate_operation() * amp(movement_direction * dx), self.x_element())

                    # Make sure that we measure after the pulse has settled
                    align(self.x_element(), self.y_element(), self.readout_element())
                    if self.wait_time() >= 4:
                        wait(self.wait_time() // 4, self.readout_element())
                    # Measurement
                    measurement_macro(measured_element=self.readout_element(), I=I, I_stream=I_st, Q=Q, Q_stream=Q_st)
                    save(Vx, Vx_st)
                    save(Vy, Vy_st)
                # aligning and ramping to zero to return to initial state
                align(self.x_element(), self.y_element(), self.readout_element())
                ramp_to_zero(self.x_element(), duration=4)
                ramp_to_zero(self.y_element(), duration=4)
                save(average, n_st)

            with stream_processing():
                I_st.buffer(self.n_points() * self.n_points()).average().save("I")
                Q_st.buffer(self.n_points() * self.n_points()).average().save("Q")
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
            n = self.n_points()
            return {
                "I": [[0] * n] * n,
                "Q": [[0] * n] * n,
                "R": [[0] * n] * n,
                "Phi": [[0] * n] * n,
                "Vx": [[0] * n] * n,
                "Vy": [[0] * n] * n,
            }
        else:
            I = 0
            Q = 0
            R = 0
            phase = 0
            order = spiral(self.n_points()).T
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
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), I[order])
                        plt.xlabel("Vx [V]")
                        plt.ylabel("Vy [V]")
                        plt.colorbar()
                        plt.subplot(222)
                        plt.cla()
                        plt.title("Q [V]")
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), Q[order])
                        plt.xlabel("Vx [V]")
                        plt.ylabel("Vy [V]")
                        plt.colorbar()
                        plt.subplot(223)
                        plt.cla()
                        plt.title("R [V]")
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), R[order])
                        plt.xlabel("Vx [V]")
                        plt.ylabel("Vy [V]")
                        plt.colorbar()
                        plt.subplot(224)
                        plt.cla()
                        plt.title("phase [deg]")
                        plt.pcolor(self.Vx_axis(), self.Vy_axis(), phase[order])
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
        return {"I": I[order], "Q": Q[order], "R": R[order], "Phi": phase[order]}
