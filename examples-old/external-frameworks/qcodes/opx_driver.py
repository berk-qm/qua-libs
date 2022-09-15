import time
from typing import Dict, Optional

import numpy as np
from qcodes import (
    Instrument,
    Parameter,
    MultiParameter,
)
from qcodes.utils.helpers import abstractmethod
from qcodes.utils.validators import Numbers
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from macros import round_to_fixed, measurement_macro, spiral

# noinspection PyAbstractClass
class OPX(Instrument):
    """
    Driver for interacting with QM OPX
    """

    def __init__(self, config: Dict, name: str = "OPX", host=None, port=None, **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
        """
        super().__init__(name, **kwargs)

        self.qm = None
        self.qmm = None
        self.config = None
        self.result_handles = None
        self.job = None
        self.prog_id = None
        self.live_in_python = False
        self.live_plot = False
        self.connect(host=host, port=port)
        self.set_config(config=config)

        self.add_parameter("results", label="results", get_cmd=self.get_res)

        self.add_parameter(
            "sim_time",
            unit="ns",
            label="sim_time",
            initial_value=100000,
            vals=Numbers(
                4,
            ),
            get_cmd=None,
            set_cmd=None,
        )

    @abstractmethod
    def get_prog(self):
        pass

    def get_res(self):
        return None

    def execute_prog(self, prog):
        self.job = self.qm.execute(prog)
        self.result_handles = self.job.result_handles

    def simulate_prog(self, prog, duration=1000):
        self.job = self.qm.simulate(prog, SimulationConfig(duration))
        self.job.get_simulated_samples().con1.plot()
        self.result_handles = self.job.result_handles

    def simulate_and_read(self, prog):
        self.simulate_prog(prog, duration=self.sim_time())
        self.result_handles.wait_for_all_values()
        return self.get_res()

    def compile_prog(self, prog):
        self.prog_id = self.qm.compile(prog)

    def execute_compiled_prog(self):
        if self.prog_id is not None:
            pending_job = self.qm.queue.add_compiled(self.prog_id)
            self.job = pending_job.wait_for_execution()
            self.result_handles = self.job.result_handles

    def set_config(self, config):
        self.config = config
        self.qm = self.qmm.open_qm(self.config, close_other_machines=True)

    def connect(self, host=None, port=None):
        self.qmm = QuantumMachinesManager(host=host, port=port)
        self.connect_message()

    def close(self) -> None:
        if self.qm is not None:
            self.qm.close()
        super().close()

    def halt(self) -> None:
        if self.job is not None:
            self.job.halt()

    def connect_message(self, idn_param: str = "IDN", begin_time: Optional[float] = None) -> None:
        """
        Print a standard message on initial connection to an instrument.

        Args:
            idn_param: Name of parameter that returns ID dict.
                Default ``IDN``.
            begin_time: ``time.time()`` when init started.
                Default is ``self._t0``, set at start of ``Instrument.__init__``.
        """
        idn = {"vendor": "Quantum Machines"}
        idn.update(self.get(idn_param))
        if idn["server"][0] == "1":
            idn["model"] = "OPX"
        elif idn["server"][0] == "2":
            idn["model"] = "OPX+"
        else:
            idn["model"] = ""
        t = time.time() - (begin_time or self._t0)

        con_msg = (
            "Connected to: {vendor} {model} in {t:.2f}s. "
            "QOP Version = {server}, SDK Version = {client}.".format(t=t, **idn)
        )
        print(con_msg)
        self.log.info(f"Connected to instrument: {idn}")

    def get_idn(self) -> Dict[str, Optional[str]]:
        return self.qmm.version()


class MyCounter(Parameter):
    def __init__(self, name):
        # only name is required
        super().__init__(
            name,
            label="Times this has been read",
            vals=Numbers(1, 1e9),
            docstring="counts how many times get has been called " "but can be reset to any integer >= 0 by set",
        )
        self._count = 0

    # you must provide a get method, a set method, or both.
    def get_raw(self):
        self._count += 1
        return self._count

    def set_raw(self, val):
        self._count = val
        return self._count


# noinspection PyAbstractClass
class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """

    def __init__(self, startparam, stopparam, numpointsparam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._startparam = startparam
        self._stopparam = stopparam
        self._numpointsparam = numpointsparam

    def get_raw(self):
        return np.linspace(self._startparam(), self._stopparam(), self._numpointsparam())


# noinspection PyAbstractClass
class GeneratedSetPointsSpan(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """

    def __init__(self, spanparam, centerparam, numpointsparam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spanparam = spanparam
        self._centerparam = centerparam
        self._numpointsparam = numpointsparam

    def get_raw(self):
        return np.linspace(
            self._centerparam() - self._spanparam() / 2,
            self._centerparam() + self._spanparam() / 2,
            self._numpointsparam(),
        )


# noinspection PyAbstractClass
class QMDemodParameters(MultiParameter):
    def __init__(self, instr, params, name, names, units, shapes=None, setpoints=None, *args, **kwargs):
        if shapes is None:
            shapes = ((),) * len(params)
        if setpoints is None:
            setpoints = ((),) * len(params)
        super().__init__(name=name, names=names, units=units, shapes=shapes, setpoints=setpoints, *args, **kwargs)

        self._instr = instr
        self._params = params

    def get_raw(self):
        vals = []

        result = self._instr.get_res()
        print(result)
        for param in self._params:
            if param.lower() == "x" or param.lower() == "i":
                vals.append(result["I"])
            elif param.lower() == "y" or param.lower() == "q":
                vals.append(result["Q"])
            elif param.lower() == "r":
                vals.append(result["R"])
            elif param.lower() == "phase" or param.lower() == "phi":
                vals.append(result["Phi"])
            elif param.lower() == "vx":
                vals.append(result["Vx"])
            elif param.lower() == "vy":
                vals.append(result["Vy"])
            else:
                raise NotImplementedError("Only X (I), Y (Q), R or Phase (Phi) are valid inputs")
        return tuple(vals)
