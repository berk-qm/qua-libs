"""
The process to generate the initial state and the build config function:
1. write the build_config function with references to the state
2. copy the config as a key in the state dict variable
3. replace the references in the state by placeholders
4. clear the state from not required fields
5. open a qm
6. generate the bootstrap_state_old.json
"""
import numpy as np
from typing import List, Dict
from scipy.signal.windows import gaussian
from scipy import interpolate


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the I & Q ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [
        float(N * x)
        for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]
    ]


# layer 1: bare state
quam = {
    "analog_outputs": [
        {
            "controller": "con1",
            "output": 1,
            "offset": 0.0,
        },
        {
            "controller": "con1",
            "output": 2,
            "offset": 0.0,
        },
        {
            "controller": "con1",
            "output": 3,
            "offset": 0.0,
        },
        {
            "controller": "con1",
            "output": 4,
            "offset": 0.0,
        },
        {"controller": "con1", "output": 5, "offset": 0.0},
        {"controller": "con1", "output": 6, "offset": 0.0},
        {"controller": "con1", "output": 7, "offset": 0.0},
        {"controller": "con1", "output": 8, "offset": 0.0},
        {"controller": "con1", "output": 9, "offset": 0.0},
        {"controller": "con1", "output": 10, "offset": 0.0},
    ],
    "analog_inputs": [
        {"controller": "con1", "input": 1, "offset": 0.0, "gain_db": 0},
        {"controller": "con1", "input": 2, "offset": 0.0, "gain_db": 0},
    ],
    "analog_waveforms": [
        {"name": "const_wf", "type": "constant", "samples": [0.2]},
        {"name": "saturation_drive_wf", "type": "constant", "samples": [0.2]},
        {
            "name": "zero_wf",
            "type": "constant",
            "samples": [0.0],
        },
    ],
    "digital_waveforms": [{"name": "ON", "samples": [(1, 0)]}],
    "pulses": [
        {
            "name": "const_pulse",
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        {
            "name": "saturation_pulse",
            "operation": "control",
            "length": 100000,
            "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
        },
    ],
    "readout_lines": [
        {
            "length": 2e-6,  # Sec
            "lo_freq": 6.57e9,
            "lo_power": 15,  # dBm
            "I_down": ["con1", 9],  # Down-conversion channels
            "Q_down": ["con1", 10],
        }
    ],
    "readout_resonators": [
        {
            "f_res": 6.45218e9,  # Hz
            "q_factor": None,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "time_of_flight": 260,
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_res": 6.53269e9,  # Hz
            "q_factor": None,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "time_of_flight": 260,
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
    ],
    "qubits": [
        {
            "f_01": 4.52503e9,  # Hz
            "anharmonicity": None,
            "rabi_freq": None,
            "t1": 18e-6,
            "t2": None,
            "t2*": 5e-6,
            "driving": {
                "gate_len": 60e-9,  # Sec
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"90": 0.1, "180": 0.2},
            },
            "wiring": {
                "lo_freq": 4.59e9,
                "lo_power": 11,  # dBm
                "I": ["con1", 1],
                "Q": ["con1", 2],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_01": 4.65097e9,  # Hz
            "anharmonicity": None,
            "rabi_freq": None,
            "t1": 18e-6,
            "t2": None,
            "t2*": 5e-6,
            "driving": {
                "gate_len": 60e-9,  # Sec
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"90": 0.1, "180": 0.2},
            },
            "wiring": {
                "lo_freq": 4.72e9,
                "lo_power": 18,  # dBm
                "I": ["con1", 3],
                "Q": ["con1", 4],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
    ],
    "single_qubit_operations": [
        {"direction": "x", "angle": 180},
        {"direction": "x", "angle": -180},
        {"direction": "x", "angle": 90},
        {"direction": "x", "angle": -90},
        {"direction": "y", "angle": 180},
        {"direction": "y", "angle": -180},
        {"direction": "y", "angle": 90},
        {"direction": "y", "angle": -90},
    ],
    "two_qubit_gates": [
        {
            "name": "cr_c0t1",
            "control": 0,
            "target": 1,
            "gate_time": None,
            "correction_matrix": IQ_imbalance(0, 0),
            "control_pulses": {
                "simple": {
                    "amplitude": 0.1,
                    "duration": 16e-9,
                    "phase": 0,  # in degrees
                },
                "echo": {
                    "amplitude": 0.1,
                    "duration": 16e-9,
                    "phase": 0,
                },
                "cancellation_tone": {
                    "amplitude": 0.1,
                    "phase": 0,
                },
            },
            "interaction strength": {
                "ix": None,
                "iy": None,
                "iz": None,
                "zx": None,
                "zy": None,
                "zz": None,
            },
        },
        {
            "name": "cr_c1t0",
            "control": 1,
            "target": 0,
            "gate_time": None,
            "correction_matrix": IQ_imbalance(0, 0),
            "control_pulses": {
                "simple": {
                    "amplitude": 0.1,
                    "duration": 16e-9,
                    "phase": 0,  # in degrees
                },
                "echo": {
                    "amplitude": 0.1,
                    "duration": 16e-9,
                    "phase": 0,
                },
                "cancellation_tone": {
                    "amplitude": 0.1,
                    "phase": 0,
                },
            },
            "interaction strength": {
                "ix": None,
                "iy": None,
                "iz": None,
                "zx": None,
                "zy": None,
                "zz": None,
            },
        },
    ],
    "running strategy": {"running": True, "start": [], "end": []},
}


def add_qubits(quam: Dict, config: Dict):
    for q in range(len(quam["qubits"])):
        wiring = quam["qubits"][q]["wiring"]
        config["elements"][f"q{q}"] = {
            "mixInputs": {
                "I": (wiring["I"][0], wiring["I"][1]),
                "Q": (wiring["Q"][0], wiring["Q"][1]),
                "lo_frequency": wiring["lo_freq"],
                "mixer": f"mixer_q{q}",
            },
            "intermediate_frequency": round(quam["qubits"][q]["f_01"])
            - wiring["lo_freq"],
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
            },
        }
        if f"mixer_q{q}" not in config["mixers"]:
            config["mixers"][f"mixer_q{q}"] = []
        config["mixers"][f"mixer_q{q}"].append(
            {
                "intermediate_frequency": round(quam["qubits"][q]["f_01"])
                - wiring["lo_freq"],
                "lo_frequency": wiring["lo_freq"],
                "correction": wiring["correction_matrix"],
            }
        )


def add_readout_resonators(quam, config):
    for r, v in enumerate(quam["readout_resonators"]):  # r - idx, v - value
        readout_line = quam["readout_lines"][v["wiring"]["readout_line_index"]]

        config["elements"][f"rr{r}"] = {
            "mixInputs": {
                "I": (readout_line["I_down"][0], readout_line["I_down"][1]),
                "Q": (readout_line["Q_down"][0], readout_line["Q_down"][1]),
                "lo_frequency": round(readout_line["lo_freq"]),
                "mixer": "mixer_rr",
            },
            "intermediate_frequency": round(
                v["f_res"] - readout_line["lo_freq"]
            ),
            "operations": {
                "cw": "const_pulse",
                "readout": f"readout_pulse_rr{r}",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": v["wiring"]["time_of_flight"],
            "smearing": 0,
        }
        if "mixer_rr" not in config["mixers"]:
            config["mixers"]["mixer_rr"] = []
        config["mixers"]["mixer_rr"].append(
            {
                "intermediate_frequency": round(
                    v["f_res"] - readout_line["lo_freq"]
                ),
                "lo_frequency": readout_line["lo_freq"],
                "correction": v["wiring"]["correction_matrix"],
            }
        )
        config["waveforms"][f"readout_wf_rr{r}"] = {
            "type": "constant",
            "sample": v["readout_amplitude"],
        }
        config["pulses"][f"readout_pulse_rr{r}"] = {
            "operation": "measurement",
            "length": round(readout_line["length"] * 1e9),
            "waveforms": {
                "I": f"readout_wf_rr{r}",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": f"rotated_cosine_weights_rr{r}",
                "rotated_sin": f"rotated_sine_weights_rr{r}",
                "rotated_minus_sin": f"rotated_minus_sine_weights_rr{r}",
            },
            "digital_marker": "ON",
        }
        rot_angle_in_pi = v["rotation_angle"] / 180.0 * np.pi
        config["integration_weights"]["cosine_weights"] = {
            "cosine": [(1.0, round(readout_line["length"] * 1e9))],
            "sine": [(0.0, round(readout_line["length"] * 1e9))],
        }
        config["integration_weights"]["sine_weights"] = {
            "cosine": [(0.0, round(readout_line["length"] * 1e9))],
            "sine": [(1.0, round(readout_line["length"] * 1e9))],
        }
        config["integration_weights"]["minus_sine_weights"] = {
            "cosine": [(0.0, round(readout_line["length"] * 1e9))],
            "sine": [(-1.0, round(readout_line["length"] * 1e9))],
        }
        config["integration_weights"][f"rotated_cosine_weights_rr{r}"] = {
            "cosine": [
                (np.cos(rot_angle_in_pi), round(readout_line["length"] * 1e9))
            ],
            "sine": [
                (-np.sin(rot_angle_in_pi), round(readout_line["length"] * 1e9))
            ],
        }
        config["integration_weights"][f"rotated_sine_weights_rr{r}"] = {
            "cosine": [
                (np.sin(rot_angle_in_pi), round(readout_line["length"] * 1e9))
            ],
            "sine": [
                (np.cos(rot_angle_in_pi), round(readout_line["length"] * 1e9))
            ],
        }
        config["integration_weights"][f"rotated_minus_sine_weights_rr{r}"] = {
            "cosine": [
                (-np.sin(rot_angle_in_pi), round(readout_line["length"] * 1e9))
            ],
            "sine": [
                (-np.cos(rot_angle_in_pi), round(readout_line["length"] * 1e9))
            ],
        }


def add_qb_rot(
    quam: Dict,
    config: Dict,
    q: int,
    angle: int,
    direction: str,
    wf_I: List,
    wf_Q: List = None,
):
    """Add single qubit operation

    Args:
        quam (Dict): state of the system
        config (Dict): OPX config we are building/editing
        q (int): index of qubit to whom we are adding operation
        angle (int): angle in degrees
        direction (str):  "x" or "y"
        wf_I: I waveform
        wf_Q: Optional, Q waveform
    """
    if direction not in ["x", "y"]:
        raise ValueError(
            f"Only x and y are accepted directions, received {direction}"
        )
    if q >= len(quam["qubits"]):
        raise ValueError(
            f"Qubit {q} is not configured in quam. Please add qubit {q} first."
        )
    if type(angle) != int:
        raise ValueError("Only integers are accepted as angle.")

    if direction == "x":
        direction_angle = 0
    elif direction == "y":
        direction_angle = np.pi / 2

    if wf_Q is None:
        wf_Q = np.zeros(len(wf_I))
    if len(wf_I) != len(wf_Q):
        raise ValueError("wf_I and wf_Q should have same lengths!")

    wv = np.sign(angle) * (
        wf_I * np.cos(direction_angle) - wf_Q * np.sin(direction_angle)
    )
    if np.all((wv == 0)):
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"] = {
            "type": "constant"
        }
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"]["sample"] = 0
    else:
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"] = {
            "type": "arbitrary"
        }
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"]["samples"] = wv

    wv = np.sign(angle) * (
        wf_I * np.sin(direction_angle) + wf_Q * np.cos(direction_angle)
    )
    if np.all((wv == 0)):
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"] = {
            "type": "constant"
        }
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"]["sample"] = 0
    else:
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"] = {
            "type": "arbitrary"
        }
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"]["samples"] = wv

    config["pulses"][f"{direction}{angle}_pulse_q{q}"] = {
        "operation": "control",
        "length": len(wf_I),
        "waveforms": {
            "I": f"{direction}{angle}_I_wf_q{q}",
            "Q": f"{direction}{angle}_Q_wf_q{q}",
        },
    }
    config["elements"][f"q{q}"]["operations"][
        f"{direction}{angle}"
    ] = f"{direction}{angle}_pulse_q{q}"


def add_cross_resonance_gates(quam, config):
    for v in quam["two_qubit_gates"]:
        gate = v["name"]
        config["elements"][gate] = {
            "mixInputs": {
                "I": (
                    config["elements"][f"q{v['control']}"]["mixInputs"]["I"][0],
                    config["elements"][f"q{v['control']}"]["mixInputs"]["I"][1],
                ),
                "Q": (
                    config["elements"][f"q{v['control']}"]["mixInputs"]["Q"][0],
                    config["elements"][f"q{v['control']}"]["mixInputs"]["Q"][1],
                ),
                "lo_frequency": config["elements"][f'q{v["control"]}'][
                    "mixInputs"
                ]["lo_frequency"],
                "mixer": f'mixer_q{v["control"]}',
            },
            "intermediate_frequency": round(
                quam["qubits"][v["target"]]["f_01"]
            )
            - quam["qubits"][v["control"]]["wiring"]["lo_freq"],
            "operations": {},
        }

        # simple pulses
        config["elements"][gate]["operations"]["simple"] = f"{gate}_simple"
        config["pulses"][f"{gate}_simple"] = {
            "operation": "control",
            "length": round(v["control_pulses"]["simple"]["duration"] * 1e9),
            "waveforms": {
                "I": f"{gate}_simple_wf_i",
                "Q": f"{gate}_simple_wf_q",
            },
        }
        config["waveforms"][f"{gate}_simple_wf_i"] = {
            "type": "constant",
            "sample": v["control_pulses"]["simple"]["amplitude"]
            * np.cos(v["control_pulses"]["simple"]["phase"] / 180 * np.pi),
        }
        config["waveforms"][f"{gate}_simple_wf_q"] = {
            "type": "constant",
            "sample": v["control_pulses"]["simple"]["amplitude"]
            * np.sin(v["control_pulses"]["simple"]["phase"] / 180 * np.pi),
        }

        # echo sequence
        config["elements"][gate]["operations"][
            "echo_plus"
        ] = f"{gate}_echo_plus"
        config["pulses"][f"{gate}_echo_plus"] = {
            "operation": "control",
            "length": round(v["control_pulses"]["echo"]["duration"] * 1e9),
            "waveforms": {
                "I": f"{gate}_echo_plus_wf_i",
                "Q": f"{gate}_echo_plus_wf_q",
            },
        }
        config["waveforms"][f"{gate}_echo_plus_wf_i"] = {
            "type": "constant",
            "sample": v["control_pulses"]["echo"]["amplitude"]
            * np.cos(v["control_pulses"]["echo"]["phase"] / 180 * np.pi),
        }
        config["waveforms"][f"{gate}_echo_plus_wf_q"] = {
            "type": "constant",
            "sample": v["control_pulses"]["echo"]["amplitude"]
            * np.sin(v["control_pulses"]["echo"]["phase"] / 180 * np.pi),
        }

        config["elements"][gate]["operations"][
            "echo_minus"
        ] = f"{gate}_echo_minus"
        config["pulses"][f"{gate}_echo_minus"] = {
            "operation": "control",
            "length": round(v["control_pulses"]["echo"]["duration"] * 1e9),
            "waveforms": {
                "I": f"{gate}_echo_minus_wf_i",
                "Q": f"{gate}_echo_minus_wf_q",
            },
        }
        config["waveforms"][f"{gate}_echo_minus_wf_i"] = {
            "type": "constant",
            "sample": -1
            * v["control_pulses"]["echo"]["amplitude"]
            * np.cos(v["control_pulses"]["echo"]["phase"] / 180 * np.pi),
        }
        config["waveforms"][f"{gate}_echo_minus_wf_q"] = {
            "type": "constant",
            "sample": -1
            * v["control_pulses"]["echo"]["amplitude"]
            * np.sin(v["control_pulses"]["echo"]["phase"] / 180 * np.pi),
        }

        # cancellation tone on target element
        config["elements"][f'q{v["target"]}']["operations"][
            f"{gate}_cancellation_plus"
        ] = f"{gate}_cancellation_plus"
        config["pulses"][f"{gate}_cancellation_plus"] = {
            "operation": "control",
            "length": round(v["control_pulses"]["echo"]["duration"] * 1e9),
            "waveforms": {
                "I": f"{gate}_cancellation_plus_wf_i",
                "Q": f"{gate}_cancellation_plus_wf_q",
            },
        }
        config["waveforms"][f"{gate}_cancellation_plus_wf_i"] = {
            "type": "constant",
            "sample": v["control_pulses"]["cancellation_tone"]["amplitude"]
            * np.cos(
                v["control_pulses"]["cancellation_tone"]["phase"] / 180 * np.pi
            ),
        }
        config["waveforms"][f"{gate}_cancellation_plus_wf_q"] = {
            "type": "constant",
            "sample": v["control_pulses"]["cancellation_tone"]["amplitude"]
            * np.sin(
                v["control_pulses"]["cancellation_tone"]["phase"] / 180 * np.pi
            ),
        }

        config["elements"][f'q{v["target"]}']["operations"][
            f"{gate}_cancellation_minus"
        ] = f"{gate}_cancellation_minus"
        config["pulses"][f"{gate}_cancellation_minus"] = {
            "operation": "control",
            "length": round(v["control_pulses"]["echo"]["duration"] * 1e9),
            "waveforms": {
                "I": f"{gate}_cancellation_minus_wf_i",
                "Q": f"{gate}_cancellation_minus_wf_q",
            },
        }
        config["waveforms"][f"{gate}_cancellation_minus_wf_i"] = {
            "type": "constant",
            "sample": -1
            * v["control_pulses"]["cancellation_tone"]["amplitude"]
            * np.cos(
                v["control_pulses"]["cancellation_tone"]["phase"] / 180 * np.pi
            ),
        }
        config["waveforms"][f"{gate}_cancellation_minus_wf_q"] = {
            "type": "constant",
            "sample": -1
            * v["control_pulses"]["cancellation_tone"]["amplitude"]
            * np.sin(
                v["control_pulses"]["cancellation_tone"]["phase"] / 180 * np.pi
            ),
        }

        config["mixers"][f"mixer_q{v['control']}"].append(
            {
                "intermediate_frequency": round(
                    quam["qubits"][v["target"]]["f_01"]
                )
                - quam["qubits"][v["control"]]["wiring"]["lo_freq"],
                "lo_frequency": quam["qubits"][v["control"]]["wiring"][
                    "lo_freq"
                ],
                "correction": v["correction_matrix"],
            }
        )


def add_control_operation_single(config, element, operation_name, wf):
    pulse_name = element + "_" + operation_name + "_in"
    config["waveforms"][pulse_name + "_single"] = {
        "type": "arbitrary",
        "samples": list(wf),
    }
    config["pulses"][pulse_name] = {
        "operation": "control",
        "length": len(wf),
        "waveforms": {"I": pulse_name + "_single"},
    }
    config["elements"][element]["operations"][operation_name] = pulse_name


def add_control_operation_iq(config, element, operation_name, wf_i, wf_q):
    pulse_name = element + "_" + operation_name + "_in"
    config["waveforms"][pulse_name + "_i"] = {
        "type": "arbitrary",
        "samples": list(wf_i),
    }
    config["waveforms"][pulse_name + "_q"] = {
        "type": "arbitrary",
        "samples": list(wf_q),
    }
    config["pulses"][pulse_name] = {
        "operation": "control",
        "length": len(wf_i),
        "waveforms": {"I": pulse_name + "_i", "Q": pulse_name + "_q"},
    }
    config["elements"][element]["operations"][operation_name] = pulse_name


def add_analog_outputs(quam, config):
    for o in quam["analog_outputs"]:
        if o["controller"] not in config["controllers"]:
            config["controllers"][o["controller"]] = {}
        if "analog_outputs" not in config["controllers"][o["controller"]]:
            config["controllers"][o["controller"]]["analog_outputs"] = {}
        config["controllers"][o["controller"]]["analog_outputs"][
            str(o["output"])
        ] = {"offset": o["offset"]}


def add_analog_inputs(quam, config):
    for i in quam["analog_inputs"]:
        if i["controller"] not in config["controllers"]:
            config["controllers"][i["controller"]] = {}
        if "analog_inputs" not in config["controllers"][i["controller"]]:
            config["controllers"][i["controller"]]["analog_inputs"] = {}
        config["controllers"][i["controller"]]["analog_inputs"][
            str(i["input"])
        ] = {
            "offset": i["offset"],
            "gain_db": i["gain_db"],
        }


def add_analog_waveforms(quam, config):
    for wf in quam["analog_waveforms"]:
        if wf["type"] == "constant":
            if len(wf["samples"]) != 1:
                raise ValueError(
                    f'Constant analog waveform {wf["name"]} has to have samples length of 1 (currently {len(wf["samples"])})'
                )

            config["waveforms"][wf["name"]] = {
                "type": wf["type"],
                "sample": wf["samples"][0],
            }
        else:
            if len(wf["samples"]) <= 1:
                raise ValueError(
                    f'Analog waveform {wf["name"]} has single sample, and should be then of type "constant" instead of {wf["type"]}.'
                )
            config["waveforms"][wf["name"]] = {
                "type": wf["type"],
                "samples": wf["samples"],
            }


def add_digital_waveforms(quam, config):
    for wf in quam["digital_waveforms"]:
        config["digital_waveforms"][wf["name"]] = {"samples": wf["samples"]}


def add_pulses(quam, config):
    for pulse in quam["pulses"]:
        config["pulses"][pulse["name"]] = {
            "operation": pulse["operation"],
            "length": pulse["length"],
            "waveforms": {
                "I": pulse["waveforms"]["I"],
                "Q": pulse["waveforms"]["Q"],
            },
        }


def build_config(quam):
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {},
                "digital_outputs": {},
                "analog_inputs": {},
            }
        },
        "elements": {},
        "pulses": {},
        "waveforms": {},
        "digital_waveforms": {},
        "integration_weights": {},
        "mixers": {},
    }

    add_analog_outputs(quam, config)

    add_analog_inputs(quam, config)

    add_analog_waveforms(quam, config)

    add_digital_waveforms(quam, config)

    add_pulses(quam, config)

    add_qubits(quam, config)

    add_readout_resonators(quam, config)

    for single_qubit_operation in quam["single_qubit_operations"]:
        for q in range(len(quam["qubits"])):
            if quam["qubits"][q]["driving"]["gate_shape"] == "gaussian":
                add_qb_rot(
                    quam,
                    config,
                    q,
                    single_qubit_operation["angle"],
                    single_qubit_operation["direction"],
                    quam["qubits"][q]["driving"]["angle2volt"][
                        str(abs(single_qubit_operation["angle"]))
                    ]
                    * gaussian(
                        round(quam["qubits"][q]["driving"]["gate_len"] * 1e9),
                        round(
                            quam["qubits"][q]["driving"]["gate_sigma"] * 1e9
                        ),
                    ),
                )  # +180 and -180 have same amplitude
            else:
                raise ValueError(
                    f'Gate shape {quam["qubits"][q]["driving"]["gate_shape"]} not recognized.'
                )

    add_cross_resonance_gates(quam, config)
    return config


def generate_bootstrap_state():
    config = build_config(quam)
    # from qm.QuantumMachinesManager import QuantumMachinesManager
    # qmm = QuantumMachinesManager(host="172.16.2.103", port=82)
    # qm = qmm.open_qm(config)

    import json

    with open("bootstrap_state.json", "w") as fp:
        json.dump(quam, fp, indent=2)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open("configuration_parsed.json", "w") as fp:
        json.dump(build_config(quam), fp, cls=NumpyEncoder, indent=2)
        # convert_file.write(json.dumps(build_config(quam)))


if __name__ == "__main__":
    generate_bootstrap_state()
