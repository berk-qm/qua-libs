"""
The process to generate the initial state and the build config function:
1. write the build_config function with references to the state
2. copy the config as a key in the state dict variable
3. replace the references in the state by placeholders
4. clear the state from not required fields
5. open a qm
6. generate the bootstrap_state_old.json
"""

from quam import QuAM
import numpy as np
from typing import List, Dict
from scipy.signal.windows import gaussian
import json
import os
from datetime import datetime
from quam_sdk.viewers import qprint
from qm.qua import set_dc_offset
from pprint import pprint
import matplotlib.pyplot as plt
from qualang_tools.config.integration_weights_tools import convert_integration_weights

READOUT_RESONATORS_PER_FEED_LINE = 6

NUMBER_OF_QUBITS_W_CHARGE = 2
NUMBER_OF_QUBITS_WO_CHARGE = 4
NUMBER_OF_DRIVE_LINES = 1
NUMBER_OF_QP_INJECTORS = 2
NUMBER_OF_QUBITS_PER_DRIVE_LINE = NUMBER_OF_QUBITS_W_CHARGE + NUMBER_OF_QUBITS_WO_CHARGE


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the 'I' & 'Q' ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the 'I' & 'Q' ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


def find_lo_freq(state: QuAM, index: int):
    for x in state.drive_lines:
        if index in x.qubits:
            return x.lo_freq
    raise ValueError(f"Qubit {index} is not associated with any lo in state!")


def add_qubits(state: QuAM, config: Dict, qb_list: list):
    for q in qb_list:
        wiring = state.qubits[q].wiring
        lo_freq = find_lo_freq(state, q)
        if state.qubits[q].threads.thread_cond:
            config["elements"][state.qubits[q].name] = {
                "thread": state.qubits[q].threads.thread,
                "mixInputs": {
                    "I": (
                        state.drive_lines[wiring.drive_line_index].I.controller,
                        state.drive_lines[wiring.drive_line_index].I.channel,
                    ),
                    "Q": (
                        state.drive_lines[wiring.drive_line_index].Q.controller,
                        state.drive_lines[wiring.drive_line_index].Q.channel,
                    ),
                    "lo_frequency": lo_freq,
                    # "mixer": f"mixer_drive_line{q}",
                    "mixer": f"mixer_drive_line{state.qubits[q].wiring.drive_line_index}",
                },
                "intermediate_frequency": round(state.qubits[q].f_01) - lo_freq,
                "digitalInputs": {
                    "switch": {
                        "port": (state.drive_lines[state.qubits[q].wiring.drive_line_index].switch.controller, state.drive_lines[state.qubits[q].wiring.drive_line_index].switch.channel),
                        "delay": state.qubits[q].wiring.switch_delay,
                        "buffer": state.qubits[q].wiring.switch_buffer,
                    },
                },
                "operations": {
                    state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                },
            }
        else:
            config["elements"][state.qubits[q].name] = {
                "mixInputs": {
                    "I": (
                        state.drive_lines[wiring.drive_line_index].I.controller,
                        state.drive_lines[wiring.drive_line_index].I.channel,
                    ),
                    "Q": (
                        state.drive_lines[wiring.drive_line_index].Q.controller,
                        state.drive_lines[wiring.drive_line_index].Q.channel,
                    ),
                    "lo_frequency": lo_freq,
                    # "mixer": f"mixer_drive_line{q}",
                    "mixer": f"mixer_drive_line{state.qubits[q].wiring.drive_line_index}",
                },
                "intermediate_frequency": round(state.qubits[q].f_01) - lo_freq,
                "digitalInputs": {
                    "switch": {
                        "port": (state.drive_lines[state.qubits[q].wiring.drive_line_index].switch.controller, state.drive_lines[state.qubits[q].wiring.drive_line_index].switch.channel),
                        "delay": state.qubits[q].wiring.switch_delay,
                        "buffer": state.qubits[q].wiring.switch_buffer,
                    },
                },
                "operations": {
                    state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                },
            }
        # add offsets
        config["controllers"][state.drive_lines[wiring.drive_line_index].I.controller]["analog_outputs"][
            str(state.drive_lines[wiring.drive_line_index].I.channel)
        ]["offset"] = state.drive_lines[wiring.drive_line_index].I.offset
        config["controllers"][state.drive_lines[wiring.drive_line_index].Q.controller]["analog_outputs"][
            str(state.drive_lines[wiring.drive_line_index].Q.channel)
        ]["offset"] = state.drive_lines[wiring.drive_line_index].Q.offset

        # add charge element
        config["elements"][state.qubits[q].name + "_charge"] = {
            "singleInput": {
                "port": (
                    wiring.charge_line.controller,
                    wiring.charge_line.channel,
                )
            },
            "operations": {
                state.common_operation.name: f"{state.common_operation.name}_single_pulse",
            },
        }
        # add operations for charge line
        for op in state.qubits[q].sequence_states.constant:
            config["elements"][state.qubits[q].name + "_charge"]["operations"][op.name] = (
                state.qubits[q].name + f"_charge_{op.name}"
            )
            # add pulse
            config["pulses"][state.qubits[q].name + f"_charge_{op.name}"] = {
                "operation": "control",
                "length": op.length * 1e9,
                "waveforms": {"single": state.qubits[q].name + f"_charge_{op.name}" + "_wf"},
            }
            config["waveforms"][state.qubits[q].name + f"_charge_{op.name}" + "_wf"] = {
                "type": "constant",
                "sample": op.amplitude,
            }
        for op in state.qubits[q].sequence_states.arbitrary:
            config["elements"][state.qubits[q].name + "_charge"]["operations"][op.name] = (
                state.qubits[q].name + f"_charge_{op.name}"
            )
            # add pulse
            config["pulses"][state.qubits[q].name + f"_charge_{op.name}"] = {
                "operation": "control",
                "length": len(op.waveform),
                "waveforms": {"single": state.qubits[q].name + f"_charge_{op.name}" + "_wf"},
            }
            config["waveforms"][state.qubits[q].name + f"_charge_{op.name}" + "_wf"] = {
                "type": "arbitrary",
                "samples": op.waveform,
            }

        config["elements"][state.qubits[q].name + "_charge_sticky"] = {
            "singleInput": {
                "port": (
                    wiring.charge_line.controller,
                    wiring.charge_line.channel,
                )
            },
            "operations": {
                state.common_operation.name: f"{state.common_operation.name}_single_pulse",
            },
            "hold_offset" "": {"duration": 1},
        }
        # add operations for charge line
        for op in state.qubits[q].sequence_states.constant:
            config["elements"][state.qubits[q].name + "_charge_sticky"]["operations"][op.name] = (
                state.qubits[q].name + f"_charge_{op.name}"
            )

            # add pulse
            config["pulses"][state.qubits[q].name + f"_charge_{op.name}"] = {
                "operation": "control",
                "length": op.length * 1e9,
                "waveforms": {"single": state.qubits[q].name + f"_charge_{op.name}" + "_wf"},
            }
            config["waveforms"][state.qubits[q].name + f"_charge_{op.name}" + "_wf"] = {
                "type": "constant",
                "sample": op.amplitude,
            }
        for op in state.qubits[q].sequence_states.arbitrary:
            config["elements"][state.qubits[q].name + "_charge_sticky"]["operations"][op.name] = (
                state.qubits[q].name + f"_charge_{op.name}"
            )
            # add pulse
            config["pulses"][state.qubits[q].name + f"_charge_{op.name}"] = {
                "operation": "control",
                "length": len(op.waveform),
                "waveforms": {"single": state.qubits[q].name + f"_charge_{op.name}" + "_wf"},
            }
            config["waveforms"][state.qubits[q].name + f"_charge_{op.name}" + "_wf"] = {
                "type": "arbitrary",
                "samples": op.waveform,
            }
        # add filters
        config["controllers"][wiring.charge_line.controller]["analog_outputs"][str(wiring.charge_line.channel)][
            "filter"
        ] = {
            "feedforward": wiring.charge_filter_coefficients.feedforward,
            "feedback": wiring.charge_filter_coefficients.feedback,
        }
        # add offsets
        config["controllers"][wiring.charge_line.controller]["analog_outputs"][str(wiring.charge_line.channel)][
            "offset"
        ] = wiring.charge_line.offset

    # add cross talk
    for i in range(len(state.crosstalk_matrix.fast)):
        crosstalk = {}
        q_i = state.qubits[i]
        for j in range(len(state.crosstalk_matrix.fast[i])):
            q_j = state.qubits[j]
            crosstalk[q_j.wiring.charge_line.channel] = state.crosstalk_matrix.fast[i][j]
        if q_i.index in qb_list and q_j.index in qb_list:
            config["controllers"][q_i.wiring.charge_line.controller]["analog_outputs"][
                str(q_i.wiring.charge_line.channel)
            ]["crosstalk"] = crosstalk


def add_qubits_wo_charge(state: QuAM, config: Dict, qb_wo_charge_list: list):
    for q in qb_wo_charge_list:
        wiring = state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring
        lo_freq = find_lo_freq(state, q)
        if state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].threads.thread_cond:
            config["elements"][state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name] = {
                "thread": state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].threads.thread,
                "mixInputs": {
                    "I": (
                        state.drive_lines[wiring.drive_line_index].I.controller,
                        state.drive_lines[wiring.drive_line_index].I.channel,
                    ),
                    "Q": (
                        state.drive_lines[wiring.drive_line_index].Q.controller,
                        state.drive_lines[wiring.drive_line_index].Q.channel,
                    ),
                    "lo_frequency": lo_freq,
                    # "mixer": f"mixer_drive_line{q}",
                    "mixer": f"mixer_drive_line{state.qubits_wo_charge[q-NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index}",
                },
                "intermediate_frequency": round(state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].f_01) - lo_freq,
                "digitalInputs": {
                    "switch": {
                        "port": (state.drive_lines[state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index].switch.controller,
                                 state.drive_lines[state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index].switch.channel),
                        "delay": state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.switch_delay,
                        "buffer": state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.switch_buffer,
                    },
                },
                "operations": {
                    state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                },
            }
        else:
            config["elements"][state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name] = {
                "mixInputs": {
                    "I": (
                        state.drive_lines[wiring.drive_line_index].I.controller,
                        state.drive_lines[wiring.drive_line_index].I.channel,
                    ),
                    "Q": (
                        state.drive_lines[wiring.drive_line_index].Q.controller,
                        state.drive_lines[wiring.drive_line_index].Q.channel,
                    ),
                    "lo_frequency": lo_freq,
                    # "mixer": f"mixer_drive_line{q}",
                    "mixer": f"mixer_drive_line{state.qubits_wo_charge[q-NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index}",
                },
                "intermediate_frequency": round(state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].f_01) - lo_freq,
                "digitalInputs": {
                    "switch": {
                        "port": (state.drive_lines[state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index].switch.controller,
                                 state.drive_lines[state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index].switch.channel),
                        "delay": state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.switch_delay,
                        "buffer": state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.switch_buffer,
                    },
                },
                "operations": {
                    state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                },
            }
        # add offsets
        config["controllers"][state.drive_lines[wiring.drive_line_index].I.controller]["analog_outputs"][
            str(state.drive_lines[wiring.drive_line_index].I.channel)
        ]["offset"] = state.drive_lines[wiring.drive_line_index].I.offset
        config["controllers"][state.drive_lines[wiring.drive_line_index].Q.controller]["analog_outputs"][
            str(state.drive_lines[wiring.drive_line_index].Q.channel)
        ]["offset"] = state.drive_lines[wiring.drive_line_index].Q.offset


def add_mixers(state: QuAM, config: Dict, qb_list: list):
    for q in range(len(state.drive_lines)):
        lo_freq = state.drive_lines[q].lo_freq

        if f"mixer_drive_line{q}" not in config["mixers"]:
            config["mixers"][f"mixer_drive_line{q}"] = []

        for j in state.drive_lines[q].qubits:
            if j in qb_list:
                config["mixers"][f"mixer_drive_line{q}"].append(
                    {
                        "intermediate_frequency": round(state.qubits[j].f_01) - lo_freq,
                        "lo_frequency": lo_freq,
                        "correction": IQ_imbalance(
                            state.qubits[j].wiring.correction_matrix.gain,
                            state.qubits[j].wiring.correction_matrix.phase,
                        ),
                    }
                )
            else:
                config["mixers"][f"mixer_drive_line{q}"].append(
                    {
                        "intermediate_frequency": round(state.qubits_wo_charge[j - len(qb_list)].f_01) - lo_freq,
                        "lo_frequency": lo_freq,
                        "correction": IQ_imbalance(
                            state.qubits_wo_charge[j - len(qb_list)].wiring.correction_matrix.gain,
                            state.qubits_wo_charge[j - len(qb_list)].wiring.correction_matrix.phase,
                        ),
                    }
                )

        for z in range(len(state.qubits)):
            for t in range(len(state.drive_lines[q].qubits)):
                if z == state.drive_lines[q].qubits[t] and z in qb_list:
                    config["elements"][state.qubits[z].name]["mixInputs"]["mixer"] = f"mixer_drive_line{q}"


def add_readout_resonators(state: QuAM, config: Dict, qb_list: list, qb_wo_charge_list: list):
    for r, v in enumerate(state.readout_resonators):  # r - idx, v - value
        if r in qb_list:
            readout_line = state.readout_lines[v.wiring.readout_line_index]

            if state.readout_resonators[r].threads.thread_cond:
                config["elements"][state.readout_resonators[r].name] = {
                    "thread": state.readout_resonators[r].threads.thread,
                    "mixInputs": {
                        "I": (
                            readout_line.I_up.controller,
                            readout_line.I_up.channel,
                        ),
                        "Q": (
                            readout_line.Q_up.controller,
                            readout_line.Q_up.channel,
                        ),
                        "lo_frequency": round(readout_line.lo_freq),
                        "mixer": f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}",
                    },
                    "intermediate_frequency": round(v.f_opt - readout_line.lo_freq),
                    "digitalInputs": {
                        "switch": {
                            "port": (state.readout_lines[0].switch.controller, state.readout_lines[0].switch.channel),
                            "delay": v.wiring.switch_delay,
                            "buffer": v.wiring.switch_buffer,
                        },
                    },
                    "operations": {
                        state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                        "readout": f"readout_pulse_" + state.readout_resonators[r].name,
                    },
                    "outputs": {
                        "out1": (
                            readout_line.I_down.controller,
                            readout_line.I_down.channel,
                        ),
                        # "out2": (
                        #     readout_line.Q_down.controller,
                        #     readout_line.Q_down.channel,
                        # ),
                    },
                    "time_of_flight": v.wiring.time_of_flight,
                    "smearing": 0,
                }
            else:
                config["elements"][state.readout_resonators[r].name] = {
                    "mixInputs": {
                        "I": (
                            readout_line.I_up.controller,
                            readout_line.I_up.channel,
                        ),
                        "Q": (
                            readout_line.Q_up.controller,
                            readout_line.Q_up.channel,
                        ),
                        "lo_frequency": round(readout_line.lo_freq),
                        "mixer": f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}",
                    },
                    "intermediate_frequency": round(v.f_opt - readout_line.lo_freq),
                    "digitalInputs": {
                        "switch": {
                            "port": (state.readout_lines[0].switch.controller, state.readout_lines[0].switch.channel),
                            "delay": v.wiring.switch_delay,
                            "buffer": v.wiring.switch_buffer,
                        },
                    },
                    "operations": {
                        state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                        "readout": f"readout_pulse_" + state.readout_resonators[r].name,
                    },
                    "outputs": {
                        "out1": (
                            readout_line.I_down.controller,
                            readout_line.I_down.channel,
                        ),
                        # "out2": (
                        #     readout_line.Q_down.controller,
                        #     readout_line.Q_down.channel,
                        # ),
                    },
                    "time_of_flight": v.wiring.time_of_flight,
                    "smearing": 0,
                }
            # add mixers
            if f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}" not in config["mixers"]:
                config["mixers"][f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}"] = []
            config["mixers"][f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}"].append(
                {
                    "intermediate_frequency": round(v.f_opt - readout_line.lo_freq),
                    "lo_frequency": readout_line.lo_freq,
                    "correction": IQ_imbalance(v.wiring.correction_matrix.gain, v.wiring.correction_matrix.phase),
                }
            )
            # add offset
            config["controllers"][readout_line.I_up.controller]["analog_outputs"][str(readout_line.I_up.channel)][
                "offset"
            ] = readout_line.I_up.offset
            config["controllers"][readout_line.Q_up.controller]["analog_outputs"][str(readout_line.Q_up.channel)][
                "offset"
            ] = readout_line.Q_up.offset
            config["controllers"][readout_line.I_down.controller]["analog_inputs"][str(readout_line.I_down.channel)][
                "offset"
            ] = readout_line.I_down.offset
            # config["controllers"][readout_line.Q_down.controller]["analog_inputs"][str(readout_line.Q_down.channel)][
            #     "offset"
            # ] = readout_line.Q_down.offset
            # add gain
            config["controllers"][readout_line.I_down.controller]["analog_inputs"][str(readout_line.I_down.channel)][
                "gain_db"
            ] = readout_line.I_down.gain_db
            # config["controllers"][readout_line.Q_down.controller]["analog_inputs"][str(readout_line.Q_down.channel)][
            #     "gain_db"
            # ] = readout_line.Q_down.gain_db

            config["waveforms"][f"readout_wf_" + state.readout_resonators[r].name] = {
                "type": "constant",
                "sample": v.readout_amplitude,
            }
            config["pulses"][f"readout_pulse_" + state.readout_resonators[r].name] = {
                "operation": "measurement",
                "length": round(readout_line.length * 1e9),
                "waveforms": {
                    "I": f"readout_wf_" + state.readout_resonators[r].name,
                    "Q": "zero_wf",
                },
                "integration_weights": {
                    "cos": "cosine_weights",
                    "sin": "sine_weights",
                    "minus_sin": "minus_sine_weights",
                    "rotated_cos": f"rotated_cosine_weights_" + state.readout_resonators[r].name,
                    "rotated_sin": f"rotated_sine_weights_" + state.readout_resonators[r].name,
                    "rotated_minus_sin": f"rotated_minus_sine_weights_" + state.readout_resonators[r].name,
                },
                "digital_marker": "ON",
            }
            rot_angle_in_pi = v.rotation_angle / 180.0 * np.pi
            config["integration_weights"]["cosine_weights"] = {
                "cosine": [(1.0, round(readout_line.length * 1e9))],
                "sine": [(0.0, round(readout_line.length * 1e9))],
            }
            config["integration_weights"]["sine_weights"] = {
                "cosine": [(0.0, round(readout_line.length * 1e9))],
                "sine": [(1.0, round(readout_line.length * 1e9))],
            }
            config["integration_weights"]["minus_sine_weights"] = {
                "cosine": [(0.0, round(readout_line.length * 1e9))],
                "sine": [(-1.0, round(readout_line.length * 1e9))],
            }
            config["integration_weights"][f"rotated_cosine_weights_" + state.readout_resonators[r].name] = {
                "cosine": [(np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))],
                "sine": [(-np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))],
            }
            config["integration_weights"][f"rotated_sine_weights_" + state.readout_resonators[r].name] = {
                "cosine": [(np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))],
                "sine": [(np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))],
            }
            config["integration_weights"][f"rotated_minus_sine_weights_" + state.readout_resonators[r].name] = {
                "cosine": [(-np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))],
                "sine": [(-np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))],
            }
            # Custom integration weights
            for weight in state.readout_resonators[r].integration_weights:
                config["integration_weights"][weight.name + "_weights_" + state.readout_resonators[r].name] = {
                    "cosine": convert_integration_weights(weight.cosine, N=int(readout_line.length * 1e9)),
                    "sine": convert_integration_weights(weight.sine, N=int(readout_line.length * 1e9)),
                }

                config["pulses"][f"readout_pulse_" + state.readout_resonators[r].name]["integration_weights"][
                    weight.name
                ] = (weight.name + "_weights_" + state.readout_resonators[r].name)
        if r in qb_wo_charge_list:
            readout_line = state.readout_lines[v.wiring.readout_line_index]

            if state.readout_resonators[r].threads.thread_cond:
                config["elements"][state.readout_resonators[r].name] = {
                    "thread": state.readout_resonators[r].threads.thread,
                    "mixInputs": {
                        "I": (
                            readout_line.I_up.controller,
                            readout_line.I_up.channel,
                        ),
                        "Q": (
                            readout_line.Q_up.controller,
                            readout_line.Q_up.channel,
                        ),
                        "lo_frequency": round(readout_line.lo_freq),
                        "mixer": f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}",
                    },
                    "intermediate_frequency": round(v.f_opt - readout_line.lo_freq),
                    "digitalInputs": {
                        "switch": {
                            "port": (state.readout_lines[0].switch.controller, state.readout_lines[0].switch.channel),
                            "delay": v.wiring.switch_delay,
                            "buffer": v.wiring.switch_buffer,
                        },
                    },
                    "operations": {
                        state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                        "readout": f"readout_pulse_" + state.readout_resonators[r].name,
                    },
                    "outputs": {
                        "out1": (
                            readout_line.I_down.controller,
                            readout_line.I_down.channel,
                        ),
                        # "out2": (
                        #     readout_line.Q_down.controller,
                        #     readout_line.Q_down.channel,
                        # ),
                    },
                    "time_of_flight": v.wiring.time_of_flight,
                    "smearing": 0,
                }
            else:
                config["elements"][state.readout_resonators[r].name] = {
                    "mixInputs": {
                        "I": (
                            readout_line.I_up.controller,
                            readout_line.I_up.channel,
                        ),
                        "Q": (
                            readout_line.Q_up.controller,
                            readout_line.Q_up.channel,
                        ),
                        "lo_frequency": round(readout_line.lo_freq),
                        "mixer": f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}",
                    },
                    "intermediate_frequency": round(v.f_opt - readout_line.lo_freq),
                    "digitalInputs": {
                        "switch": {
                            "port": (state.readout_lines[0].switch.controller, state.readout_lines[0].switch.channel),
                            "delay": v.wiring.switch_delay,
                            "buffer": v.wiring.switch_buffer,
                        },
                    },
                    "operations": {
                        state.common_operation.name: f"{state.common_operation.name}_IQ_pulse",
                        "readout": f"readout_pulse_" + state.readout_resonators[r].name,
                    },
                    "outputs": {
                        "out1": (
                            readout_line.I_down.controller,
                            readout_line.I_down.channel,
                        ),
                        # "out2": (
                        #     readout_line.Q_down.controller,
                        #     readout_line.Q_down.channel,
                        # ),
                    },
                    "time_of_flight": v.wiring.time_of_flight,
                    "smearing": 0,
                }
            # add mixers
            if f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}" not in config["mixers"]:
                config["mixers"][f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}"] = []
            config["mixers"][f"mixer_readout_line{state.readout_resonators[r].wiring.readout_line_index}"].append(
                {
                    "intermediate_frequency": round(v.f_opt - readout_line.lo_freq),
                    "lo_frequency": readout_line.lo_freq,
                    "correction": IQ_imbalance(v.wiring.correction_matrix.gain, v.wiring.correction_matrix.phase),
                }
            )
            # add offset
            config["controllers"][readout_line.I_up.controller]["analog_outputs"][str(readout_line.I_up.channel)][
                "offset"
            ] = readout_line.I_up.offset
            config["controllers"][readout_line.Q_up.controller]["analog_outputs"][str(readout_line.Q_up.channel)][
                "offset"
            ] = readout_line.Q_up.offset
            config["controllers"][readout_line.I_down.controller]["analog_inputs"][str(readout_line.I_down.channel)][
                "offset"
            ] = readout_line.I_down.offset
            # config["controllers"][readout_line.Q_down.controller]["analog_inputs"][str(readout_line.Q_down.channel)][
            #     "offset"
            # ] = readout_line.Q_down.offset
            # add gain
            config["controllers"][readout_line.I_down.controller]["analog_inputs"][str(readout_line.I_down.channel)][
                "gain_db"
            ] = readout_line.I_down.gain_db
            # config["controllers"][readout_line.Q_down.controller]["analog_inputs"][str(readout_line.Q_down.channel)][
            #     "gain_db"
            # ] = readout_line.Q_down.gain_db

            config["waveforms"][f"readout_wf_" + state.readout_resonators[r].name] = {
                "type": "constant",
                "sample": v.readout_amplitude,
            }
            config["pulses"][f"readout_pulse_" + state.readout_resonators[r].name] = {
                "operation": "measurement",
                "length": round(readout_line.length * 1e9),
                "waveforms": {
                    "I": f"readout_wf_" + state.readout_resonators[r].name,
                    "Q": "zero_wf",
                },
                "integration_weights": {
                    "cos": "cosine_weights",
                    "sin": "sine_weights",
                    "minus_sin": "minus_sine_weights",
                    "rotated_cos": f"rotated_cosine_weights_" + state.readout_resonators[r].name,
                    "rotated_sin": f"rotated_sine_weights_" + state.readout_resonators[r].name,
                    "rotated_minus_sin": f"rotated_minus_sine_weights_" + state.readout_resonators[r].name,
                },
                "digital_marker": "ON",
            }
            rot_angle_in_pi = v.rotation_angle / 180.0 * np.pi
            config["integration_weights"]["cosine_weights"] = {
                "cosine": [(1.0, round(readout_line.length * 1e9))],
                "sine": [(0.0, round(readout_line.length * 1e9))],
            }
            config["integration_weights"]["sine_weights"] = {
                "cosine": [(0.0, round(readout_line.length * 1e9))],
                "sine": [(1.0, round(readout_line.length * 1e9))],
            }
            config["integration_weights"]["minus_sine_weights"] = {
                "cosine": [(0.0, round(readout_line.length * 1e9))],
                "sine": [(-1.0, round(readout_line.length * 1e9))],
            }
            config["integration_weights"][f"rotated_cosine_weights_" + state.readout_resonators[r].name] = {
                "cosine": [(np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))],
                "sine": [(-np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))],
            }
            config["integration_weights"][f"rotated_sine_weights_" + state.readout_resonators[r].name] = {
                "cosine": [(np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))],
                "sine": [(np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))],
            }
            config["integration_weights"][f"rotated_minus_sine_weights_" + state.readout_resonators[r].name] = {
                "cosine": [(-np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))],
                "sine": [(-np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))],
            }
            # Custom integration weights
            for weight in state.readout_resonators[r].integration_weights:
                config["integration_weights"][weight.name + "_weights_" + state.readout_resonators[r].name] = {
                    "cosine": convert_integration_weights(weight.cosine, N=int(readout_line.length * 1e9)),
                    "sine": convert_integration_weights(weight.sine, N=int(readout_line.length * 1e9)),
                }

                config["pulses"][f"readout_pulse_" + state.readout_resonators[r].name]["integration_weights"][
                    weight.name
                ] = (weight.name + "_weights_" + state.readout_resonators[r].name)


def add_qb_rot(
    state,
    config: Dict,
    q: int,
    angle: int,
    direction: str,
    wf_I,
    wf_Q=None,
):
    """Add single qubit operation

    Args:
        state: state of the system
        config (Dict): OPX config we are building/editing
        q (int): index of qubit to whom we are adding operation
        angle (int): angle in degrees
        direction (str):  "x" or "y"
        wf_I: 'I' waveform
        wf_Q: Optional, 'Q' waveform
    """
    if direction not in ["x", "y"]:
        raise ValueError(f"Only x and y are accepted directions, received {direction}")
    if q >= len(state.qubits):
        raise ValueError(f"Qubit {q} is not configured in state. Please add qubit {q} first.")
    if type(angle) != int:
        raise ValueError("Only integers are accepted as angle.")

    if direction == "x":
        direction_angle = 0
    elif direction == "y":
        direction_angle = np.pi / 2
    else:
        raise ValueError(f"Direction {direction} is not valid. Only x and y are accepted as direction.")

    if wf_Q is None:
        wf_Q = np.zeros(len(wf_I))
    if len(wf_I) != len(wf_Q):
        raise ValueError("wf_I and wf_Q should have same lengths!")

    wv = np.sign(angle) * (wf_I * np.cos(direction_angle) - wf_Q * np.sin(direction_angle))
    if np.all((wv == wv[0])):
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits[q].name] = {"type": "constant"}
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits[q].name]["sample"] = wv[0]
    else:
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits[q].name] = {"type": "arbitrary"}
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits[q].name]["samples"] = wv

    wv = np.sign(angle) * (wf_I * np.sin(direction_angle) + wf_Q * np.cos(direction_angle))
    if np.all((wv == wv[0])):
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits[q].name] = {"type": "constant"}
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits[q].name]["sample"] = wv[0]
    else:
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits[q].name] = {"type": "arbitrary"}
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits[q].name]["samples"] = wv

    config["pulses"][f"{direction}{angle}_pulse_" + state.qubits[q].name] = {
        "operation": "control",
        "length": len(wf_I),
        "waveforms": {
            "I": f"{direction}{angle}_I_wf_" + state.qubits[q].name,
            "Q": f"{direction}{angle}_Q_wf_" + state.qubits[q].name,
        },
        "digital_marker": "ON",
    }
    config["elements"][state.qubits[q].name]["operations"][f"{direction}{angle}"] = (
        f"{direction}{angle}_pulse_" + state.qubits[q].name
    )


def add_qb_wo_charge_rot(
    state,
    config: Dict,
    q: int,
    angle: int,
    direction: str,
    wf_I,
    wf_Q=None,
):
    """Add single qubit operation

    Args:
        state: state of the system
        config (Dict): OPX config we are building/editing
        q (int): index of qubit to whom we are adding operation
        angle (int): angle in degrees
        direction (str):  "x" or "y"
        wf_I: 'I' waveform
        wf_Q: Optional, 'Q' waveform
    """
    if direction not in ["x", "y"]:
        raise ValueError(f"Only x and y are accepted directions, received {direction}")
    if q >= len(state.qubits_wo_charge):
        raise ValueError(f"Qubit {q} is not configured in state. Please add qubit {q} first.")
    if type(angle) != int:
        raise ValueError("Only integers are accepted as angle.")

    if direction == "x":
        direction_angle = 0
    elif direction == "y":
        direction_angle = np.pi / 2
    else:
        raise ValueError(f"Direction {direction} is not valid. Only x and y are accepted as direction.")

    if wf_Q is None:
        wf_Q = np.zeros(len(wf_I))
    if len(wf_I) != len(wf_Q):
        raise ValueError("wf_I and wf_Q should have same lengths!")

    wv = np.sign(angle) * (wf_I * np.cos(direction_angle) - wf_Q * np.sin(direction_angle))
    if np.all((wv == wv[0])):
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits_wo_charge[q].name] = {"type": "constant"}
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits_wo_charge[q].name]["sample"] = wv[0]
    else:
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits_wo_charge[q].name] = {"type": "arbitrary"}
        config["waveforms"][f"{direction}{angle}_I_wf_" + state.qubits_wo_charge[q].name]["samples"] = wv

    wv = np.sign(angle) * (wf_I * np.sin(direction_angle) + wf_Q * np.cos(direction_angle))
    if np.all((wv == wv[0])):
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits_wo_charge[q].name] = {"type": "constant"}
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits_wo_charge[q].name]["sample"] = wv[0]
    else:
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits_wo_charge[q].name] = {"type": "arbitrary"}
        config["waveforms"][f"{direction}{angle}_Q_wf_" + state.qubits_wo_charge[q].name]["samples"] = wv

    config["pulses"][f"{direction}{angle}_pulse_" + state.qubits_wo_charge[q].name] = {
        "operation": "control",
        "length": len(wf_I),
        "waveforms": {
            "I": f"{direction}{angle}_I_wf_" + state.qubits_wo_charge[q].name,
            "Q": f"{direction}{angle}_Q_wf_" + state.qubits_wo_charge[q].name,
        },
        "digital_marker": "ON",
    }
    config["elements"][state.qubits_wo_charge[q].name]["operations"][f"{direction}{angle}"] = (
        f"{direction}{angle}_pulse_" + state.qubits_wo_charge[q].name
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


def add_controllers(state: QuAM, config, d_outputs: list, qb_list: list, qb_wo_charge_list: list, injector_list: list):
    for con in state.controllers:
        config["controllers"][con] = {}
        config["controllers"][con]["analog_outputs"] = {}
        config["controllers"][con]["analog_inputs"] = {}
        config["controllers"][con]["digital_outputs"] = {}
        # Add digital output channels
        for i in d_outputs:
            config["controllers"][con]["digital_outputs"][str(i)] = {}
        # Add qubit and readout resonator channels
        for i in qb_list:
            # Add qubit channels
            wiring = state.qubits[i].wiring
            config["controllers"][con]["analog_outputs"][str(state.drive_lines[wiring.drive_line_index].I.channel)] = {
                "offset": 0.0
            }
            config["controllers"][con]["analog_outputs"][str(state.drive_lines[wiring.drive_line_index].Q.channel)] = {
                "offset": 0.0
            }
            config["controllers"][con]["analog_outputs"][str(wiring.charge_line.channel)] = {
                "offset": state.qubits[i].wiring.analog_channel_offset
            }
            # Add resonator channels
            readout_line = state.readout_lines[state.readout_resonators[i].wiring.readout_line_index]
            config["controllers"][con]["analog_inputs"][str(readout_line.I_down.channel)] = {
                "offset": 0.0,
                "gain_db": 0,
            }
            # config["controllers"][con]["analog_inputs"][str(readout_line.Q_down.channel)] = {
            #     "offset": 0.0,
            #     "gain_db": 0,
            # }
            config["controllers"][con]["analog_outputs"][str(readout_line.I_up.channel)] = {"offset": 0.0}
            config["controllers"][con]["analog_outputs"][str(readout_line.Q_up.channel)] = {"offset": 0.0}
        for i in qb_wo_charge_list:
            # Add qubit channels
            wiring = state.qubits_wo_charge[i - NUMBER_OF_QUBITS_W_CHARGE].wiring
            config["controllers"][con]["analog_outputs"][str(state.drive_lines[wiring.drive_line_index].I.channel)] = {
                "offset": 0.0
            }
            config["controllers"][con]["analog_outputs"][str(state.drive_lines[wiring.drive_line_index].Q.channel)] = {
                "offset": 0.0
            }
            # config["controllers"][con]["analog_outputs"][str(wiring.charge_line.channel)] = {"offset": 0.0}
            # Add resonator channels
            readout_line = state.readout_lines[state.readout_resonators[i].wiring.readout_line_index]
            config["controllers"][con]["analog_inputs"][str(readout_line.I_down.channel)] = {
                "offset": 0.0,
                "gain_db": 0,
            }
            # config["controllers"][con]["analog_inputs"][str(readout_line.Q_down.channel)] = {
            #     "offset": 0.0,
            #     "gain_db": 0,
            # }
            config["controllers"][con]["analog_outputs"][str(readout_line.I_up.channel)] = {"offset": 0.0}
            config["controllers"][con]["analog_outputs"][str(readout_line.Q_up.channel)] = {"offset": 0.0}
        for i in injector_list:
            # Add injector channels
            wiring = state.qp_injectors[i].wiring
            config["controllers"][con]["analog_outputs"][str(wiring.injector_line.channel)] = {
                "offset": state.qp_injectors[i].analog_channel_offset
            }


def add_digital_waveforms(state: QuAM, config):
    for wf in state.digital_waveforms:
        config["digital_waveforms"][wf.name] = {"samples": wf.samples}


def add_common_operation(state: QuAM, config: dict):
    config["waveforms"]["const_wf"] = {
        "type": "constant",
        "sample": state.common_operation.amplitude,
    }
    config["waveforms"]["zero_wf"] = {
        "type": "constant",
        "sample": 0.0,
    }
    config["pulses"][f"{state.common_operation.name}_IQ_pulse"] = {
        "operation": "control",
        "length": round(state.common_operation.duration * 1e9),
        "waveforms": {
            "I": "const_wf",
            "Q": "zero_wf",
        },
        "digital_marker": "ON",
    }
    config["pulses"][f"{state.common_operation.name}_single_pulse"] = {
        "operation": "control",
        "length": round(state.common_operation.duration * 1e9),
        "waveforms": {
            "single": "const_wf",
        },
        "digital_marker": "ON",
    }


def add_qp_injector(state: QuAM, config: Dict, injector_list: list):
    for i in injector_list:
        wiring = state.qp_injectors[i].wiring
        # add injector element
        config["elements"][state.qp_injectors[i].name] = {
            "singleInput": {
                "port": (
                    wiring.injector_line.controller,
                    wiring.injector_line.channel,
                )
            },
            "operations": {
                state.common_operation.name: f"{state.common_operation.name}_single_pulse",
                "injector": f"{state.qp_injectors[i].name}_single_pulse",
            },
        }

        # add injector sticky element
        config["elements"][state.qp_injectors[i].name + "_sticky"] = {
            "singleInput": {
                "port": (
                    wiring.injector_line.controller,
                    wiring.injector_line.channel,
                )
            },
            "operations": {
                state.common_operation.name: f"{state.common_operation.name}_single_pulse",
                "injector": f"{state.qp_injectors[i].name}_single_pulse",
            },
            "hold_offset": {"duration": 1},
        }

        # add operation and pulse
        config["pulses"][state.qp_injectors[i].name + f"_single_pulse"] = {
            "operation": "control",
            "length": state.qp_injectors[i].injection_length * 1e9,
            "waveforms": {"single": state.qp_injectors[i].name + "_wf"},
        }
        config["waveforms"][state.qp_injectors[i].name + "_wf"] = {
            "type": "constant",
            "sample": state.qp_injectors[i].injection_voltage,
        }


def build_config(state, digital_out: list, qubits: list, qubits_wo_charge: list, injector_list: list, shape: str):
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

    add_controllers(
        state,
        config,
        d_outputs=digital_out,
        qb_list=qubits,
        qb_wo_charge_list=qubits_wo_charge,
        injector_list=injector_list,
    )

    add_common_operation(state, config)

    add_digital_waveforms(state, config)

    add_qubits(state, config, qb_list=qubits)

    add_qubits_wo_charge(state, config, qb_wo_charge_list=qubits_wo_charge)

    add_readout_resonators(state, config, qb_list=qubits, qb_wo_charge_list=qubits_wo_charge)

    add_qp_injector(state, config, injector_list=injector_list)

    add_mixers(state, config, qb_list=qubits)

    for single_qubit_operation in state.single_qubit_operations:
        for q in qubits:
            for z in state.qubits[q].driving.__dict__.get("_schema").get("required"):
                if z == shape:
                    pulse = get_qubit_gate(state, q, z)
                    if pulse.shape == "gaussian":
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_rot(
                            state,
                            config,
                            q,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            amplitude
                            * gaussian(
                                round(pulse.length * 1e9),
                                round(pulse.sigma * 1e9),
                            ),
                        )  # +180 and -180 have same amplitude
                    elif pulse.shape == "square":
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_rot(
                            state,
                            config,
                            q,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            np.array([amplitude] * round(pulse.length * 1e9)),
                        )  # +180 and -180 have same amplitude
                    elif pulse.shape == "drag_gaussian":
                        from qualang_tools.config.waveform_tools import (
                            drag_gaussian_pulse_waveforms,
                        )

                        drag_I, drag_Q = np.array(
                            drag_gaussian_pulse_waveforms(
                                1,
                                round(pulse.length * 1e9),
                                round(pulse.sigma * 1e9),
                                alpha=pulse.alpha,
                                detuning=pulse.detuning,
                                anharmonicity=state.qubits[q].anharmonicity,
                            )
                        )
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_rot(
                            state,
                            config,
                            q,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            amplitude * drag_I,
                            amplitude * drag_Q,
                        )  # +180 and -180 have same amplitude
                    elif pulse.shape == "drag_cosine":
                        from qualang_tools.config.waveform_tools import (
                            drag_cosine_pulse_waveforms,
                        )

                        drag_I, drag_Q = np.array(
                            drag_cosine_pulse_waveforms(
                                1,
                                round(pulse.length * 1e9),
                                alpha=pulse.alpha,
                                detuning=pulse.detuning,
                                anharmonicity=state.qubits[q].anharmonicity,
                            )
                        )
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_rot(
                            state,
                            config,
                            q,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            amplitude * drag_I,
                            amplitude * drag_Q,
                        )  # +180 and -180 have same amplitude
                    else:
                        raise ValueError(f"Gate shape {pulse.shape} not recognized.")
        for q in qubits_wo_charge:
            for z in (
                state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].driving.__dict__.get("_schema").get("required")
            ):
                if z == shape:
                    pulse = get_qubit_wo_charge_gate(state, q - NUMBER_OF_QUBITS_W_CHARGE, z)
                    if pulse.shape == "gaussian":
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_wo_charge_rot(
                            state,
                            config,
                            q - NUMBER_OF_QUBITS_W_CHARGE,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            amplitude
                            * gaussian(
                                round(pulse.length * 1e9),
                                round(pulse.sigma * 1e9),
                            ),
                        )  # +180 and -180 have same amplitude
                    elif pulse.shape == "square":
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_wo_charge_rot(
                            state,
                            config,
                            q - NUMBER_OF_QUBITS_W_CHARGE,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            np.array([amplitude] * round(pulse.length * 1e9)),
                        )  # +180 and -180 have same amplitude
                    elif pulse.shape == "drag_gaussian":
                        from qualang_tools.config.waveform_tools import (
                            drag_gaussian_pulse_waveforms,
                        )

                        drag_I, drag_Q = np.array(
                            drag_gaussian_pulse_waveforms(
                                1,
                                round(pulse.length * 1e9),
                                round(pulse.sigma * 1e9),
                                alpha=pulse.alpha,
                                detuning=pulse.detuning,
                                anharmonicity=state.qubits[q].anharmonicity,
                            )
                        )
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_wo_charge_rot(
                            state,
                            config,
                            q - NUMBER_OF_QUBITS_W_CHARGE,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            amplitude * drag_I,
                            amplitude * drag_Q,
                        )  # +180 and -180 have same amplitude
                    elif pulse.shape == "drag_cosine":
                        from qualang_tools.config.waveform_tools import (
                            drag_cosine_pulse_waveforms,
                        )

                        drag_I, drag_Q = np.array(
                            drag_cosine_pulse_waveforms(
                                1,
                                round(pulse.length * 1e9),
                                alpha=pulse.alpha,
                                detuning=pulse.detuning,
                                anharmonicity=state.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].anharmonicity,
                            )
                        )
                        if abs(single_qubit_operation.angle) == 180:
                            amplitude = pulse.angle2volt.deg180
                        elif abs(single_qubit_operation.angle) == 90:
                            amplitude = pulse.angle2volt.deg90
                        else:
                            raise ValueError(
                                "Unknown angle for single qubit operation" f" {single_qubit_operation.angle}"
                            )
                        add_qb_wo_charge_rot(
                            state,
                            config,
                            q - NUMBER_OF_QUBITS_W_CHARGE,
                            single_qubit_operation.angle,
                            single_qubit_operation.direction,
                            amplitude * drag_I,
                            amplitude * drag_Q,
                        )  # +180 and -180 have same amplitude
                    else:
                        raise ValueError(f"Gate shape {pulse.shape} not recognized.")

    return config


def save(state: QuAM, filename: str, reuse_existing_values: bool = False):
    """Saves quam data to file

    Args:
        filename (str): destination file name
        reuse_existing_values (bool, optional): if destination file exists, it will try
        to reuse key values from that file. Defaults to False.
    """

    if reuse_existing_values:
        if os.path.isfile(filename):
            with open(filename, "r") as file:
                old_values = json.load(file)

            for key, value in old_values.items():
                if key in state._json and (type(value) is not list or len(value) == len(state._json[key])):
                    state._json[key] = value

    with open(filename, "w") as file:
        json.dump(state._json, file)


## API
def get_wiring(state: QuAM):
    """
    Print the state connectivity.
    """
    s = " " * 40 + "STATE WIRING\n"
    s += "-" * 110 + "\n"
    for d in range(len(state.drive_lines)):
        s += f"drive line {d} connected to channel {state.drive_lines[d].I.channel} (I) and {state.drive_lines[d].Q.channel} (Q) of controller '{state.drive_lines[d].I.controller}' "
        s += f"with qubits "
        qq = []
        for q in range(len(state.qubits)):
            if d == state.qubits[q].wiring.drive_line_index:
                qq.append(q)
        for q in range(len(state.qubits_wo_charge)):
            if d == state.qubits_wo_charge[q].wiring.drive_line_index:
                qq.append(q + NUMBER_OF_QUBITS_W_CHARGE)
        s += str(qq) + "\n"
    s += "-" * 110 + "\n"
    for r in range(len(state.readout_lines)):
        s += f"readout line {r} connected to channel {state.readout_lines[r].I_up.channel} (I) and {state.readout_lines[r].Q_up.channel} (Q) of controller '{state.readout_lines[r].I_up.controller}' "
        s += f"with readout resonators "
        rrr = []
        for rr in range(len(state.readout_resonators)):
            if r == state.readout_resonators[rr].wiring.readout_line_index:
                rrr.append(rr)
        s += str(rrr) + "\n"
    s += "-" * 110 + "\n"
    for q in range(len(state.qubits)):
        s += f"charge line {q} connected to channel {state.qubits[q].wiring.charge_line.channel} of controller '{state.qubits[q].wiring.charge_line.controller}'\n"
    print(s)


def get_sequence_state(state: QuAM, index: int, sequence_state: str = None):
    """
    Get the sequence state object of a given qubit.

    :param index: index of the qubit to be retrieved.
    :param sequence_state: name of the sequence.
    :return: the sequence state object. Print the list of sequence states if 'sequence_state' is None.
    """

    seq = {"constant": [], "arbitrary": []}
    for key in state.qubits[index].sequence_states.constant:
        seq["constant"].append(key.name)
    for key in state.qubits[index].sequence_states.arbitrary:
        seq["arbitrary"].append(key.name)

    if sequence_state is None:
        pprint(seq)
        return

    for seq in state.qubits[index].sequence_states.arbitrary:
        if seq.name == sequence_state:
            return seq
    for seq in state.qubits[index].sequence_states.constant:
        if seq.name == sequence_state:
            return seq
    raise ValueError(
        f"The sequence state '{sequence_state}' is not defined in the state. The available sequence states are {seq}."
    )


def get_charge_bias_point(state: QuAM, index: int, charge_bias_point: str = None):
    """
    Get the charge bias point for a given qubit.

    :param index: index of the qubit to be retrieved.
    :param charge_bias_point: name of the charge bias point.
    :return: charge bias point object. Print the list of charge bias point if 'charge_bias_point' is None.
    """
    points = []
    for key in state.qubits[index].charge_bias_points:
        points.append(key.name)
    if charge_bias_point is None:
        qprint(state.qubits[index].charge_bias_points)
        return
    for bias in state.qubits[index].charge_bias_points:
        if bias.name == charge_bias_point:
            return bias
    raise ValueError(
        f"The charge_bias_point '{charge_bias_point}' is not defined in the state for qubit {index}. The available charge bias points are {points}."
    )


def get_qubit(state: QuAM, qubit_name: str):
    """
    Get the qubit object corresponding to the specified qubit name.

    :param qubit_name: name of the qubit to get.
    :return: the qubit object.
    """
    for q in range(len(state.qubits)):
        if state.qubits[q].name == qubit_name:
            return state.qubits[q]
    raise ValueError(f"The qubit '{qubit_name}' is not defined in the state.")


def get_resonator(state: QuAM, resonator_name: str):
    """
    Get the readout resonator object corresponding to the specified resonator name.

    :param resonator_name: name of the qubit to get.
    :return: the qubit object.
    """
    for q in range(len(state.qubits)):
        if state.readout_resonators[q].name == resonator_name:
            return state.qubits[q]
    raise ValueError(f"The readout resonator '{resonator_name}' is not defined in the state.")


def get_qubit_gate(state: QuAM, index: int, shape: str):
    """
    Get the gate of a given qubit from its shape.

    :param index: index of the qubit to be retrieved.
    :param shape: name of the gate as defined under the qubit driving section.
    :return: the qubit gate object.
    """
    for z in state.qubits[index].driving.__dict__.get("_schema").get("required"):
        if z == shape:
            return state.qubits[index].driving.__getattribute__(shape)
    raise ValueError(
        f"The gate '{shape}' is not defined in the state. The available gates are {state.qubits[index].driving.__dict__.get('_schema').get('required')}"
    )


def get_qubit_wo_charge_gate(state: QuAM, index: int, shape: str):
    """
    Get the gate of a given qubit from its shape.

    :param index: index of the qubit to be retrieved.
    :param shape: name of the gate as defined under the qubit driving section.
    :return: the qubit gate object.
    """
    for z in state.qubits_wo_charge[index].driving.__dict__.get("_schema").get("required"):
        if z == shape:
            return state.qubits_wo_charge[index].driving.__getattribute__(shape)
    raise ValueError(
        f"The gate '{shape}' is not defined in the state. The available gates are {state.qubits_wo_charge[index].driving.__dict__.get('_schema').get('required')}"
    )


def get_readout_IF(state: QuAM, index: int) -> float:
    """
    Get the intermediate frequency of the readout resonator specified by its index.

    :param index: index of the readout resonator to be retrieved.
    :return: the intermediate frequency in Hz.
    """
    return (
        state.readout_resonators[index].f_opt
        - state.readout_lines[state.readout_resonators[index].wiring.readout_line_index].lo_freq
    )


def get_qubit_IF(state: QuAM, index: int) -> float:
    """
    Get the intermediate frequency of the qubit specified by its index.

    :param index: index of the qubit to be retrieved.
    :return: the intermediate frequency in Hz.
    """
    if index < NUMBER_OF_QUBITS_W_CHARGE:
        return state.qubits[index].f_01 - state.drive_lines[state.qubits[index].wiring.drive_line_index].lo_freq
    else:
        return state.qubits_wo_charge[index - NUMBER_OF_QUBITS_W_CHARGE].f_01 - state.drive_lines[state.qubits_wo_charge[index - NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index].lo_freq


def _calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    return A, B, C


def set_f_res_vs_charge_vertex(state: QuAM, index: int, three_points: List[tuple]):
    """
    Set the vertex corresponding to the resonator frequency vs charge parabola from three points.

    :param index: index of the readout resonator to be retrieved.
    :param three_points: list of three tuples corresponding to the three points located on the parabola to fit defined as [(x1, y1), (x2, y2), (x3, y3)].
    """
    a, b, c = _calc_parabola_vertex(
        three_points[0][0],
        three_points[0][1],
        three_points[1][0],
        three_points[1][1],
        three_points[2][0],
        three_points[2][1],
    )
    state.readout_resonators[index].f_res_vs_charge.a = a
    state.readout_resonators[index].f_res_vs_charge.b = b
    state.readout_resonators[index].f_res_vs_charge.c = c


def get_f_res_from_charge(state: QuAM, index: int, charge_bias: float) -> float:
    """
    Get the resonance frequency of the specified readout resonator for a given charge bias.
    The vertex of the resonator frequency vs charge parabola must be set beforehand.

    :param index: index of the readout resonator to retrieve.
    :param charge_bias: value of the charge bias at which the resonance frequency will be derived.
    :return: the readout resonator resonance frequency corresponding to the specified charge bias.
    """
    vertex = state.readout_resonators[index].f_res_vs_charge
    return vertex.a * charge_bias**2 + vertex.b * charge_bias + vertex.c


def save_results(state: QuAM, filename, figures: List = ()):
    now = datetime.now()
    now = now.strftime("%m%d%Y_%H%M%S")
    directory = state.results.directory + f"{now[:8]}/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # Save json
    state.save(directory + f"{now[9:]}_state_after_{filename}.json")
    # Save figure
    for i in range(len(figures)):
        if len(figures) > 1:
            figures[i].savefig(directory + f"{now[9:]}_{filename}" + f"_fig{i}")
        else:
            figures[i].savefig(directory + f"{now[9:]}_{filename}")
    # Save data
    # tbd


if __name__ == "__main__":
    # if we execute directly config.py this tests that configuration is ok
    machine = QuAM("quam_bootstrap_state.json")
    qubit_list = [0, 1]
    qubit_wo_charge_list = [2, 3, 4, 5]
    injector_list = [0]
    digital = []
    configuration = build_config(machine, digital, qubit_list, qubit_wo_charge_list, injector_list, shape="drag_cosine")
    qprint(machine)
    machine.get_wiring()
