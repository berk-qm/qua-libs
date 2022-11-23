import itertools

import numpy as np
from itertools import chain


# readout_pulse_length = 1000
#
# _config = {
#     "version": 1,
#     "controllers": {
#         "con1": {
#             "type": "opx1",
#             "analog_outputs": {
#                 1: {"offset": +0.0},
#                 2: {"offset": +0.0},
#                 3: {"offset": +0.0},
#             },
#             'analog_inputs': {
#                 1: {'offset': 0.0},
#                 2: {'offset': 0.0},
#             }
#         }
#     },
#     "elements": {
#         "lockin": {
#             "singleInput": {"port": ("con1", 1)},
#             "intermediate_frequency": lockin_freq,
#             "operations": {
#                 "CW": "constPulse",
#                 'readout': 'readout_pulse'
#             },
#             'outputs': {'out1': ('con1', 1)},
#             'time_of_flight': 184,
#             'smearing': 0,
#         },
#         "qe": {
#             "singleInput": {"port": ("con1", 1)},
#             "intermediate_frequency": lockin_freq,
#             "operations": {
#                 "CW": "constPulse",
#                 'readout': 'readout_pulse'
#             },
#             'hold_offset': {'duration': 1},
#             'outputs': {'out1': ('con1', 1)},
#             'time_of_flight': 184,
#             'smearing': 0,
#         },
#     },
#     "pulses": {
#         "constPulse": {
#             "operation": "control",
#             "length": 16,  # in ns
#             "waveforms": {"single": "const_wf"},
#         },
#         "readout_pulse": {
#             'operation': 'measurement',
#             'length': readout_pulse_length,
#             'waveforms': {
#                 'single': 'zero_wf',
#             },
#             'integration_weights': {
#                 'integ_weights_cos': 'integW_cosine',
#                 'integ_weights_sin': 'integW_sine',
#             },
#             'digital_marker': 'ON',
#         },
#     },
#
#     "waveforms": {
#         "const_wf": {"type": "constant", "sample": 0.2},
#         'zero_wf': {"type": "constant", "sample": 0.3}
#     },
#     'digital_waveforms': {
#         'ON': {
#             'samples': [(1, 0)]
#         }
#     },
#     'integration_weights': {
#         'integW_cosine': {
#             'cosine': [1.0] * int(readout_pulse_length / 4),  #[(1.0, readout_pulse_length),]
#             'sine': [0.0] * int(readout_pulse_length / 4),
#         },
#         'integW_sine': {
#             'cosine': [0.0] * int(readout_pulse_length / 4),
#             'sine': [1.0] * int(readout_pulse_length / 4),
#         },
#     },
# }


def drag_pulse(N: int, amp, alpha: float, anharomonicity_factor: float):
    """
    the IQ samples of a drag pulse, as defined in Zijun Chen thesis.
    Amplitude is 0.5 - 2^-16
    todo: add AC stark shift
    todo: verify anharmonicity factor sign and definition
    :param N: number of samples
    :param alpha: DRAG alpha coefficient
    :param anharomonicity_factor: the factor (relative to 1) of how much there is anharmonicity in the transmon
    """
    t = np.linspace(0, N - 1, N)
    sig_i = amp * ((0.5 - 2 ** -16) * 0.5 * (1 - np.cos(2 * np.pi * t / (N - 1))))
    sig_q = amp * (-(0.5 - 2 ** -16) * 0.5 * - (alpha / (anharomonicity_factor * (N - 1))) * np.sin(
        2 * np.pi * t / (N - 1)))
    return np.array([sig_i, sig_q])


def constant_pulse(N: int, val: float):
    return val * np.ones(N)


c1_phxz = {"c1_0": (0, 1, 0), "c1_1": (0, -0.5, 0), "c1_2": (0.5, -0.5, 1), "c1_3": (0.5, -0.5, -0.5),
           "c1_4": (0, 0.5, 0.5), "c1_5": (0, 0, 0.5)}

# s1_phxz = {"s1_0": (0,0,0), "s1_1": (0,0.5,0.5), "s1_2": (0,0.5,0.5)} #ORIG
s1_phxz = {"s1_0": (0, 1, 0), "s1_1": (0, 0.5, 0.5), "s1_2": (0, 0.5, 0.5)}

cnot_phxz = {"cnot_0_0": (0.5, 0.5, -1), "cnot_0_1": (0, 0, 1),
             "cnot_2_0": (0, 1, 0), "cnot_2_1": (0, 0, 0),
             "cnot_4_0": (0.5, 0.5, 0.5), "cnot_4_1": (-1, 0.5, 1)
             }

swap_phxz = {'swap_0_0': (0, 0.5, 0.5), 'swap_0_1': (0.5, 0.5, 0),
             'swap_2_0': (0, 0.5, 0.5), 'swap_2_1': (0.5, 0.5, 0),
             'swap_4_0': (0, 0.5, 0.5), 'swap_4_1': (0.5, 0.5, 0),
             'swap_6_0': (0, 0, -0.5), 'swap_6_1': (0, 1, -0.5)
             }

# pauli_phxz = {"pauli_0": (0,0,0), "pauli_1": (1,0,0), "pauli_2": (0,1,0), "pauli_3": (1,0,0.5)} #ORIG
pauli_phxz = {"pauli_0": (0, 0.7, 0), "pauli_1": (1, 0.7, 0), "pauli_2": (0, 0.7, 0), "pauli_3": (1, 0.7, 0.5)}


class ConfigBuilder:
    PI_PULSE_LEN = 20

    def __init__(self, pi_pulse_len=PI_PULSE_LEN, cz_pulse_len=16, pulsers_per_qubit=3):
        self.pi_pulse_len = pi_pulse_len
        self.cz_pulse_len = cz_pulse_len
        self.config = {}
        self.readout_len = 256
        self.pulsers_per_qubit = pulsers_per_qubit
        self.combine_pulsers_of_qubits = True
        return

    def build(self):
        self._place_template()
        self.bake_c1()
        self.bake_cnot()
        self.bake_swap()
        self.bake_iswap()
        return

    def tuple_to_concrete_pulse(self, t, prior_z_phase=None):
        if t[1] != 0:
            p = drag_pulse(self.pi_pulse_len, t[1], 0, 1)
        else:
            p = np.zeros(shape=(2, self.pi_pulse_len))
        z_factor = 1 if prior_z_phase is None else prior_z_phase
        p[0] = p[0] * np.cos(t[0] * np.pi * z_factor) + p[1] * np.sin(t[0] * np.pi * z_factor)
        p[1] = p[0] * -np.sin(t[0] * np.pi * z_factor) + p[1] * np.cos(t[0] * np.pi * z_factor)
        p[np.where(p == 0)] = 0.0
        return p

    def bake_gates(self, core_gate_num_pi_pulses, core_gate_num_coupler, name, has_s1: bool):
        gate_core_length = core_gate_num_pi_pulses * self.pi_pulse_len + core_gate_num_coupler * self.cz_pulse_len
        gate_total_length = gate_core_length + 2 * self.pi_pulse_len + has_s1 * self.pi_pulse_len  # c1 + s1 + pauli
        if core_gate_num_pi_pulses > 0:
            gate_core_moment_lengths = np.cumsum(
                [0, self.pi_pulse_len, self.cz_pulse_len, self.pi_pulse_len, self.cz_pulse_len, self.pi_pulse_len] +
                ([self.cz_pulse_len, self.pi_pulse_len] if name == "swap" else []))
        elif core_gate_num_coupler > 0:
            gate_core_moment_lengths = np.cumsum([0, self.cz_pulse_len, self.cz_pulse_len])
        else:
            gate_core_moment_lengths = []

        gate_core_q0_xy_wf = np.zeros(shape=(2, gate_core_length))
        gate_core_q1_xy_wf = np.zeros(shape=(2, gate_core_length))
        gate_core_coupler_wf = np.zeros(shape=(gate_core_length))

        coupler_wf = np.zeros(shape=(gate_total_length))

        cumulative_phase = [0, 0]
        if name == "iswap":
            r1, r2 = [], [0, 1]
        elif name == "c1":
            r1, r2 = [], []
        else:
            r1, r2 = range(0, 2 * core_gate_num_pi_pulses, 2), range(1, 2 * core_gate_num_coupler, 2)

        gate_phxz = {"c1": {}, "cnot": cnot_phxz, "iswap": {}, "swap": swap_phxz}[name]
        for mom in r1:
            phase = 0
            gate_core_q0_xy_wf[:, gate_core_moment_lengths[mom]:gate_core_moment_lengths[mom + 1]] = \
                self.tuple_to_concrete_pulse(gate_phxz[f"{name}_{mom}_0"], prior_z_phase=phase)
            gate_core_q1_xy_wf[:,
            gate_core_moment_lengths[mom]:gate_core_moment_lengths[mom + 1]] = self.tuple_to_concrete_pulse(
                gate_phxz[f"{name}_{mom}_1"],
                prior_z_phase=phase)
            cumulative_phase[0] *= phase
            cumulative_phase[0] *= phase

        for mom in r2:
            gate_core_coupler_wf[gate_core_moment_lengths[mom]: gate_core_moment_lengths[mom + 1]] = constant_pulse(
                self.cz_pulse_len, 0.2)

        coupler_wf[self.pi_pulse_len: self.pi_pulse_len + gate_core_length] = gate_core_coupler_wf

        for c1, s1, pauli in itertools.product(range(6), range(3 if has_s1 else 1), range(4)):
            c1_wf = self.tuple_to_concrete_pulse(c1_phxz[f"c1_{c1}"])
            s1_wf = self.tuple_to_concrete_pulse(s1_phxz[f"s1_{s1}"]) if has_s1 else np.zeros(shape=(2, 0))
            pauli_wf = self.tuple_to_concrete_pulse(pauli_phxz[f"pauli_{pauli}"])
            gate_q0_wf = np.concatenate((c1_wf, gate_core_q0_xy_wf, s1_wf, pauli_wf), axis=1)
            gate_q1_wf = np.concatenate((c1_wf, gate_core_q1_xy_wf, s1_wf, pauli_wf), axis=1)

            s1_str = f"_s1_{s1}" if has_s1 else ""
            full_name = f"{name}_c1_{c1}{s1_str}_p_{pauli}"
            self.config["waveforms"][f"{full_name}_xy_q0_i"] = {"type": "arbitrary", 'samples': gate_q0_wf[0]}
            self.config["waveforms"][f"{full_name}_xy_q0_q"] = {"type": "arbitrary", 'samples': gate_q0_wf[1]}
            self.config["waveforms"][f"{full_name}_xy_q1_i"] = {"type": "arbitrary", 'samples': gate_q1_wf[0]}
            self.config["waveforms"][f"{full_name}_xy_q1_q"] = {"type": "arbitrary", 'samples': gate_q1_wf[1]}
            # Waveform of coupler
            self.config["waveforms"][f"{full_name}_coupler"] = {"type": "arbitrary", 'samples': coupler_wf}

            self.config["pulses"][f"{full_name}_pulse_xy_q0"] = {"operation": "control",
                                                                 "length": gate_total_length,
                                                                 "waveforms": {"I": f"{full_name}_xy_q0_i",
                                                                               "Q": f"{full_name}_xy_q0_q"}}
            self.config["pulses"][f"{full_name}_pulse_xy_q1"] = {"operation": "control",
                                                                 "length": gate_total_length,
                                                                 "waveforms": {"I": f"{full_name}_xy_q1_i",
                                                                               "Q": f"{full_name}_xy_q1_q"}}
            self.config["pulses"][f"{full_name}_pulse_coupler"] = {"operation": "control",
                                                                   "length": gate_total_length,
                                                                   "waveforms": {"single": f"{full_name}_coupler"}}

            self.config["elements"][f"qubit0_xy"]["operations"][f"{full_name}"] = f"{full_name}_pulse_xy_q0"
            self.config["elements"][f"qubit1_xy"]["operations"][f"{full_name}"] = f"{full_name}_pulse_xy_q1"
            self.config["elements"][f"qubit0_z"]["operations"][f"{full_name}"] = f"{full_name}_pulse_coupler"
            self.config["elements"][f"qubit1_z"]["operations"][f"{full_name}"] = f"{full_name}_pulse_coupler"
            self.config["elements"][f"coupler"]["operations"][f"{full_name}"] = f"{full_name}_pulse_coupler"

    def bake_c1(self):
        self.bake_gates(0, 0, "c1", False)

    def bake_cnot(self):
        self.bake_gates(3, 2, "cnot", True)

    def bake_iswap(self):
        self.bake_gates(0, 2, "iswap", True)

    def bake_swap(self):
        self.bake_gates(4, 3, "swap", False)
        pass

    def _place_template(self):
        num_analog_outputs = self.pulsers_per_qubit * 2 * 2 + 3 if not self.combine_pulsers_of_qubits else 7
        elements_dict = {f"qubit{i // self.pulsers_per_qubit}_xy": {
            "mixInputs": {"I": (
                "con1", i * 2 + 1 if not self.combine_pulsers_of_qubits else 2 * (i // self.pulsers_per_qubit) + 1),
                "Q": ("con1", i * 2 + 2 if not self.combine_pulsers_of_qubits else 2 * (
                        i // self.pulsers_per_qubit) + 2)},
            "intermediate_frequency": 0,
            "operations": {
                "pi": "pi_pulse",
                "readout_pulse": "readout_pulse",
            },
            'outputs': {f'out{i // self.pulsers_per_qubit + 1}': ('con1', i // self.pulsers_per_qubit + 1)},
            'time_of_flight': 184,
            'smearing': 0,
        } for i in range(self.pulsers_per_qubit * 2)}
        elements_dict.update({elem: {
            "singleInput": {"port": ("con1", port)},
            "intermediate_frequency": 0,
            "operations": {
                "coupler_tone": "coupler_pulse"
            },
        } for elem, port in
            zip(["qubit0_z", "qubit1_z", "coupler"], range(num_analog_outputs - 2, num_analog_outputs + 1))},
        )

        self.config = {
            "version": 1,
            "controllers": {
                "con1": {
                    "type": "opxp",
                    "analog_outputs": {i: {'offset': +0.0} for i in range(1, num_analog_outputs + 1)},
                    'analog_inputs': {
                        1: {'offset': 0.0},
                        2: {'offset': 0.0},
                    }
                }
            },
            "elements":
                elements_dict,
            "pulses": {
                "pi_pulse": {
                    "operation": "control",
                    "length": self.pi_pulse_len,
                    "waveforms": {"I": "pi_i", "Q": "pi_q"},
                },
                "coupler_pulse": {
                    "operation": "control",
                    "length": self.cz_pulse_len,
                    "waveforms": {"single": "const_wf"},
                },
                "readout_pulse": {
                    'operation': 'measurement',
                    'length': self.readout_len,
                    'waveforms': {
                        'I': 'zero_wf',
                        'Q': 'zero_wf'
                    },
                    'integration_weights': {
                        'integ_weights_cos': 'integW_cosine',
                        'integ_weights_sin': 'integW_sine',
                    },
                    'digital_marker': 'ON',
                },
            },
            "waveforms": {
                "pi_i": {
                    "type": "arbitrary",
                    "samples": drag_pulse(self.pi_pulse_len, 1, 0, 1)[0].tolist(),
                },
                "pi_q": {
                    "type": "arbitrary",
                    "samples": drag_pulse(self.pi_pulse_len, 1, 0, 1)[1].tolist(),
                },
                "const_wf": {
                    "type": "constant", "sample": 0.2
                },
                'zero_wf': {
                    "type": "constant", "sample": 0.3}
            },
            'digital_waveforms': {
                'ON': {
                    'samples': [(1, 0)]
                }
            },
            'integration_weights': {
                'integW_cosine': {
                    'cosine': [1.0] * int(self.readout_len),  # [(1.0, readout_pulse_length),]
                    'sine': [0.0] * int(self.readout_len),
                },
                'integW_sine': {
                    'cosine': [0.0] * int(self.readout_len),
                    'sine': [1.0] * int(self.readout_len),
                },
            },
        }
