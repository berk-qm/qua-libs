import numpy as np
from itertools import chain

lockin_freq = 50e5

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


def drag_pulse(N: int,amp,  alpha: float, anharomonicity_factor: float):
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
    sig_q = amp * (-(0.5 - 2 ** -16) * 0.5 * - (alpha / (anharomonicity_factor * (N - 1))) * np.sin(2 * np.pi * t / (N - 1)))
    return np.array([sig_i, sig_q])

def constant_pulse(N: int, val: float):
    return val * np.ones(N)


c1_phxz = {"c1_0": (0,1,0), "c1_1": (0,-0.5,0), "c1_2": (0.5,-0.5,1), "c1_3": (0.5,-0.5, -0.5),
                   "c1_4": (0,0.5,0.5), "c1_5": (0,0,0.5)}

# s1_phxz = {"s1_0": (0,0,0), "s1_1": (0,0.5,0.5), "s1_2": (0,0.5,0.5)} #ORIG
s1_phxz = {"s1_0": (0,1,0), "s1_1": (0,0.5,0.5), "s1_2": (0,0.5,0.5)}

cnot_phxz = {"cnot_0_0": (0.5,0.5,-1), "cnot_0_1": (0,0,1),
             "cnot_2_0": (0,1,0), "cnot_2_1": (0, 0, 0),
             "cnot_4_0": (0.5,0.5,0.5), "cnot_4_1": (-1,0.5,1)
             }

swap_phxz = {'swap_0_0': (0,0.5,0.5), 'swap_0_1': (0.5,0.5, 0),
            'swap_2_0': (0,0.5,0.5), 'swap_2_1': (0.5,0.5, 0),
            'swap_4_0': (0,0.5,0.5), 'swap_4_1': (0.5,0.5, 0),
            'swap_6_0': (0,0,-0.5), 'swap_6_1': (0,1,-0.5)
             }

# pauli_phxz = {"pauli_0": (0,0,0), "pauli_1": (1,0,0), "pauli_2": (0,1,0), "pauli_3": (1,0,0.5)} #ORIG
pauli_phxz = {"pauli_0": (0,0.7,0), "pauli_1": (1,0.7,0), "pauli_2": (0,0.7,0), "pauli_3": (1,0.7,0.5)}


class ConfigBuilder:

    PI_PULSE_LEN = 72
    _GATES_TO_PULSERS_3P = {"C1": 0, "PAULI": 1, "CNOT": 2, "ISWAP": 2, "SWAP": 2}
    _GATES_TO_PULSERS_1P = {"C1": 0, "PAULI": 0, "CNOT": 0, "ISWAP": 0, "SWAP": 0}
    _GATES_TO_PULSERS = {}

    def __init__(self, pi_pulse_len=PI_PULSE_LEN, cz_pulse_len=16, bake_cnot=True, bake_swap=True, pulsers_per_qubit=3):
        self.pi_pulse_len = pi_pulse_len
        self.cz_pulse_len = cz_pulse_len
        self.bake_cnot = bake_cnot
        self.bake_swap = bake_swap
        self.config = {}
        self.readout_len = 256
        self.pulsers_per_qubit = pulsers_per_qubit
        ConfigBuilder._GATES_TO_PULSERS = ConfigBuilder._GATES_TO_PULSERS_3P if pulsers_per_qubit == 3 else ConfigBuilder._GATES_TO_PULSERS_1P
        self.combine_pulsers_of_qubits = True
        return

    def build(self):
        self._place_template()
        self.add_elemntry_gates()
        self.add_cnot_gates()
        self.add_swap_gates()
        if self.bake_cnot:
            self.bake_cnot_wf()
        if self.bake_swap:
            self.bake_swap_wf()
        return self.config

    def tuple_to_concrete_pulse(self, t, prior_z_phase=None):
        if t[1] != 0:
            p = drag_pulse(self.pi_pulse_len,t[1], 0, 1)
        else:
            p = np.zeros(shape=(2, self.pi_pulse_len))
        z_factor = 1 if prior_z_phase is None else prior_z_phase
        p[0] = p[0] * np.cos(t[0] * np.pi * z_factor) + p[1] * np.sin(t[0] * np.pi * z_factor)
        p[1] = p[0] *  -np.sin(t[0] * np.pi * z_factor) + p[1] * np.cos(t[0] * np.pi * z_factor)
        p[np.where(p == 0)] = 0.0
        return p

    def add_elemntry_gates(self):
        for p, v in list(c1_phxz.items()) + list(pauli_phxz.items()) + list(s1_phxz.items()):
            iq_pulse = self.tuple_to_concrete_pulse(v)
            self.config["waveforms"][p + "_i"] = {"type": "arbitrary", 'samples': iq_pulse[0]}
            self.config["waveforms"][p + "_q"] = {"type": "arbitrary", 'samples': iq_pulse[1]}
            self.config["pulses"][p + "_pulse"] = {"operation": "control",
                                              "length": len(iq_pulse[0]),
                                              "waveforms": {"I": p + "_i", "Q": p + "_q"}}
            for i in range(self.pulsers_per_qubit):
                self.config["elements"][f"qubit0_xy_p{i}"]["operations"][p] = p + "_pulse"
                self.config["elements"][f"qubit1_xy_p{i}"]["operations"][p] = p + "_pulse"
        return

    def add_cnot_gates(self):
        for g, v in cnot_phxz.items():
            moment = int(g.split("_")[1])
            elem = int(g.split("_")[-1])
            pulser = self._GATES_TO_PULSERS["CNOT"]
            if moment > 0:
                iq_pulse = self.tuple_to_concrete_pulse(v, cnot_phxz[f"cnot_{moment - 2}_{elem}"][2])
            else:
                iq_pulse = self.tuple_to_concrete_pulse(v)
            self.config["waveforms"][g + "_i"] = {"type": "arbitrary", 'samples': iq_pulse[0]}
            self.config["waveforms"][g + "_q"] = {"type": "arbitrary", 'samples': iq_pulse[1]}
            self.config["pulses"][g + "_pulse"] = {"operation": "control", "length": len(iq_pulse[0]),
                                              "waveforms": {"I": g + "_i", "Q": g + "_q"}}
            self.config["elements"][f"qubit{elem}_xy_p{pulser}"]["operations"][g] = g + "_pulse"

    def add_swap_gates(self):
        for g, v in swap_phxz.items():
            moment = int(g.split("_")[1])
            elem = int(g.split("_")[-1])
            pulser = self._GATES_TO_PULSERS["SWAP"]
            if moment > 0:
                iq_pulse = self.tuple_to_concrete_pulse(v, swap_phxz[f"swap_{moment - 2}_{elem}"][2])
            else:
                iq_pulse = self.tuple_to_concrete_pulse(v)
            self.config["waveforms"][g + "_i"] = {"type": "arbitrary", 'samples': iq_pulse[0]}
            self.config["waveforms"][g + "_q"] = {"type": "arbitrary", 'samples': iq_pulse[1]}
            self.config["pulses"][g + "_pulse"] = {"operation": "control", "length": len(iq_pulse[0]),
                                              "waveforms": {"I": g + "_i", "Q": g + "_q"}}
            self.config["elements"][f"qubit{elem}_xy_p{pulser}"]["operations"][g] = g + "_pulse"

    def bake_cnot_wf(self):
        cnot_length = 3 * self.pi_pulse_len + 2 * self.cz_pulse_len
        self.cnot_length = cnot_length
        moment_lengths = np.cumsum([0, self.pi_pulse_len, self.cz_pulse_len, self.pi_pulse_len, self.cz_pulse_len, self.pi_pulse_len])
        qubit0_xy_wf = np.zeros(shape=(2, cnot_length))
        qubit1_xy_wf = np.zeros(shape=(2, cnot_length))
        qubit_0_z_wf = np.zeros(shape=(cnot_length))
        qubit_1_z_wf = np.zeros(shape=(cnot_length))
        coupler_wf = np.zeros(shape=(cnot_length))

        cumulative_phase = [0, 0]
        for mom in [0, 2, 4]:
            phase = 0
            qubit0_xy_wf[:, moment_lengths[mom]:moment_lengths[mom+1]] = self.tuple_to_concrete_pulse(cnot_phxz[f"cnot_{mom}_0"],
                                                                                    prior_z_phase=phase)
            qubit1_xy_wf[:, moment_lengths[mom]:moment_lengths[mom+1]] = self.tuple_to_concrete_pulse(cnot_phxz[f"cnot_{mom}_1"],
                                                                                    prior_z_phase=phase)
            cumulative_phase[0] *= phase
            cumulative_phase[0] *= phase

        for mom in [1,3]:
            qubit_0_z_wf[moment_lengths[mom]: moment_lengths[mom+1]] = constant_pulse(self.cz_pulse_len, 0.2)
            qubit_1_z_wf[moment_lengths[mom]: moment_lengths[mom+1]] = constant_pulse(self.cz_pulse_len, 0.2)
            coupler_wf[moment_lengths[mom]: moment_lengths[mom + 1]] = constant_pulse(self.cz_pulse_len, 0.2)

        # Waveforms of xy for both qubits
        self.config["waveforms"]["cnot_xy_0_i"] = {"type": "arbitrary", 'samples': qubit0_xy_wf[0]}
        self.config["waveforms"]["cnot_xy_0_q"] = {"type": "arbitrary", 'samples': qubit0_xy_wf[1]}
        self.config["waveforms"]["cnot_xy_1_i"] = {"type": "arbitrary", 'samples': qubit1_xy_wf[0]}
        self.config["waveforms"]["cnot_xy_1_q"] = {"type": "arbitrary", 'samples': qubit1_xy_wf[1]}

        # Waveform of couplers
        self.config["waveforms"]["cnot_z_0"] = {"type": "arbitrary", 'samples': qubit_0_z_wf}
        self.config["waveforms"]["cnot_z_1"] = {"type": "arbitrary", 'samples': qubit_1_z_wf}
        self.config["waveforms"]["cnot_coupler"] = {"type": "arbitrary", 'samples': coupler_wf}


        self.config["pulses"]["cnot_baked_pulse_xy_0"] = {"operation": "control", "length": cnot_length,
                                               "waveforms": {"I": "cnot_xy_0_i", "Q": "cnot_xy_0_q"}}
        self.config["pulses"]["cnot_baked_pulse_xy_1"] = {"operation": "control", "length": cnot_length,
                                               "waveforms": {"I": "cnot_xy_1_i", "Q": "cnot_xy_1_q"}}
        self.config["pulses"]["cnot_baked_pulse_z_0"] = {"operation": "control", "length": cnot_length,
                                               "waveforms": {"single": "cnot_z_0"}}
        self.config["pulses"]["cnot_baked_pulse_z_1"] = {"operation": "control", "length": cnot_length,
                                               "waveforms": {"single": "cnot_z_1"}}
        self.config["pulses"]["cnot_baked_pulse_coupler"] = {"operation": "control", "length": cnot_length,
                                               "waveforms": {"single": "cnot_coupler"}}

        pulser = self._GATES_TO_PULSERS["CNOT"]
        self.config["elements"][f"qubit0_xy_p{pulser}"]["operations"]["cnot"] = "cnot_baked_pulse_xy_0"
        self.config["elements"][f"qubit1_xy_p{pulser}"]["operations"]["cnot"] = "cnot_baked_pulse_xy_1"
        self.config["elements"][f"qubit0_z"]["operations"]["cnot"] = "cnot_baked_pulse_z_0"
        self.config["elements"][f"qubit1_z"]["operations"]["cnot"] = "cnot_baked_pulse_z_1"
        self.config["elements"][f"coupler"]["operations"]["cnot"] = "cnot_baked_pulse_coupler"
        return

    def bake_swap_wf(self):
        swap_length = 4 * self.pi_pulse_len + 3 * self.cz_pulse_len
        self.swap_length = swap_length
        moment_lengths = np.cumsum(
            [0, self.pi_pulse_len, self.cz_pulse_len, self.pi_pulse_len, self.cz_pulse_len, self.pi_pulse_len,
             self.cz_pulse_len, self.pi_pulse_len])
        qubit0_xy_wf = np.zeros(shape=(2, swap_length))
        qubit1_xy_wf = np.zeros(shape=(2, swap_length))
        qubit_0_z_wf = np.zeros(shape=(swap_length))
        qubit_1_z_wf = np.zeros(shape=(swap_length))
        coupler_wf = np.zeros(shape=(swap_length))

        for mom in [0, 2, 4, 6]:
            qubit0_xy_wf[:, moment_lengths[mom]: moment_lengths[mom+1]] = self.tuple_to_concrete_pulse(swap_phxz[f"swap_{mom}_0"])
            qubit1_xy_wf[:, moment_lengths[mom]: moment_lengths[mom+1]] = self.tuple_to_concrete_pulse(swap_phxz[f"swap_{mom}_1"])
        for mom in [1,3,5]:
            qubit_0_z_wf[moment_lengths[mom]: moment_lengths[mom+1]] = constant_pulse(self.cz_pulse_len, 0.2)
            qubit_1_z_wf[moment_lengths[mom]: moment_lengths[mom+1]] = constant_pulse(self.cz_pulse_len, 0.2)
            coupler_wf[moment_lengths[mom]: moment_lengths[mom+1]] = constant_pulse(self.cz_pulse_len, 0.2)

        # Waveforms of xy for both qubits
        self.config["waveforms"]["swap_xy_0_i"] = {"type": "arbitrary", 'samples': qubit0_xy_wf[0]}
        self.config["waveforms"]["swap_xy_0_q"] = {"type": "arbitrary", 'samples': qubit0_xy_wf[1]}
        self.config["waveforms"]["swap_xy_1_i"] = {"type": "arbitrary", 'samples': qubit1_xy_wf[0]}
        self.config["waveforms"]["swap_xy_1_q"] = {"type": "arbitrary", 'samples': qubit1_xy_wf[1]}

        # Waveform of couplers
        self.config["waveforms"]["swap_z_0"] = {"type": "arbitrary", 'samples': qubit_0_z_wf}
        self.config["waveforms"]["swap_z_1"] = {"type": "arbitrary", 'samples': qubit_1_z_wf}
        self.config["waveforms"]["swap_coupler"] = {"type": "arbitrary", 'samples': coupler_wf}

        self.config["pulses"]["swap_baked_pulse_xy_0"] = {"operation": "control", "length": swap_length,
                                                          "waveforms": {"I": "swap_xy_0_i", "Q": "swap_xy_0_q"}}
        self.config["pulses"]["swap_baked_pulse_xy_1"] = {"operation": "control", "length": swap_length,
                                                          "waveforms": {"I": "swap_xy_1_i", "Q": "swap_xy_1_q"}}
        self.config["pulses"]["swap_baked_pulse_z_0"] = {"operation": "control", "length": swap_length,
                                                         "waveforms": {"single": "swap_z_0"}}
        self.config["pulses"]["swap_baked_pulse_z_1"] = {"operation": "control", "length": swap_length,
                                                         "waveforms": {"single": "swap_z_1"}}
        self.config["pulses"]["swap_baked_pulse_coupler"] = {"operation": "control", "length": swap_length,
                                                             "waveforms": {"single": "swap_coupler"}}

        pulser = self._GATES_TO_PULSERS["SWAP"]
        self.config["elements"][f"qubit0_xy_p{pulser}"]["operations"]["swap"] = "swap_baked_pulse_xy_0"
        self.config["elements"][f"qubit1_xy_p{pulser}"]["operations"]["swap"] = "swap_baked_pulse_xy_1"
        self.config["elements"][f"qubit0_z"]["operations"]["swap"] = "swap_baked_pulse_z_0"
        self.config["elements"][f"qubit1_z"]["operations"]["swap"] = "swap_baked_pulse_z_1"
        self.config["elements"][f"coupler"]["operations"]["swap"] = "swap_baked_pulse_coupler"

    def _place_template(self):
        num_analog_outputs = self.pulsers_per_qubit * 2 * 2 + 3 if not self.combine_pulsers_of_qubits else 7
        elements_dict = {f"qubit{i // self.pulsers_per_qubit}_xy_p{i % self.pulsers_per_qubit}": {
                "mixInputs": {"I": ("con1", i * 2 + 1 if not self.combine_pulsers_of_qubits else 2 * (i//self.pulsers_per_qubit) + 1),
                              "Q": ("con1", i * 2 + 2 if not self.combine_pulsers_of_qubits else 2 * (i//self.pulsers_per_qubit) + 2)},
                "intermediate_frequency": 0,
                "operations": {
                    "pi": "pi_pulse",
                    "readout_pulse": "readout_pulse",
                },
                'outputs': {f'out{i//self.pulsers_per_qubit + 1}': ('con1', i//self.pulsers_per_qubit  + 1 )},
                'time_of_flight': 184,
                'smearing': 0,
            } for i in range(self.pulsers_per_qubit * 2)}
        elements_dict.update({elem: {
                "singleInput": {"port": ("con1", port)},
                "intermediate_frequency": 0,
                "operations": {
                    "coupler_tone": "coupler_pulse"
                },
            } for elem, port in zip(["qubit0_z", "qubit1_z", "coupler"], range(num_analog_outputs-2, num_analog_outputs + 1))},
            )

        self.config =  {
        "version": 1,
        "controllers": {
            "con1": {
                "type": "opxp",
                "analog_outputs": { i: {'offset': +0.0} for i in range(1, num_analog_outputs+1)},
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
                "samples": drag_pulse(self.pi_pulse_len,1, 0, 1)[0].tolist(),
            },
            "pi_q": {
                "type": "arbitrary",
                "samples": drag_pulse(self.pi_pulse_len,1, 0, 1)[1].tolist(),
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
                'sine': [0.0] * int(self.readout_len ),
            },
            'integW_sine': {
                'cosine': [0.0] * int(self.readout_len),
                'sine': [1.0] * int(self.readout_len ),
            },
        },
    }
