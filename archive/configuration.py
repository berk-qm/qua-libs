import numpy as np

lockin_freq = 50e5

readout_pulse_length = 1000

_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
            },
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0},
            }
        }
    },
    "elements": {
        "lockin": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": lockin_freq,
            "operations": {
                "CW": "constPulse",
                'readout': 'readout_pulse'
            },
            'outputs': {'out1': ('con1', 1)},
            'time_of_flight': 184,
            'smearing': 0,
        },
        "qe": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": lockin_freq,
            "operations": {
                "CW": "constPulse",
                'readout': 'readout_pulse'
            },
            'hold_offset': {'duration': 1},
            'outputs': {'out1': ('con1', 1)},
            'time_of_flight': 184,
            'smearing': 0,
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": 16,  # in ns
            "waveforms": {"single": "const_wf"},
        },
        "readout_pulse": {
            'operation': 'measurement',
            'length': readout_pulse_length,
            'waveforms': {
                'single': 'zero_wf',
            },
            'integration_weights': {
                'integ_weights_cos': 'integW_cosine',
                'integ_weights_sin': 'integW_sine',
            },
            'digital_marker': 'ON',
        },
    },

    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        'zero_wf': {"type": "constant", "sample": 0.3}
    },
    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },
    'integration_weights': {
        'integW_cosine': {
            'cosine': [1.0] * int(readout_pulse_length / 4),  #[(1.0, readout_pulse_length),]
            'sine': [0.0] * int(readout_pulse_length / 4),
        },
        'integW_sine': {
            'cosine': [0.0] * int(readout_pulse_length / 4),
            'sine': [1.0] * int(readout_pulse_length / 4),
        },
    },
}

def drag_pulse(N: int, alpha: float, anharomonicity_factor: float):
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
    sig_i = (0.5 - 2 ** -16) * 0.5 * (1 - np.cos(2 * np.pi * t / (N - 1)))
    sig_q = -(0.5 - 2 ** -16) * 0.5 * - (alpha / (anharomonicity_factor * (N - 1))) * np.sin(2 * np.pi * t / (N - 1))
    return np.array([sig_i, sig_q])


pi_len = 36
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opxp",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
                4: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qubit0": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2)},
            "intermediate_frequency": 0,
            "operations": {
                "pi": "pi_pulse",
            },
        },
        # "qe2": {
        #     "mixInputs": {"I": ("con1", 3), "Q": ("con1", 4)},
        #     "intermediate_frequency": 0,
        #     "operations": {
        #         "pi": "pi_pulse",
        #     },
        # },
    },
    "pulses": {
        "pi_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "pi_i", "Q": "pi_q"},
        },
    },
    "waveforms": {
        "pi_i": {
            "type": "arbitrary",
            "samples": drag_pulse(pi_len, 0, 1)[0].tolist(),
        },
        "pi_q": {
            "type": "arbitrary",
            "samples": drag_pulse(pi_len, 0, 1)[1].tolist(),
        },
    }
}