import numpy as np

lockin_freq = 50e5

readout_pulse_length = 1000

config = {
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
