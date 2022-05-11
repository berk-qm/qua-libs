import numpy as np
from qm.qua import declare, fixed, measure, dual_demod, assign
from scipy.signal.windows import gaussian
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms


#######################
# AUXILIARY FUNCTIONS #
#######################

# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

    :param g: relative gain imbalance between the I & Q ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians), set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#############
# VARIABLES #
#############

qop_ip = "127.0.0.1"
qop_port = 80

# Qubits
qubit_IF = 50e6
qubit_LO = 7e9
mixer_qubit_g = 0.0
mixer_qubit_phi = 0.0

qubit_T1 = int(10e3)

saturation_len = 1000
saturation_amp = 0.1
const_len = 100
const_amp = 0.1
square_pi_len = 100
square_pi_amp = 0.1

gauss_len = 20
gauss_sigma = gauss_len / 5
gauss_amp = 0.35
gauss_wf = gauss_amp * gaussian(gauss_len, gauss_sigma)

x180_len = 40
x180_sigma = x180_len / 5
x180_amp = 0.35
x180_drag_wf, x180_drag_der_wf = np.array(
    drag_gaussian_pulse_waveforms(x180_amp, x180_len, x180_sigma, alpha=0, delta=1)
)
# No DRAG when alpha=0, it's just a gaussian.

x90_len = x180_len
x90_sigma = x90_len / 5
x90_amp = x180_amp / 2
x90_drag_wf, x90_drag_der_wf = np.array(drag_gaussian_pulse_waveforms(x90_amp, x90_len, x90_sigma, alpha=0, delta=1))
# No DRAG when alpha=0, it's just a gaussian.

# Readout Resonator
resonator_IF = 60e6
resonator_LO = 5.5e9
mixer_resonator_g = 0.0
mixer_resonator_phi = 0.0

time_of_flight = 180

short_readout_len = 500
short_readout_amp = 0.4
readout_len = 5000
readout_amp = 0.2
long_readout_len = 50000
long_readout_amp = 0.1

# bosonic mode
bmode_IF = 66e6

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # I qubit
                2: {"offset": 0.0},  # Q qubit
                3: {"offset": 0.0},  # readout resonator (rr)
                4: {"offset": 0.0},  # bosonic mode
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0}
            },
        },
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "const": "const_pulse_IQ",
                "saturation": "saturation_pulse",
                "gauss": "gaussian_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
            },
        },
        "rr": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": resonator_IF,
            "operations": {
                "const": "const_pulse_ssb",
                "readout": "readout_pulse"
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "bmode": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": bmode_IF,
            "operations": {
                "const": "const_pulse_ssb",
            },
        },
    },
    "pulses": {
        "const_pulse_IQ": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "const_pulse_ssb": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "single": "const_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {
                "I": "gauss_wf",
                "Q": "zero_wf",
            },
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "x180_drag_wf",
                "Q": "x180_drag_der_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_drag_wf",
                "Q": "x90_drag_der_wf",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "readout_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
        "x180_drag_wf": {"type": "arbitrary", "samples": x180_drag_wf.tolist()},
        "x180_drag_der_wf": {"type": "arbitrary", "samples": x180_drag_der_wf.tolist()},
        "x90_drag_wf": {"type": "arbitrary", "samples": x90_drag_wf.tolist()},
        "x90_drag_der_wf": {"type": "arbitrary", "samples": x90_drag_der_wf.tolist()},
        "readout_wf": {"type": "constant", "sample": readout_amp},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(mixer_qubit_g, mixer_qubit_phi),
            }
        ],
    },
}
