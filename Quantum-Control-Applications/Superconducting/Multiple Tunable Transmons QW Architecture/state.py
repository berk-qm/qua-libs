"""
The process to generate the initial state and the build config function:
1. write the build_config function with references to the state
2. copy the config as a key in the state dict variable
3. replace the references in the state by placeholders
4. clear the state from not required fields
5. open a qm
6. generate the bootstrap_state_old.json
"""
import quam_sdk.constructor
import numpy as np
from scipy.signal.windows import dpss
from architecture import *


"""
For 1 OPX we have: 
    up to 4 fixed freq transmons or 2 charge tunable + 1 fixed freq transmons  
    1 readout line coupled to up to 4 readout resonators 
"""

# layer 1: bare state QUantum Abstract Machine
state = {
    # Make these functions directly available from the machine to be called in specific programs
    "_func": [
        "config.build_config",
        "config.save",
        "config.save_results",
        "config.get_wiring",
        "config.get_sequence_state",
        "config.get_qubit",
        "config.get_resonator",
        "config.get_qubit_gate",
        "config.get_charge_bias_point",
        "config.get_readout_IF",
        "config.get_qubit_IF",
        "config.set_f_res_vs_charge_vertex",
        "config.get_f_res_from_charge",
    ],
    "network": {"qop_ip": "172.16.2.103", "port": 80},
    "controllers": ["con1"],
    # Standard digital waveforms
    "digital_waveforms": [{"name": "ON", "samples": [[1, 0]]}],
    # Just put conventional pulses to all elements; qubit, readout & charge drives will be added later
    "common_operation": {
        "_docs": "an operation which is common to all elements",
        "name": "const",
        "duration": 16e-9,  # minimum length so that manipulation is at desired
        "duration_docs": "pulse length [s]",
        "amplitude": 0.2,
        "amplitude_docs": "pulse amplitude [V]",
    },
    # Readout lines containing information about the readout length, LO frequency and power,
    # and connectivity for the up- and down-conversion sides
    "readout_lines": [
        {
            "length": 100e-9,
            "length_docs": "readout time on this readout line [s]",
            "lo_freq": 6.5e9,
            "lo_freq_docs": "LO frequency for readout line [Hz]",
            "lo_power": 15,
            "lo_power_docs": "LO power for readout line [dBm]",
            "I_up": {"controller": "con1", "channel": 9, "offset": 0.0},
            "Q_up": {"controller": "con1", "channel": 10, "offset": 0.0},
            "I_down": {"controller": "con1", "channel": 1, "offset": 0.0, "gain_db": 1},
            # "Q_down": {"controller": "con1", "channel": 2, "offset": 0.0, "gain_db": 1},
            "switch": {"controller": "con1", "channel": 9},
            "switch_docs": "digital output declaration",
        },
    ],
    "readout_resonators": [
        {
            "index": i,
            "name": f"resonator_{i}",
            "f_res": 6.7e9,
            "f_res_docs": "Resonator resonance frequency [Hz].",
            "f_opt": 6.7e9,
            "f_opt_docs": "Resonator optimal readout frequency [Hz] (used in QUA).",
            "readout_regime": "low_power",
            "readout_amplitude": 0.1,
            "readout_amplitude_docs": "Readout amplitude for this resonator [V]. Must be within [-0.5, 0.5).",
            "rotation_angle": 0.0,
            "rotation_angle_docs": "Angle by which to rotate the IQ blobs to place the separation along the 'I' quadrature [degrees].",
            "integration_weights": [
                {"name": "optimal_cos", "cosine": [1.0] * 25, "sine": [0.0] * 25},
                {"name": "optimal_sin", "cosine": [0.0] * 25, "sine": [1.0] * 25},
                {"name": "optimal_minus_sin", "cosine": [0.0] * 25, "sine": [-1.0] * 25},
            ],
            "integration_weights_docs": "Arbitrary integration weights defined as lists of tuples whose first element is the value of the integration weight and second element is the duration in ns for which this value should be used [(1.0, readout_len)]. The duration must be divisible by 4.",
            "ge_threshold": 0.0,
            "ge_threshold_docs": "Threshold (in demod unit) along the 'I' quadrature discriminating between qubit ground and excited states.",
            "readout_fidelity": 0.84,
            "q_factor": 1e4,
            "chi": 1e6,
            "relaxation_time": 5e-6,
            "relaxation_time_docs": "Resonator relaxation time [s].",
            "f_res_vs_charge": {
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
            },
            "f_res_vs_charge_docs": "Vertex of the resonator frequency vs charge bias parabola as a * bias**2 + b * bias + c",
            "wiring": {
                "readout_line_index": 0,
                "readout_line_index_docs": "Index of the readout line connected to this resonator.",
                "time_of_flight": 272,
                "time_of_flight_docs": "Time of flight for this resonator [ns].",
                "correction_matrix": {"gain": 0.0, "phase": 0.0},
                "maximum_amplitude": 0.4,
                "maximum_amplitude_docs": "max amplitude in volts above which the mixer will send higher harmonics.",
                "switch_delay": 0,
                "switch_delay_docs": "delay of digital pulse",
                "switch_buffer": 0,
                "switch_buffer_docs": "buffer of digital pulse"
            },
            "threads": {
                "thread_cond": True,
                "thread_cond_docs": "Boolean to decided to put manual threds or not",
                "thread": threads[i],
                "thread_docs": "Manual thread being allocated",
            },
        }
        for i in range(READOUT_RESONATORS_PER_FEED_LINE)
    ],
    "drive_lines": [
        {
            "qubits": [i * NUMBER_OF_QUBITS_PER_DRIVE_LINE + j for j in range(NUMBER_OF_QUBITS_PER_DRIVE_LINE)],
            "qubits_docs": "qubits associated with this drive line",
            "lo_freq": 5.5e9,
            "lo_freq_docs": "LO frequency [Hz]",
            "lo_power": 15,
            "lo_power_docs": "LO power to drive line [dBm]",
            "I": {"controller": "con1", "channel": 1 + 3 * i, "offset": 0.0},
            "Q": {"controller": "con1", "channel": 2 + 3 * i, "offset": 0.0},
            "switch": {"controller": "con1", "channel": 1 + i},
            "switch_docs": "digital output declaration",
        }
        for i in range(NUMBER_OF_DRIVE_LINES)
    ],
    "qubits": [
        {
            "index": i,
            "name": f"qubit_{i}",
            "f_01": 5.7e9,
            "f_01_docs": "0-1 transition frequency [Hz]",
            "df": 1e6,
            "df_docs": "Half of charge dispersion measured in spectroscopy [Hz]",
            "anharmonicity": 350e6,
            "anharmonicity_docs": "Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]",
            "rabi_freq": 0,
            "rabi_freq_docs": "Qubit Rabi frequency [Hz]",
            "t1": 18e-6,
            "t1_docs": "Relaxation time T1 [s]",
            "t2": 5e-6,
            "t2_docs": "Dephasing time T2 [s]",
            "t2star": 1e-6,
            "t2star_docs": "Dephasing time T2* [s]",
            "ramsey_det": 10e6,
            "ramsey_det_docs": "Detuning to observe ramsey fringes [Hz]",
            "driving": {
                "drag_gaussian": {
                    "length": 80e-9,
                    "length_docs": "The pulse length [s]",
                    "sigma": 10e-9,
                    "sigma_docs": "The gaussian standard deviation (only for gaussian pulses) [s]",
                    "alpha": 0.0,
                    "alpha_docs": "The DRAG coefficient alpha.",
                    "detuning": 1,
                    "detuning_docs": "The frequency shift to correct for AC stark shift [Hz].",
                    "shape": "drag_gaussian",
                    "shape_docs": "Shape of the gate",
                    "angle2volt": {"deg90": 0.25, "deg180": 0.49},
                    "angle2volt_docs": "Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V.",
                },
                "drag_cosine": {
                    "length": 80e-9,
                    "length_docs": "The pulse length [s]",
                    "alpha": 0.0,
                    "alpha_docs": "The DRAG coefficient alpha.",
                    "detuning": 1,
                    "detuning_docs": "The frequency shift to correct for AC stark shift [Hz].",
                    "shape": "drag_cosine",
                    "shape_docs": "Shape of the gate",
                    "angle2volt": {"deg90": 0.25, "deg180": 0.49},
                    "angle2volt_docs": "Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V.",
                },
                "square": {
                    "length": 80e-9,
                    "length_docs": "The pulse length [s]",
                    "shape": "square",
                    "shape_docs": "Shape of the gate",
                    "angle2volt": {"deg90": 0.25, "deg180": 0.49},
                    "angle2volt_docs": "Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V.",
                },
            },
            "wiring": {
                "drive_line_index": int(np.floor(i / NUMBER_OF_QUBITS_PER_DRIVE_LINE)),
                "drive_line_index_docs": "Index of the readout line connected to this qubit.",
                "correction_matrix": {"gain": 0.0, "phase": 0.0},
                "maximum_amplitude": 0.4,
                "maximum_amplitude_docs": "max amplitude in volts above which the mixer will send higher harmonics.",
                "analog_channel_offset": 0.0,
                "analog_channel_offset_docs": "Voltage value to nullify inheret analog channel offset [V]",
                "charge_line": {"controller": "con1", "channel": 5 + i, "offset": 0.0},
                "charge_filter_coefficients": {
                    "feedforward": [],
                    "feedback": [],
                },
                "switch_delay": 0,
                "switch_delay_docs": "delay of digital pulse",
                "switch_buffer": 0,
                "switch_buffer_docs": "buffer of digital pulse"
            },
            "threads": {
                "thread_cond": True,
                "thread_cond_docs": "Boolean to decided to put manual threds or not",
                "thread": threads[i],
                "thread_docs": "Manual thread being allocated",
            },
            "charge_bias_points": [
                {
                    "name": "degeneracy_point",
                    "value": 0.0,
                    "value_docs": "Bias voltage to set qubit to degeneracy between even and odd parity [V]",
                },
                {
                    "name": "max_dispersion_point",
                    "value": 0.0,
                    "value_docs": "Bias voltage that maximizes the frequency separation between even and odd parity [V]",
                },
                {
                    "name": "working_point",
                    "value": 0.0,
                    "value_docs": "Arbitrary bias voltage for your own desired working point",
                },
            ],
            "sequence_states": {
                "constant": [
                    {
                        "name": "qubit_spectroscopy",
                        "amplitude": 0.4,
                        "amplitude_docs": "[V]",
                        "length": 1e-6,
                        "length_docs": "[s]",
                    },
                ],
                "arbitrary": [
                    {
                        "name": "slepian",
                        "waveform": (dpss(200, 5) * 0.5)[:100].tolist(),
                        "waveform_docs": "points describing the waveform shape",
                    }
                ],
            },
        }
        for i in range(NUMBER_OF_QUBITS)
    ],
    "crosstalk_matrix": {
        # index 0, 1 -> correspond to qubit0 talking to qubit1
        "static": [
            [1.0 if i == j else 0.0 for i in range(NUMBER_OF_QUBITS)] for j in range(NUMBER_OF_QUBITS)
        ],
        "fast": [
            [1.0 if i == j else 0.0 for i in range(NUMBER_OF_QUBITS)] for j in range(NUMBER_OF_QUBITS)
        ],
    },
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
    "charge_lines": [
        {
            "analog_channel_offset": 0.0,
            "analog_channel_offset_docs": "Voltage value to nullify inheret analog channel offset [V]",
            "charge_line": {"controller": "con1", "channel": 5 + i, "offset": 0.0},
            "charge_filter_coefficients": {
                "feedforward": [],
                "feedback": [],
            },
            "charge_bias_points": [
                {
                    "name": "degeneracy_point",
                    "value": 0.0,
                    "value_docs": "Bias voltage to set qubit to degeneracy between even and odd parity [V]",
                },
                {
                    "name": "max_dispersion_point",
                    "value": 0.0,
                    "value_docs": "Bias voltage that maximizes the frequency separation between even and odd parity [V]",
                },
                {
                    "name": "working_point",
                    "value": 0.0,
                    "value_docs": "Arbitrary bias voltage for your own desired working point",
                },
            ],
        }
        for i in range(NUMBER_OF_CHARGE_LINES)
    ],
    "qp_injectors": [
        {
            "index": i,
            "name": f"qp_injector_{i}",
            "energy_gap": 0.01,
            "energy_gap_docs": "Superconducting energy gap [V]",
            "injection_voltage": 0.001,
            "injection_voltage_docs": "Injection pulse voltage for phonon injection experiment [V]",
            "injection_length": 10e-6,
            "injection_length_docs": "Injection pulse lenght for phonon injection experiment [s]",
            "analog_channel_offset": 0.0,
            "analog_channel_offset_docs": "Voltage value to nullify inheret analog channel offset [V]",
            "wiring": {
                "injector_line": {"controller": "con1", "channel": 7 + i, "offset": 0.0},
            },
        }
        for i in range(NUMBER_OF_QP_INJECTORS)
    ],
    "results": {"directory": ""},
    "running_strategy": {"running": True, "start": [], "end": []},
}

# Now we use QuAM SDK

quam_sdk.constructor.quamConstructor(state, reuse_existing_values=False)
