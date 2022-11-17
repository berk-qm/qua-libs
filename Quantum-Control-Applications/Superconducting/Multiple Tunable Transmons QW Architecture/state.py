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


"""
For 1 OPX we have: 
    up to 4 fixed freq transmons or 2 flux tunable + 1 fixed freq transmons  
    1 readout line coupled to up to 4 readout resonators 
"""

READOUT_RESONATORS_PER_FEED_LINE = 2

NUMBER_OF_QUBITS = 2
NUMBER_OF_DRIVE_LINES = 2
NUMBER_OF_QUBITS_PER_DRIVE_LINE = 1


# layer 1: bare state QUantum Abstract Machine
state = {
    # Make these functions directly available from the machine to be called in specific programs
    "_func": [
        "config.build_config",
        "config.save",
        "config.get_wiring",
        "config.get_sequence_state",
        "config.get_qubit",
        "config.get_resonator",
        "config.get_driving",
        "config.get_qubit_gate",
        "config.get_length",
        "config.set_length",
        "config.get_flux_bias_point",
    ],
    "network": {"qop_ip": "172.16.2.103", "port": 85},
    "controllers": ["con1"],
    # Standard digital waveforms
    "digital_waveforms": [{"name": "ON", "samples": [[1, 0]]}],
    # Just put standard pulses; qubit, readout & flux drives will be added later
    "common_operation": {
        "_docs": "an operation which is common to all elements",
        "name": "const",
        "duration": 100e-9,
        "amplitude": 0.2,
    },
    # Readout lines containing information about the readout length, LO frequency and power,
    # and connectivity for the up- and down-conversion sides
    "readout_lines": [
        {
            "length": 100e-9,  # sec
            "length_docs": "readout time on this readout line [s]",
            "lo_freq": 7e9,  # Hz
            "lo_freq_docs": "LO frequency for readout line [Hz]",
            "lo_power": 15,  # dBm
            "lo_power_docs": "LO power for readout line [dBm]",
            "I_up": {"controller": "con1", "channel": 9, "offset": 0.0},
            "Q_up": {"controller": "con1", "channel": 10, "offset": 0.0},
            "I_down": {"controller": "con1", "channel": 1, "offset": 0.0, "gain_db": 1},
            "Q_down": {"controller": "con1", "channel": 2, "offset": 0.0, "gain_db": 1},
        },
    ],
    "readout_resonators": [
        {
            "index": i,
            "name": f"resonator_{i}",
            "f_res": 7.1e9,  # Hz
            "f_res_docs": "Resonator frequency [Hz]",
            "readout_regime": "low_power",
            "readout_amplitude": 0.24,
            "readout_amplitude_docs": "Readout amplitude for this resonator [V]. Must be within [-0.5, 0.5).",
            "rotation_angle": 41.3,  # degrees
            "rotation_angle_docs": "Angle by which to rotate the IQ blobs to place the separation along the 'I' quadrature [degrees].",
            "ge_threshold": 0.0,  # degrees
            "ge_threshold_docs": "Threshold along the 'I' quadrature discriminating between qubit ground and excited states.",
            "opt_readout_frequency": 6.52503e9,
            "readout_fidelity": 0.84,
            "q_factor": 1e4,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "readout_line_index_docs": "Index of the readout line connected to this resonator.",
                "time_of_flight": 272,
                "time_of_flight_docs": "Time of flight for this resonator [ns].",
                "correction_matrix": {"gain": 0.0, "phase": 0.0},
            },
        }
        for i in range(READOUT_RESONATORS_PER_FEED_LINE)
    ],
    "drive_lines": [
        {
            "qubits": [i * NUMBER_OF_QUBITS_PER_DRIVE_LINE + j for j in range(NUMBER_OF_QUBITS_PER_DRIVE_LINE)],
            "qubits_docs": "qubits associated with this drive line",
            "lo_freq": 4.5e9,  # Hz
            "lo_freq_docs": "LO frequency [Hz]",
            "lo_power": 15,  # dB
            "lo_power_docs": "LO power to drive line [dBm]",
            "I": {"controller": "con1", "channel": 1 + 2 * i, "offset": 0.0},
            "Q": {"controller": "con1", "channel": 2 + 2 * i, "offset": 0.0},
        }
        for i in range(NUMBER_OF_DRIVE_LINES)
    ],
    "qubits": [
        {
            "index": i,
            "name": f"qubit_{i}",
            "f_01": 4.52503e9,  # Hz
            "f_01_docs": "0-1 transition frequency [Hz]",
            "anharmonicity": 350e6,
            "anharmonicity_docs": "Qubit anharmonicity defined as the difference in energy between the 2-1 and the 1-0 energy levels [Hz]",
            "rabi_freq": 0,
            "rabi_freq_docs": "Qubit Rabi frequency [Hz]",
            "t1": 18e-6,
            "t1_docs": "Relaxation time T1 [s]",
            "t2": 5e-6,
            "t2_docs": "Dephasing time T2 [s]",
            "t2star": 1e-6,
            "t2star_docs": "Dephasing time T2* [s]",
            "driving": {
                "drag_gaussian": {
                    "gate_len": 60e-9,  # Sec
                    "gate_len_docs": "The pulse length [s]",
                    "gate_sigma": 10e-9,
                    "gate_sigma_docs": "The gaussian standard deviation (only for gaussian pulses) [s]",
                    "alpha": 0.0,
                    "alpha_docs": "The DRAG coefficient alpha.",
                    "detuning": 1,
                    "detuning_docs": "The frequency shift to correct for AC stark shift [Hz].",
                    "gate_shape": "drag_gaussian",
                    "gate_shape_docs": "Shape of the gate",
                    "angle2volt": {"deg90": 0.25, "deg180": 0.49},
                    "angle2volt_docs": "Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V.",
                },
                "drag_cosine": {
                    "gate_len": 60e-9,  # Sec
                    "gate_len_docs": "The pulse length [s]",
                    "alpha": 0.0,
                    "alpha_docs": "The DRAG coefficient alpha.",
                    "detuning": 1,
                    "detuning_docs": "The frequency shift to correct for AC stark shift [Hz].",
                    "gate_shape": "drag_cosine",
                    "gate_shape_docs": "Shape of the gate",
                    "angle2volt": {"deg90": 0.25, "deg180": 0.49},
                    "angle2volt_docs": "Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V.",
                },
            },
            "wiring": {
                "drive_line_index": int(np.floor(i / NUMBER_OF_QUBITS_PER_DRIVE_LINE)),
                "correction_matrix": {"gain": 0.0, "phase": 0.0},
                "flux_line": {"controller": "con1", "channel": 8 - i, "offset": 0.0},
                "flux_filter_coef": {
                    "feedforward": [],
                    "feedback": [],
                },
            },
            "flux_bias_points": [
                {"name": "flux_insensitive_point", "value": 0.1},
                {"name": "flux_zero_frequency_point", "value": 0.1},
                {"name": "anti_crossing", "value": 0.1},
            ],
            "sequence_states": {
                "constant": [
                    {"name": "dissipative_stabilization", "amplitude": 0.2, "length": 200},
                    {"name": "Excitation", "amplitude": 0.3, "length": 80},
                    {"name": "Free_evolution", "amplitude": 0.2, "length": 200},
                    {"name": "Jump", "amplitude": 0.4, "length": 16},
                    {"name": "Readout", "amplitude": 0.35, "length": 1000},
                    {"name": "flux_balancing", "amplitude": -0.35, "length": 400},
                ],
                "arbitrary": [{"name": "slepian", "waveform": (dpss(200, 5) * 0.5)[:100].tolist()}],
            },
        }
        for i in range(NUMBER_OF_QUBITS)
    ],
    # measure qubit 1 while playing a flux to qubit 0
    "crosstalk_matrix": {
        # index 0, 1 -> correspond to qubit0 talking to qubit1
        "static": [[1.0 if i == j else 0.0 for i in range(NUMBER_OF_QUBITS)] for j in range(NUMBER_OF_QUBITS)],
        "fast": [[1.0 if i == j else 0.0 for i in range(NUMBER_OF_QUBITS)] for j in range(NUMBER_OF_QUBITS)],
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
    "two_qubit_gates": [],
    "running_strategy": {"running": True, "start": [], "end": []},
}

# Now we use QuAM SDK

quam_sdk.constructor.quamConstructor(state)
