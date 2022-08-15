import itertools
import pickle
from itertools import combinations, product
from typing import Tuple, List, Dict
import numpy as np
import time
import random
import matplotlib.pyplot as pplot

import cirq
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from matplotlib import pyplot as plt
import conf
import qua_simple_tableau


class Utils:

    @staticmethod
    def py_bin_vector_to_int32(vec):
        return int(sum([i * 2 ** e for e, i in enumerate(vec)]))

    @staticmethod
    def py_bin_matrix_to_int32(mat) -> int:
        r = 0
        for row_i, row in enumerate(mat):
            rep_num = sum([i * 2 ** e for e, i in enumerate(row)])
            r |= rep_num << (row_i * 4)
        return int(r)

    @staticmethod
    def qua_add_binary_search_block(val_to_search, target_list, target_list_size: int, result_ind):
        ind = declare(int, value=(target_list_size-1)//2)
        right = declare(int, value=target_list_size-1)
        left = declare(int, value=0)

        with while_(left <= right):
            with if_(val_to_search < target_list[ind]):
                assign(right, ind - 1)
            with elif_(val_to_search > target_list[ind]):
                assign(left, ind + 1)
            with else_():
                assign(result_ind, ind)
                assign(right, left-1)
            assign(ind,(right + left) >> 1)

    @staticmethod
    def phXZ_to_tuple(phXZ_gate):
        assert isinstance(phXZ_gate, cirq.PhasedXZGate)
        return (phXZ_gate.axis_phase_exponent, phXZ_gate.x_exponent, phXZ_gate.z_exponent)

    @staticmethod
    def bitN(on_value, N):
        return (on_value & (2 ** N)) >> N

    @staticmethod
    def get_4bit(val, n):
        assert (0 <= n < 8)
        return (val & (15 << (4 * n))) >> (4 * n)


class Paulis:
    pauli_I = cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0)
    pauli_X = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
    pauli_Z = cirq.PhasedXZGate(x_exponent=0, z_exponent=1, axis_phase_exponent=0)
    pauli_Y = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0.5)

    I = np.array([0,0])
    X = np.array([1,0])
    Z = np.array([0,1])
    Y = np.array([1,1])

    def __init__(self, num_qubits):
        assert (num_qubits in [1,2])
        self.paulis_alpha_vecs_reduced = [self.I, self.X, self.Z, self.Y]
        if num_qubits == 1:
            q = cirq.LineQubit(0)
            self.paulis_circuits = [cirq.Circuit(op(q)) for op in [self.pauli_I, self.pauli_X, self.pauli_Z, self.pauli_Y]]
            self.paulis_alpha_vecs = [np.hstack(([0,0], v)) for v in self.paulis_alpha_vecs_reduced]
        else:
            q1, q2 = cirq.LineQubit.range(2)
            self.paulis_circuits = []
            for op1, op2 in product([self.pauli_I, self.pauli_X, self.pauli_Z, self.pauli_Y], repeat=2):
                self.paulis_circuits.append(cirq.Circuit([op1(q1), op2(q2)]))

            self.paulis_alpha_vecs = []
            for comb in product(self.paulis_alpha_vecs_reduced, repeat=2):
                self.paulis_alpha_vecs.append(np.hstack((comb[0], comb[1])))


class SingleQubitSymplacticCompilationData:

    _symplectic_matrices_reduced = []
    _symplectic_matrices = []

    _single_qubit_gate_conversions = {
        'I': (np.identity(2), np.zeros(2)),
        'H': (np.array([[0, 1], [1, 0]]), np.zeros(2)),
        'X': (np.identity(2), np.array([0, 1])),
        'Z': (np.identity(2), np.array([1, 0])),
        'Y': (np.identity(2), np.array([1, 1])),
        'S': (np.array([[1, 0], [1, 1]]), np.zeros(2)),
        'SX': (np.array([[1, 1], [0, 1]]), np.array([0, 1])),
        'SY': (np.array([[0, 1], [1, 0]]), np.array([1, 0])),
        '-SY': (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
        '-SX': (np.array([[1, 1], [0, 1]]), np.array([0, 0])),
    }

    q1 = cirq.LineQubit(0)
    C1_reduced_q1_XZ = [cirq.Circuit(g) for g in
                        [cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=-0.5, z_exponent=0)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=1)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=-0.5, z_exponent=-0.5)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0.5, z_exponent=0.5)(q1),
                         cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=0.5)(q1)]
                        ]

    def __init__(self):
        self.generate_symplectic_matrices_R22()
        return

    def generate_symplectic_matrices_R22(self):
        skew = np.array([[0,1], [1, 0]])
        all_matrix_space = [np.array(a) for a in product([ar for ar in product([0,1], repeat=2)], repeat=2)]
        for mat in all_matrix_space:
            if np.array_equal((mat.T @ skew @ mat) % 2, skew):
                embedded_mat = np.zeros((4,4))
                embedded_mat[:2, : 2] = mat
                SingleQubitSymplacticCompilationData._symplectic_matrices.append(embedded_mat)
        return

    def assemble_data(self):
        result_container = {'symplectics': self._symplectic_matrices,
                            'phases': None,
                            'circuits': self.C1_reduced_q1_XZ,
                            'unitaries': None}
        return result_container




class RandomizedBenchmarkProgramBuilder:

    class QuaHelpers:
        def __init__(self):
            self.beta_lut = declare(int, value=[0, 0, 0, 0, 0, 0, 3, 1, 0, 1, 0, 3, 0, 3, 1, 0])
            self.product_lut = declare(int, value=[0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
            self.row_mask = declare(int, value=int(b'1111', 2))
            self.m2_transposed = declare(int, value=0)
            self.temp_bi = declare(int, value=0)

    class QuaThenHelpers(QuaHelpers):
        def __init__(self):
            super(RandomizedBenchmarkProgramBuilder.QuaThenHelpers, self).__init__()
            self.two_alpha_21 = declare(int, value=[0, 0, 0, 0])
            self.g1_ith_col = declare(int, value=0)
            self.temp_g1_dot_alpha2 = declare(int, value=0)

    class QuaInverseHelpers(QuaHelpers):
        def __init__(self):
            super(RandomizedBenchmarkProgramBuilder.QuaInverseHelpers, self).__init__()
            self.b = declare(int, value=[0, 0, 0, 0])
            self.temp_res_inv_a = declare(int, value=[0, 0, 0, 0])
            self.inv_g1_transpose = declare(int, value=0)
            self.temp_2alpha_plus_b = declare(int, value=[0, 0, 0, 0])

    _NUM_PAULIS_SINGLE = 4
    _NUM_PAULIS = _NUM_PAULIS_SINGLE ** 2
    _CLASS_C1_SIZE_SINGLE = 3
    _CLASS_C1_SIZE = _CLASS_C1_SIZE_SINGLE ** 2
    _CLASS_CNOT_SIZE_SINGLE = 5
    _CLASS_CNOT_SIZE = _CLASS_CNOT_SIZE_SINGLE ** 2
    _CLASS_SWAP_SIZE_SINGLE = 1 + 4
    _CLASS_SWAP_SIZE = _CLASS_SWAP_SIZE_SINGLE ** 2
    _CLASS_ISWAP_SIZE_SINGLE = 1 + 2
    _CLASS_ISWAP_SIZE = _CLASS_ISWAP_SIZE_SINGLE ** 2

    _NUM_CLIFFORDS = _CLASS_CNOT_SIZE + _CLASS_C1_SIZE + _CLASS_ISWAP_SIZE + _CLASS_SWAP_SIZE

    _AMPLITUDE_SET_DELAY = 36 # (4 dispatch + 5 read from memory = 9 cycles)
    _AMPLITUDE_TRACK_DELAY = 12
    _JUMP_BETWEEN_CASES_DELAY = 36

    def __init__(self, config_builder: conf.ConfigBuilder, num_experiments, num_qubits, bake_cnot=True, bake_swap=True,
                 min_sequence_length=1, max_sequence_length=None, from_initilized_list=False):
        self.num_qubits = num_qubits
        assert num_qubits in [1,2], "Only supports 1 or 2 qubits. "
        if num_qubits == 1:
            self._NUM_PAULIS = 4
            self._NUM_CLIFFORDS = 6
        self.config_builder = config_builder
        self.bake_cnot = bake_cnot
        self.bake_swap = bake_swap
        self.num_experiments = num_experiments
        self.min_seq_length = min_sequence_length
        self.max_sequence_length = self._NUM_CLIFFORDS if max_sequence_length is None else min(max_sequence_length, self._NUM_CLIFFORDS)
        self.symplectic_compilation_data = self.get_cliffords_data()
        self.paulis_compilation_data = self.get_paulis_data()
        self.from_initilized_list = from_initilized_list
        return

    def resolve_elem(self, qubit_num, channel, pulser):
        if qubit_num == "coupler":
            return "coupler"
        return f"qubit{qubit_num}_{channel}_p{pulser}"

    def get_paulis_data(self):
        p = Paulis(self.num_qubits)
        return {'alphas': p.paulis_alpha_vecs,
                'circuits': p.paulis_circuits}

    def get_cliffords_data(self):
        if self.num_qubits == 1:
            return SingleQubitSymplacticCompilationData().assemble_data()
        else:
            with open("symplectic_compilation_XZ.pkl", 'rb') as f:
                return pickle.load(f)

    def build_qua_program(self):
        py_pauli_values_list = self.py_setup_matrices_and_pulses("pauli")
        py_clifford_values_list = self.py_setup_matrices_and_pulses("clifford")
        m = self.config_builder.pulsers_per_qubit
        with program() as rb_program:
            I_0 = declare(fixed)
            I_1 = declare(fixed)
            Q_0 = declare(fixed)
            Q_1 = declare(fixed)
            then_helpers = [self.QuaThenHelpers() for i in range(m)]
            inv_helpers = [self.QuaInverseHelpers() for _ in range(m)]
            _rand = [Random(0) for _ in range(m)]
            random_tracker = [declare(int, value=[self._CLASS_C1_SIZE + 5, 10 + self._NUM_CLIFFORDS] * 20 + [0] * (736-20)) for _ in range(m)] # This will play a sequence of CNOTs
            random_tracker = [declare(int, value=[0, 10 + self._NUM_CLIFFORDS,
                                                  self._CLASS_C1_SIZE + 5, 10 + self._NUM_CLIFFORDS,
                                                  self._CLASS_CNOT_SIZE + self._CLASS_C1_SIZE + 1, 10 + self._NUM_CLIFFORDS,
                              self._CLASS_CNOT_SIZE + self._CLASS_C1_SIZE +
                                                  self._CLASS_SWAP_SIZE, 10 + self._NUM_CLIFFORDS] * 20 + [0] * (736-20)) for _ in range(m)] # This will play a sequence of CNOTs
            experiment_ind = [declare(int, value=0) for _ in range(m)]
            experiment_length = [declare(int, value=0) for _ in range(m)]

            current_clifford_ind = [declare(int, value=0) for _ in range(m)]
            number_of_gates = [declare(int, value=0) for _ in range(m)]
            current_g = [declare(int, value=0) for _ in range(m)]
            current_alpha = [declare(int, value=0) for _ in range(m)]
            prev_g = [declare(int, value=0) for _ in range(m)]
            prev_alpha = [declare(int, value=0) for _ in range(m)]
            temp_g12 = [declare(int, value=0) for _ in range(m)]
            temp_alpha12 = [declare(int, value=0) for _ in range(m)]

            lamb = [declare(int, value=18450) for _ in range(m)]
            N = [declare(int, value=0) for i in range(m)]

            pauli_mat_values_list = [declare(int, value=py_pauli_values_list) for _ in range(m)]
            clifford_mat_values_list = [declare(int, value=py_clifford_values_list) for _ in range(m)]

            inverse_g_ind = [declare(int, value=0) for _ in range(m)]
            inverse_alpha_ind = [declare(int, value=0) for _ in range(m)]

            frame_rotation_tracker = [[declare(int, value=0) for _ in range(5)] for _ in range(m)] # Track the phase of each element (qubit0, qubit1, coupler0, coupler1, coupler)
            gain_matrix_00 = [declare(fixed, value=[1.0, 0.0, -1.0, 0.0]) for _ in range(m)]
            gain_matrix_01= [declare(fixed, value=[0.0, -1.0, 0.0, 1.0]) for _ in range(m)]
            gain_matrix_10= [declare(fixed, value=[0.0, 1.0, 0.0, -1.0]) for _ in range(m)]
            gain_matrix_11 = [declare(fixed, value=[1.0, 0.0, -1.0, 0.0]) for _ in range(m)]

            self.matrix_reference = [gain_matrix_00, gain_matrix_01, gain_matrix_10, gain_matrix_11]
            self.amplitude_tracker = frame_rotation_tracker

            # single_amp_ref = declare(fixed, value=[1.0, 0.0, -1.0, 0.0])
            # self.single_amp_ref = single_amp_ref
            # align(*list(self.config_builder.config["elements"].keys()))
            for pulser in range(m):
                with for_(experiment_ind[pulser], cond=(experiment_ind[pulser] < self.num_experiments),
                              update=(experiment_ind[pulser]+1)):
                    # run the experiment several times
                    assign(experiment_length[pulser],
                           _rand[pulser].rand_int(self.max_sequence_length - self.min_seq_length) + self.min_seq_length)
                    assign(number_of_gates[pulser], 2 * (experiment_length[pulser] + 1))

                    if not self.from_initilized_list:
                        with for_(current_clifford_ind[pulser], init=0,
                                  cond=(current_clifford_ind[pulser] < 2 * experiment_length[pulser]),
                                  update=(current_clifford_ind[pulser] + 2)):
                            save(0, "start_calc")
                            assign(N[pulser], _rand[pulser].rand_int(self._NUM_CLIFFORDS * self._NUM_PAULIS))
                            assign(random_tracker[pulser][current_clifford_ind[pulser]], N[pulser] / self._NUM_PAULIS)
                            assign(random_tracker[pulser][current_clifford_ind[pulser] + 1],
                                   (N[pulser] & (self._NUM_PAULIS -1)) + self._NUM_CLIFFORDS)
                            # assign(N[pulser], random_tracker[current_clifford_ind] * random_tracker[current_clifford_ind + 1])
                            assign(current_g[pulser], clifford_mat_values_list[pulser][N[pulser] / self._NUM_PAULIS])
                            assign(current_alpha[pulser], pauli_mat_values_list[pulser][N[pulser] & (self._NUM_PAULIS-1)])
                            # Compute the g and alpha
                            qua_simple_tableau.then(current_g[pulser], current_alpha[pulser],
                                                    prev_g[pulser], prev_alpha[pulser],
                                                    temp_g12[pulser], temp_alpha12[pulser], then_helpers[pulser])
                            assign(prev_g[pulser], current_g[pulser])
                            assign(prev_alpha[pulser], current_alpha[pulser])
                            assign(current_g[pulser], temp_g12[pulser])
                            assign(current_alpha[pulser], temp_alpha12[pulser])
                            save(1, "end_calc")
                        # Compute the inverse
                        qua_simple_tableau.inverse(current_g[pulser], current_alpha[pulser], prev_g[pulser],
                                                   prev_alpha[pulser], lamb[pulser], inv_helpers[pulser])
                        self.qua_insert_clifford_2_g_index(prev_g[pulser], clifford_mat_values_list[pulser],
                                                           inverse_g_ind[pulser])
                        self.qua_insert_clifford_2_alpha_index(prev_alpha[pulser], pauli_mat_values_list[pulser],
                                                               inverse_alpha_ind[pulser])
                        assign(random_tracker[pulser][current_clifford_ind[pulser]],
                               (inverse_g_ind[pulser]*inverse_alpha_ind[pulser]) / self._NUM_PAULIS)
                        assign(random_tracker[pulser][current_clifford_ind[pulser] + 1],
                               (inverse_alpha_ind[pulser] & (self._NUM_PAULIS - 1)) + self._NUM_CLIFFORDS)
                    # save(2, "end_all_calc")
                    # ================================================================================================ #
                    # ================================================================================================ #
                    # ================================================================================================ #

                    # with for_():
                    #     with for_init_():
                    #         assign(current_clifford_ind[pulser], 0)
                    #         # assign(N[pulser], random_tracker[pulser][0])
                    #     for_cond(current_clifford_ind[pulser] < number_of_gates[pulser])
                    #     with for_update_():
                    #         assign(current_clifford_ind[pulser], current_clifford_ind[pulser] + 1)
                    #         # assign(N[pulser], random_tracker[pulser][current_clifford_ind[pulser]])
                    #     with for_body_():
                    with for_(current_clifford_ind[pulser], 0, current_clifford_ind[pulser] < number_of_gates[pulser], current_clifford_ind[pulser] + 1):
                        with switch_(random_tracker[pulser][current_clifford_ind[pulser]], unsafe=True):
                            for i in range(self._NUM_CLIFFORDS + self._NUM_PAULIS):
                                with case_(i):
                                    self._resolve_case(i, on_pulser=pulser)
                    if pulser == m - 1:
                        # wait((self.config_builder.pi_pulse_len + self._AMPLITUDE_SET_DELAY) //4, self.resolve_elem(0, "xy", m-1))
                        # wait((self.config_builder.pi_pulse_len + self._AMPLITUDE_SET_DELAY) //4, self.resolve_elem(1, "xy", m-1))
                        measure('readout_pulse',
                            self.resolve_elem(0, "xy", m-1), None,
                            demod.full("integ_weights_cos", I_0, "out1"),
                            demod.full("integ_weights_sin", Q_0, "out1"))
                        measure('readout_pulse',
                                self.resolve_elem(1, "xy", m-1), None,
                                demod.full("integ_weights_cos", I_1, "out2"),
                                demod.full("integ_weights_sin", Q_1, "out2"))
                    else:
                        wait((self.config_builder.readout_len + 184) //4,
                             *[self.resolve_elem(i, "xy", pulser) for i in range(2)])
        return rb_program


    def _resolve_case(self, i, on_pulser=None):
        if i < self._CLASS_C1_SIZE:
            self._insert_case_single_qubit_gates(*divmod(i, self._CLASS_C1_SIZE_SINGLE), pulser=on_pulser)
        elif self._CLASS_C1_SIZE <= i < self._CLASS_C1_SIZE + self._CLASS_CNOT_SIZE:
            i -= self._CLASS_C1_SIZE
            self._insert_case_CNOT(*divmod(i // 9, self._CLASS_C1_SIZE_SINGLE), i // 3 % 3, i % 3, pulser=on_pulser)
        elif self._CLASS_C1_SIZE + self._CLASS_CNOT_SIZE <= i < self._CLASS_C1_SIZE + self._CLASS_CNOT_SIZE + self._CLASS_SWAP_SIZE:
            i -= (self._CLASS_C1_SIZE + self._CLASS_CNOT_SIZE)
            self._insert_case_ISWAP(*divmod(i // 9, self._CLASS_C1_SIZE_SINGLE), i//3 % 3, i % 3, pulser=on_pulser)
        elif self._CLASS_C1_SIZE + self._CLASS_CNOT_SIZE + self._CLASS_SWAP_SIZE <= i < self._NUM_CLIFFORDS:
            i -= (self._CLASS_C1_SIZE + self._CLASS_CNOT_SIZE + self._CLASS_SWAP_SIZE)
            self._insert_case_SWAP(*divmod(i // 9, self._CLASS_C1_SIZE_SINGLE), pulser=on_pulser)
        elif i >= self._NUM_CLIFFORDS:
            i -= self._NUM_CLIFFORDS
            self._insert_case_pauli(*divmod(i, self._NUM_PAULIS_SINGLE), pulser=on_pulser)
        return

    def __embed_c1_play(self, q0_element, q1_element, matrix_ref, amp_tracker, c1_0, c1_1):
        play(f"c1_{c1_0}", q0_element)
        play(f"c1_{c1_1}" ,q1_element)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_0}"][2], q0_element)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_1}"][2], q1_element)
        # assign(amp_tracker[0], (amp_tracker[0] + conf.c1_phxz[f"c1_{c1_0}"][2]) & 3)
        # assign(amp_tracker[1], (amp_tracker[1] + conf.c1_phxz[f"c1_{c1_1}"][2]) & 3)


    def _insert_case_single_qubit_gates(self, c1_0, c1_1, pulser=None):
        ''' single qubit gate will be played in pulser 0. '''
        # save(3, "case0")
        play_pulser =  self.config_builder._GATES_TO_PULSERS["C1"]
        amp_tracker = self.amplitude_tracker[pulser]
        matrix_ref = [elem[pulser] for elem in self.matrix_reference]
        if pulser != play_pulser:
            assign(amp_tracker[0], (amp_tracker[0] + conf.c1_phxz[f"c1_{c1_0}"][2]) & 3)
            assign(amp_tracker[1], (amp_tracker[1] + conf.c1_phxz[f"c1_{c1_1}"][2]) & 3)
            wait_time = self.config_builder.pi_pulse_len+self._AMPLITUDE_SET_DELAY - self._JUMP_BETWEEN_CASES_DELAY
            if wait_time >= 16:
                wait(wait_time // 4, self.resolve_elem(0, "xy", pulser), self.resolve_elem(1, "xy", pulser))
        else:
            self.__embed_c1_play(self.resolve_elem(0, "xy", play_pulser), self.resolve_elem(1, "xy", play_pulser),
                                 matrix_ref, amp_tracker, c1_0, c1_1)

    def _insert_case_pauli(self, p0, p1, pulser=None):
        ''' pauli gate will be played in pulser 1. '''
        # save(3, "case1")
        play_pulser =  self.config_builder._GATES_TO_PULSERS["PAULI"]
        amp_tracker = self.amplitude_tracker[pulser]
        matrix_ref = [elem[pulser] for elem in self.matrix_reference]
        if pulser != play_pulser:
            assign(amp_tracker[0], (amp_tracker[0] + conf.pauli_phxz[f"pauli_{p0}"][2]) & 3)
            assign(amp_tracker[1], (amp_tracker[1] + conf.pauli_phxz[f"pauli_{p1}"][2]) & 3)
            wait_time = self.config_builder.pi_pulse_len + self._AMPLITUDE_SET_DELAY - self._JUMP_BETWEEN_CASES_DELAY
            if wait_time >= 16:
                wait(wait_time // 4, self.resolve_elem(0, "xy", pulser), self.resolve_elem(1, "xy", pulser))
        else:
            q1_xy = self.resolve_elem(0, "xy", play_pulser)
            q0_xy = self.resolve_elem(1, "xy", play_pulser)
            play(f"pauli_{p0}" , q0_xy)
            play(f"pauli_{p1}", q1_xy)
            frame_rotation_2pi(conf.pauli_phxz[f"pauli_{p0}"][2], q0_xy)
            frame_rotation_2pi(conf.pauli_phxz[f"pauli_{p1}"][2], q1_xy)

            # assign(amp_tracker[0], (amp_tracker[0] + conf.pauli_phxz[f"pauli_{p0}"][2]) & 3)
            # assign(amp_tracker[1], (amp_tracker[1] + conf.pauli_phxz[f"pauli_{p1}"][2]) & 3)


    def __embed_s0s1_gates(self, q0_element, q1_element, matrix_ref, amp_tracker, s1_0, s1_1):
        play(f"s1_{s1_0}" ,
             q0_element)
        play(f"s1_{s1_1}" ,
             q1_element)
        frame_rotation_2pi( conf.s1_phxz[f"s1_{s1_0}"][2], q0_element)
        frame_rotation_2pi( conf.s1_phxz[f"s1_{s1_1}"][2], q1_element)


    def __embed_coupling(self):
        play("coupler_tone", "qubit0_z")
        play("coupler_tone", "qubit1_z")
        play("coupler_tone", "coupler")

    def _insert_case_CNOT(self, c1_0, c1_1, s1_0, s1_1, pulser=None):
        # save(3, "case2")
        play_pulser = self.config_builder._GATES_TO_PULSERS["CNOT"]
        amp_tracker = self.amplitude_tracker[pulser]
        matrix_ref = [elem[pulser] for elem in self.matrix_reference]
        if pulser != play_pulser:
            # TODO: This amplitude tracker also needs to take care the effective phase resulting from the cnot
            assign(amp_tracker[0], (amp_tracker[0] + conf.s1_phxz[f"s1_{s1_0}"][2] + conf.c1_phxz[f"c1_{c1_0}"][2]) & 3)
            assign(amp_tracker[1], (amp_tracker[1] + conf.s1_phxz[f"s1_{s1_1}"][2] + conf.c1_phxz[f"c1_{c1_1}"][2]) & 3)
            wait_time = self.config_builder.cnot_length +  \
                        (self.config_builder.pi_pulse_len * 2 + self._AMPLITUDE_SET_DELAY)-\
                        self._JUMP_BETWEEN_CASES_DELAY
            if wait_time >= 16:
                wait(wait_time // 4, self.resolve_elem(0, "xy", pulser), self.resolve_elem(1, "xy", pulser))
        else:
            q0_xy = self.resolve_elem(0, "xy", play_pulser)
            q1_xy = self.resolve_elem(1, "xy", play_pulser)
            self.__embed_c1_play(q0_xy, q1_xy, matrix_ref, amp_tracker, c1_0, c1_1)
            # wait(self.config_builder.pi_pulse_len // 4, "qubit0_z", "qubit1_z", "coupler")
            align(q0_xy, q1_xy, "qubit0_z", "qubit1_z", "coupler")
            if self.bake_cnot:
                play("cnot", q0_xy)
                play("cnot", q1_xy)
                play("cnot", "qubit0_z")
                play("cnot", "qubit1_z")
                play("cnot", "coupler")
            else:
                play("cnot_0_0", q0_xy)
                play("cnot_0_1", q1_xy)
                self.__embed_coupling()
                play("cnot_2_0",q0_xy)
                play("cnot_2_1", q1_xy)
                self.__embed_coupling()
                play("cnot_4_0", q0_xy)
                play("cnot_4_1", q1_xy)
            # TODO: Need another frame rotation
            self.__embed_s0s1_gates(q0_xy, q1_xy, matrix_ref, amp_tracker, s1_0, s1_1)
            # wait((self.config_builder.pi_pulse_len) // 4, "qubit0_z", "qubit1_z", "coupler")

    def _insert_case_ISWAP(self, c1_0, c1_1, s1_0, s1_1, pulser=None):
        # save(3, "case3")
        play_pulser =  self.config_builder._GATES_TO_PULSERS["ISWAP"]
        amp_tracker = self.amplitude_tracker[pulser]
        matrix_ref = [elem[pulser] for elem in self.matrix_reference]
        if pulser != play_pulser:
            # TODO: This amplitude tracker also needs to take care the effective phase resulting from the swap
            assign(amp_tracker[0], (amp_tracker[0] + conf.s1_phxz[f"s1_{s1_0}"][2] + conf.c1_phxz[f"c1_{c1_0}"][2]) & 3)
            assign(amp_tracker[1], (amp_tracker[1] + conf.s1_phxz[f"s1_{s1_1}"][2] + conf.c1_phxz[f"c1_{c1_1}"][2]) & 3)
            wait_time = self.config_builder.cz_pulse_len * 2 +  \
                        self.config_builder.pi_pulse_len * 2 + self._AMPLITUDE_SET_DELAY + self._AMPLITUDE_TRACK_DELAY -\
                        self._JUMP_BETWEEN_CASES_DELAY
            if wait_time >= 16:
                wait(wait_time // 4, self.resolve_elem(0, "xy", pulser), self.resolve_elem(1, "xy", pulser))
        else:
            q0_xy = self.resolve_elem(0, "xy", play_pulser)
            q1_xy = self.resolve_elem(1, "xy", play_pulser)
            self.__embed_c1_play(q0_xy, q1_xy, matrix_ref, amp_tracker, c1_0, c1_1)
            # wait((self.config_builder.pi_pulse_len ) // 4, "qubit0_z", "qubit1_z", "coupler")
            align(q0_xy, q1_xy, "qubit0_z", "qubit1_z", "coupler", q0_xy, q1_xy)
            self.__embed_coupling()
            self.__embed_coupling()
            wait(self.config_builder.cz_pulse_len * 2 // 4, q0_xy, q1_xy)
            # TODO: Need another frame rotation
            self.__embed_s0s1_gates(q0_xy, q1_xy, matrix_ref, amp_tracker, s1_0, s1_1)
            wait((self.config_builder.pi_pulse_len) // 4, "qubit0_z", "qubit1_z",
                 "coupler")

    def _insert_case_SWAP(self, c1_0, c1_1, pulser=None):
        # save(3, "case4")
        play_pulser =  self.config_builder._GATES_TO_PULSERS["SWAP"]
        amp_tracker = self.amplitude_tracker[pulser]
        matrix_ref = [elem[pulser] for elem in self.matrix_reference]
        if pulser != play_pulser:
            assign(amp_tracker[0], (amp_tracker[0] + conf.c1_phxz[f"c1_{c1_0}"][2]) & 3)
            assign(amp_tracker[1], (amp_tracker[1] + conf.c1_phxz[f"c1_{c1_1}"][2]) & 3)
            wait_time = self.config_builder.swap_length + self._AMPLITUDE_SET_DELAY +  \
                        self.config_builder.pi_pulse_len + self._AMPLITUDE_TRACK_DELAY +8 -\
                        self._JUMP_BETWEEN_CASES_DELAY
            if wait_time >= 16:
                wait(wait_time // 4, self.resolve_elem(0, "xy", pulser), self.resolve_elem(1, "xy", pulser))
        else:
            q0_xy = self.resolve_elem(0, "xy", play_pulser)
            q1_xy = self.resolve_elem(1, "xy", play_pulser)
            self.__embed_c1_play(q0_xy, q1_xy, matrix_ref, amp_tracker, c1_0, c1_1)
            align(q0_xy, q1_xy, "qubit0_z", "qubit1_z", "coupler", q0_xy, q1_xy)
            # wait((self.config_builder.pi_pulse_len + self._AMPLITUDE_SET_DELAY ) // 4 - 8, "qubit0_z", "qubit1_z", "coupler")
            if self.bake_swap:
                play("swap", q0_xy)
                play("swap", q1_xy)
                play("swap", "qubit0_z")
                play("swap", "qubit1_z")
                play("swap", "coupler")
            else:
                play("swap_0_0", q0_xy)
                play("swap_0_1", q1_xy)
                self.__embed_coupling()
                play("swap_2_0", q0_xy)
                play("swap_2_1", q1_xy)
                self.__embed_coupling()
                play("swap_4_0", q0_xy)
                play("swap_4_1", q1_xy)
                self.__embed_coupling()
                play("swap_6_0", q0_xy)
                play("swap_6_1", q1_xy)

            # TODO: Add another frame rotation


    def qua_insert_clifford_2_g_index(self, c, clifford_vals_list, ind_res):
        Utils.qua_add_binary_search_block(c,clifford_vals_list , self._NUM_CLIFFORDS, ind_res)

    def qua_insert_clifford_2_alpha_index(self, c, paulis_vals_list, ind_res):
        Utils.qua_add_binary_search_block(c,paulis_vals_list , self._NUM_PAULIS, ind_res)

    def py_setup_matrices_and_pulses(self, source_type) -> List:
        source_dict_mat_to_pulse = []
        if source_type == 'pauli':
            for ind, vec in enumerate(self.paulis_compilation_data["alphas"]):
                vec_int = Utils.py_bin_vector_to_int32(vec)
                source_dict_mat_to_pulse.append(vec_int)
        else:
            for ind, mat in enumerate(self.symplectic_compilation_data["symplectics"]):
                mat_int = Utils.py_bin_matrix_to_int32(mat.astype(int))
                source_dict_mat_to_pulse.append(mat_int)

        values_list = sorted(list(source_dict_mat_to_pulse))

        return values_list


class Test:

    def __init__(self, config_builder: conf.ConfigBuilder):
        self.qmm = QuantumMachinesManager(host="localhost", port=9510)
        self.config_builder = config_builder
        return

    def run_simulation(self, _program, vars_to_save):
        job_sim = self.qmm.simulate(self.config_builder.config, _program, SimulationConfig(100000))
        res = job_sim.result_handles
        while not res.wait_for_all_values():
            time.sleep(0.5)
            print(".", end="")

        qubit_graphs = 2 * (1 if self.config_builder.combine_pulsers_of_qubits else self.config_builder.pulsers_per_qubit)
        num_of_graphs = qubit_graphs + 3
        i_to_ports = {i: [f"{i*2 +1}", f"{i*2+2}"] for i in range(qubit_graphs)}
        i_to_ports.update({i + (qubit_graphs): [f"{i + qubit_graphs * 2 + 1}"] for i in range(3)})
        if qubit_graphs == 2:
            i_to_title = {0: "qubit0_xy", 1: "qubit1_xy", 2: "qubit0_z", 3: "qubit1_z", 4: "coupler"}
        else:
            i_to_title = {0: "qubit0_xy_p0\n(C1)", 1: "qubit0_xy_p1\n(Pauli)", 2: "qubit1_xy_p0\n(C1)",
                          3: "qubit1_xy_p1\n(Pauli)", 4: "qubit0_z", 5: "qubit1_z", 6: "coupler"}
        min_lim, max_lim = job_sim.get_simulated_samples().con1.analog["1"].shape[0], 0
        for s in job_sim.get_simulated_samples().con1.analog.values():
            non_zero = np.where(s != 0)
            if len(non_zero[0]) == 0:
                continue
            min_lim = min(min_lim, non_zero[0].min())
            max_lim = max(max_lim, non_zero[0].max())
        for i in range(num_of_graphs):
            p = pplot.subplot(num_of_graphs,1,i+1)
            pplot.title(i_to_title[i], x=-0.03, y=0.9, fontdict={'horizontalalignment': 'right', 'fontsize':10, 'verticalalignment': 'top'})
            pplot.xlim([min_lim - min_lim*0.05, max_lim + max_lim * 0.15])
            pplot.ylim([-0.75, 0.75])
            pplot.xticks(fontsize=6, y=0.05)
            pplot.yticks(fontsize=6, x=0)
            pplot.ylabel(ylabel='', fontdict=dict(fontsize=8), visible=False)
            p.xaxis.set_tick_params(length=1)
            p.yaxis.set_tick_params(length=1)
            job_sim.get_simulated_samples().con1.plot(i_to_ports[i])
            if i < qubit_graphs:
                pplot.legend([f"analog_{i*2+1}", f"analog_{i*2+2}"], loc='upper right')
            else:
                pplot.legend([f"analog_{i - qubit_graphs + qubit_graphs * 2 + 1}"], loc='upper right')

        pplot.subplots_adjust(bottom=0.05, top=0.95, left=0.15)
        pplot.show()
        if isinstance(vars_to_save, list):
            final_res = {var_name: res.__getattribute__(var_name).fetch_all() for var_name in vars_to_save}
        elif isinstance(vars_to_save, str):
            final_res = {vars_to_save: res.__getattribute__(vars_to_save).fetch_all()[0]}
        else:
            final_res = None

        if hasattr(res, "program_start_ts"):
            final_res["program_start_ts"] = res.__getattribute__("program_start_ts").fetch_all()[0]

        return final_res

    def simulate_bs(self, target_list, target_list_size, val_to_search):
        ind = (target_list_size-1)//2
        right = target_list_size-1
        left = 0
        result_ind = -1
        while(left <= right):
            if (val_to_search < target_list[ind]):
                right = ind -1
            elif( val_to_search > target_list[ind]):
                left = ind + 1
            else:
                result_ind = ind
                right=left-1
            ind = int((right + left) /2)

        return result_ind

    def test_binary_search(self):
        max_int = 2**15
        list_size = 100
        value_list = sorted(random.sample(range(max_int), k=list_size))

        for i in range(10):
            target_val = value_list[random.randint(0, list_size-1)]
            # target_val = value_list[i]
            print(f"simulator ref = {self.simulate_bs(value_list, list_size, target_val)}")
            with program() as p:

                qua_list = declare(int, value=value_list)
                qua_target_val = declare(int, value=target_val)
                qua_res_ind = declare(int, value=0)
                Utils.qua_add_binary_search_block(qua_target_val, qua_list, list_size, qua_res_ind)
                save(qua_res_ind, "qua_res_ind")

            ref_ind = value_list.index(target_val)
            # print(f"{ref_ind=}/{list_size}")
            print(self.run_simulation(p, ["qua_res_ind"]))
            # if qua_res == ref_ind:
            #     print("Passed")
            # else:
            #     print(f"failed ({ref_ind=})")

        return

    def test_all_simple_tablue(self):

        with program() as p:
            g1 = declare(int)
            g2 = declare(int)
            alpha1 = declare(int)
            alpha2 = declare(int)
            g12 = declare(int)
            alpha12 = declare(int)
            then_help = RandomizedBenchmarkProgramBuilder.QuaThenHelpers()
            inv_help = RandomizedBenchmarkProgramBuilder.QuaInverseHelpers()
            inv_g = declare(int)
            inv_alpha = declare(int)
            lamb = declare(int)
            qua_simple_tableau.then(g1, alpha1, g2, alpha2, g12, alpha12, then_help)
            qua_simple_tableau.inverse(g12, alpha12, inv_g, inv_alpha, lamb, inv_help)

        self.run_simulation(p, None)
        return

    def test_general(self):
        with program() as p:
            frame_rotation_2pi(0.25, "qubit0_xy_p0")
            play("c1_0", "qubit0_xy_p0")
            frame_rotation_2pi(-0.25, "qubit0_xy_p0")
            play("c1_0", "qubit0_xy_p0")


        print(self.run_simulation(p, []))
        return


def run_experiment(num_qubits, num_experiments=1, min_sequence_length=1, max_sequence_length=None, from_initilized_list=False):
    config_builder = conf.ConfigBuilder(pulsers_per_qubit=1)
    config_builder.build()
    rb = RandomizedBenchmarkProgramBuilder(config_builder, num_experiments=num_experiments, num_qubits=num_qubits,
                                           min_sequence_length=min_sequence_length,
                                           max_sequence_length=max_sequence_length,
                                           from_initilized_list=from_initilized_list)
    p = rb.build_qua_program()
    tester = Test(config_builder)
    d = tester.run_simulation(p, ["start_calc", "end_calc" , "end_all_calc","case0","case1","case2", "case3", "case4"])
    for k,v in d.items():
        print(f"{k:<20}, {[i for i in v]}")


if __name__ == '__main__':
    # config_builder = conf.ConfigBuilder(pulsers_per_qubit=1)
    # config_builder.build()
    # tester = Test(config_builder)
    # tester.test_general()
    # Test()
    run_experiment(2,1, 3,4, from_initilized_list=True)
    # run_experiment(2,1, 2,3)

    # rb.py_setup_gates_unique_ids()
    # rb.py_setup_translation_to_gate()
    # tester.test_all_simple_tablue()
    # tester.test_general()
    # tester.test_binary_search()
    # import pickle
    # with open("symplectic_compilation_XZ.pkl", 'rb') as f:
    #     s = pickle.load(f)
    # print()
    # check_axz(s["circuits"])
    # print()