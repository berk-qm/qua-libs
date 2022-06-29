import math
import pickle
from itertools import combinations, product
from typing import Tuple, List, Dict
import numpy as np
import random
import matplotlib.pyplot as pplot

import cirq
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from matplotlib import pyplot as plt

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
    # identity = np.identity(2, dtype=np.int)
    # sigma_x = np.array([[0,1], [1,0]])
    # sigma_y = np.array([[0, -1],[1,0]])
    # sigma_z = np.array([[1,0], [0,-1]])
    # all_vals = []
    #
    # for comb in product([sigma_x, sigma_z, sigma_y, identity], repeat=2):
    #     all_vals.append(np.kron(comb[0], comb[1]))

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
    _c1_ops = [
        ('I',),  # XZ ++
        ('X',),  # XZ +-
        ('Z',),  # XZ -+
        ('Y',),  # XZ --

        ('-SX',),  # XY ++
        ('SX',),  # XY +-
        ('Y', 'SX'),  # XY -+
        ('Y', '-SX'),  # XY --

        ('X', '-SY'),  # ZX ++
        ('-SY',),  # ZX +-
        ('SY',),  # ZX -+
        ('X', 'SY'),  # ZX --

        ('-SX', '-SY'),  # ZY ++
        ('SX', '-SY'),  # ZY +-
        ('-SX', 'SY'),  # ZY -+
        ('SX', 'SY'),  # ZY --

        ('SY', 'SX'),  # YX ++
        ('-SY', '-SX'),  # YX +-
        ('SY', '-SX'),  # YX -+
        ('-SY', 'SX'),  # YX --

        ('-SX', 'SY', 'SX'),  # YZ ++
        ('SX', 'SY', 'SX'),  # YZ +-
        ('-SX', '-SY', 'SX'),  # YZ -+
        ('-SX', 'SY', '-SX'),  # YZ --
    ]

    _gate_from_op = {
        'I': cirq.I,
        'X': cirq.X,
        'Y': cirq.Y,
        'Z': cirq.Z,
        'SX': cirq.X ** 0.5,
        'SY': cirq.Y ** 0.5,
        '-SX': cirq.X ** -0.5,
        '-SY': cirq.Y ** -0.5,
    }

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
    '''
    Terminology:
        Ciruits -   Describes the whole system (all qubits) gates at all the times.
        Moment -    A point at time of operations on all qubits
        Operation - A gate played on single time on single qubit.
        Gate -      A combination of pulses creating some logical gate.

    How circuits are stored:
        - Each operation has unique id. (for now, number of operation must be <=15. It can be extended, but thats not
            supported now).
        - Per Qubit:
            - All single circuits moments (of qubit) are stored in a single integer (number of moments should be <=8.
                this is also can be extended, but not supported now).
            - Each 4 bit is a moment. The number corresponds to the operation unique id.
            - For decoding - take the 4 bit, translate them to the operation -> if operation is phased Xz, it is also
                stored inside single int represeting all the exponents.
                otherwise -> It is ISWAP so play ISWAP.

    '''

    class QuaHelpers:
        def __init__(self):
            self.a_exponent = declare(fixed, value=0)
            self.x_exponent = declare(fixed, value=0)
            self.z_exponent = declare(fixed, value=0)
            self.moment_ind_helper = declare(int, value=0)
            self.encoded_single_moment = declare(int, value=0)
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

    _CLIFFORD_G_MASK = int('1'*16, 2)
    _CLIFFORD_ALPHA_MASK = int('1'*4, 2) << 16
    _NUM_CLIFFORDS = 720
    _NUM_PAULIS = 16

    def __init__(self, num_experiments, num_qubits,  min_sequence_length=1, max_sequence_length=None):
        self.num_qubits = num_qubits
        assert num_qubits in [1,2], "Only supports 1 or 2 qubits. "
        if num_qubits == 1:
            self._NUM_PAULIS = 4
            self._NUM_CLIFFORDS = 6
        self.num_experiments = num_experiments
        self.min_seq_length = min_sequence_length
        self.max_sequence_length = self._NUM_CLIFFORDS if max_sequence_length is None else min(max_sequence_length, self._NUM_CLIFFORDS)
        self.symplectic_compilation_data = self.get_cliffords_data()
        self.paulis_compilation_data = self.get_paulis_data()
        self.__num_phXZ_gates = 0
        self.__num_gates_options = 0
        self.__gate_to_ind_map = {}
        self.__encoded_gates_vals_list = []
        return

    def get_paulis_data(self):
        p = Paulis(self.num_qubits)
        return {'alphas': p.paulis_alpha_vecs,
                'circuits': p.paulis_circuits}

    def get_cliffords_data(self):
        if self.num_qubits == 1:
            return SingleQubitSymplacticCompilationData().assemble_data()
        else:
            with open("../symplectic_compilation_XZ.pkl", 'rb') as f:
                return pickle.load(f)

    def build_qua_program(self):
        self.py_setup_gates_unique_ids()
        py_pauli_values_list, py_pauli_pulses_list = self.py_setup_matrices_and_pulses("pauli")
        py_clifford_values_list, py_clifford_pulses_list = self.py_setup_matrices_and_pulses("clifford")

        with program() as rb_program:
            helpers = self.QuaHelpers()
            then_helpers = self.QuaThenHelpers()
            inv_helpers = self.QuaInverseHelpers()
            _rand = Random()
            experiment_ind = declare(int, value=0)
            experiment_length = declare(int, value=0)

            current_clifford_ind = declare(int, value=0)
            current_g = declare(int, value=0)
            current_alpha = declare(int, value=0)
            prev_g = declare(int, value=0)
            prev_alpha = declare(int, value=0)
            g12_temp_res = declare(int, value=0)
            alpha12_temp_res = declare(int, value=0)

            lamb = declare(int, value=18450)
            clifford_rand_ind = declare(int, value=0)
            pauli_rand_ind = declare(int, value=0)

            encoded_gates_list = declare(int, value=self.__encoded_gates_vals_list)
            pauli_mat_values_list = declare(int, value=py_pauli_values_list)
            clifford_decoded_pulse_seq_q0 = declare(int, value=0)
            pauli_decoded_pulse_seq_q0 = declare(int, value=0)
            pauli_pulses_list_q0 = declare(int, value=py_pauli_pulses_list[0])
            clifford_mat_values_list = declare(int, value=py_clifford_values_list)
            clifford_pulses_list_q0 = declare(int, value=py_clifford_pulses_list[0])

            clifford_decoded_pulse_seq_q1 = None
            pauli_decoded_pulse_seq_q1 = None
            if self.num_qubits == 2:
                clifford_decoded_pulse_seq_q1 = declare(int, value=0)
                pauli_decoded_pulse_seq_q1 = declare(int, value=0)
                pauli_pulses_list_q1 = declare(int, value=py_pauli_pulses_list[1])
                clifford_pulses_list_q1 = declare(int, value=py_clifford_pulses_list[1])

            inverse_g_ind = declare(int, value=0)
            inverse_alpha_ind = declare(int, value=0)

            with for_(experiment_ind, cond=(experiment_ind < self.num_experiments), update=(experiment_ind+1)):
                # run the experiment several times
                assign(experiment_length,
                       _rand.rand_int(self.max_sequence_length - self.min_seq_length) + self.min_seq_length)
                save(experiment_length, 'exp_length')
                with for_(current_clifford_ind, init=0, cond=(current_clifford_ind < experiment_length),
                          update=(current_clifford_ind+1)):
                    # Loop to play clifford pulses
                    # Get random indices
                    # assign(clifford_rand_ind, _rand.rand_int(self._NUM_CLIFFORDS))
                    assign(clifford_rand_ind, current_clifford_ind)
                    # assign(pauli_rand_ind, _rand.rand_int(self._NUM_PAULIS))
                    assign(pauli_rand_ind, 1)
                    # Set the pulses to be played
                    assign(clifford_decoded_pulse_seq_q0, clifford_pulses_list_q0[clifford_rand_ind])
                    assign(pauli_decoded_pulse_seq_q0, pauli_pulses_list_q0[pauli_rand_ind])
                    if self.num_qubits == 2:
                        assign(clifford_decoded_pulse_seq_q1, clifford_pulses_list_q1[clifford_rand_ind])
                        assign(pauli_decoded_pulse_seq_q1, pauli_pulses_list_q1[pauli_rand_ind])
                    # Set the current g and alpha
                    assign(current_g, clifford_mat_values_list[clifford_rand_ind])
                    assign(current_alpha, pauli_mat_values_list[pauli_rand_ind])
                    # -Insert play
                    self.qua_insert_play_circuit(clifford_decoded_pulse_seq_q0, clifford_decoded_pulse_seq_q1,
                                                 encoded_gates_list, helpers)
                    self.qua_insert_play_circuit(pauli_decoded_pulse_seq_q0, pauli_decoded_pulse_seq_q1,
                                                 encoded_gates_list, helpers)
                    # -Multiply
                    qua_simple_tableau.then(current_g, current_alpha, prev_g, prev_alpha,
                                            g12_temp_res, alpha12_temp_res, then_helpers)
                    assign(prev_g, current_g)
                    assign(prev_alpha, current_alpha)
                    assign(current_g, g12_temp_res)
                    assign(current_alpha, alpha12_temp_res)

                qua_simple_tableau.inverse(current_g, current_alpha, prev_g, prev_alpha, lamb, inv_helpers)
                self.qua_insert_clifford_2_g_index(prev_g, clifford_mat_values_list, inverse_g_ind)
                self.qua_insert_clifford_2_alpha_index(prev_alpha, pauli_mat_values_list, inverse_alpha_ind)

                assign(clifford_decoded_pulse_seq_q0, clifford_pulses_list_q0[inverse_g_ind])
                assign(pauli_decoded_pulse_seq_q0, pauli_pulses_list_q0[inverse_alpha_ind])
                if self.num_qubits == 2:
                    assign(clifford_decoded_pulse_seq_q1, clifford_pulses_list_q1[inverse_g_ind])
                    assign(pauli_decoded_pulse_seq_q1, pauli_pulses_list_q1[inverse_alpha_ind])

                self.qua_insert_play_circuit(clifford_decoded_pulse_seq_q0, clifford_decoded_pulse_seq_q1,
                                             encoded_gates_list, helpers)
                self.qua_insert_play_circuit(pauli_decoded_pulse_seq_q0, pauli_decoded_pulse_seq_q1,
                                             encoded_gates_list, helpers)

                save(clifford_decoded_pulse_seq_q0, "c")

        return rb_program

    def qua_insert_play_circuit(self, pulse_seq_q0, pulse_seq_q1, encoded_values_list, helpers: QuaHelpers):
        # Decode the pulses (moments)
        assign(helpers.moment_ind_helper, 0)
        save(pulse_seq_q0, "pulse_seq_q0")
        with while_((((pulse_seq_q0 & (15 << (4*helpers.moment_ind_helper))) >> (4*helpers.moment_ind_helper)) < 15)):
            with if_(((pulse_seq_q0 & (15 << (4*helpers.moment_ind_helper))) >>
                      (4*helpers.moment_ind_helper)) < self.__num_phXZ_gates):
            # Current moment is phXZ:
            # Translate the exponent X2
            # TODO: Seperate encoded single moment so compiler can parallelize the play to each element.
                assign(helpers.encoded_single_moment,
                       encoded_values_list[((pulse_seq_q0 & (15 << (4*helpers.moment_ind_helper))) >> (4*helpers.moment_ind_helper))])
                # save(helpers.encoded_single_moment, "encoded_single_moment")
                self.qua_insert_decode_exponents(helpers.encoded_single_moment, helpers)
                # save(helpers.x_exponent, "x_exponent")
                self.qua_insert_play_phased_XZ(helpers.a_exponent, helpers.x_exponent, helpers.z_exponent, 'qubit0')
                if self.num_qubits == 2:
                    assign(helpers.encoded_single_moment,
                           encoded_values_list[((pulse_seq_q1 & (15 << (4*helpers.moment_ind_helper))) >> (4*helpers.moment_ind_helper))])
                    self.qua_insert_decode_exponents(helpers.encoded_single_moment, helpers)
                    self.qua_insert_play_phased_XZ(helpers.a_exponent, helpers.x_exponent, helpers.z_exponent, 'qubit1')
            with else_():
                # Entalgaelemtn
                save(helpers.a_exponent, "cz_else")
                # self.qua_insert_play_cz()
            assign(helpers.moment_ind_helper, helpers.moment_ind_helper + 1)
        save(helpers.moment_ind_helper, "moment_ind")

    def qua_insert_play_phased_XZ(self, a,x,z, elem):
        frame_rotation_2pi(a / 2, elem)
        play("pi" * amp(x), elem)
        frame_rotation_2pi(-a / 2, elem)
        frame_rotation_2pi(z, elem)

    def qua_insert_play_cz(self):
        # play('flux_pulse', 'coupler_element') # TODO
        play('pi', 'qe2') # TODO

    def qua_insert_clifford_2_g_index(self, c, clifford_vals_list, ind_res):
        Utils.qua_add_binary_search_block(c,clifford_vals_list , self._NUM_CLIFFORDS, ind_res)

    def qua_insert_clifford_2_alpha_index(self, c, paulis_vals_list, ind_res):
        Utils.qua_add_binary_search_block(c,paulis_vals_list , self._NUM_PAULIS, ind_res)

    def qua_insert_decode_exponents(self, encoded_single_moment, qua_helper: QuaHelpers):
        ''' encoded_single_moment - qua int with exponent params encoded inside.
         '''
        assign(qua_helper.a_exponent, (1 - Cast.to_fixed((2*Utils.bitN(encoded_single_moment, 3)))) * Cast.unsafe_cast_fixed(
                                            ((Utils.bitN(encoded_single_moment, 1)) << 28) |
                                            (Utils.bitN(encoded_single_moment, 0) << 27)))
        assign(qua_helper.x_exponent, (1 - Cast.to_fixed((2*Utils.bitN(encoded_single_moment, 7)))) * Cast.unsafe_cast_fixed(
                                            ((Utils.bitN(encoded_single_moment, 5)) << 28) |
                                            (Utils.bitN(encoded_single_moment, 4) << 27)))
        assign(qua_helper.z_exponent, (1 - Cast.to_fixed((2*Utils.bitN(encoded_single_moment, 11)))) * Cast.unsafe_cast_fixed(
                                            ((Utils.bitN(encoded_single_moment, 9)) << 28) |
                                            (Utils.bitN(encoded_single_moment, 8) << 27)))

    def py_encode_phXZ_gate(self, op: Tuple[float,float,float]):
        g = 0
        exp_val_to_encoded = {0: 0, 1: int('0010', 2), -1: int('1010', 2), 0.5: int('0001', 2), -0.5: int('1001', 2)}
        for i, exp_val in enumerate(op):
            assert (exp_val in [-1,1,0.5,-0.5,0])
            g = g | (exp_val_to_encoded[exp_val]) << (4*i)
        return g

    def py_setup_gates_unique_ids(self):
        # Get all possilbe phXZ gates + ISWAP
        # for each configuration prepare a map between number to gate
        # and for each gate prepare the decoding of single gate.
        circuits = self.symplectic_compilation_data["circuits"]
        phXZ_options = set()
        non_phXZ_options = set()
        max_momet_len = 0
        for c in circuits + self.paulis_compilation_data["circuits"]:
            max_momet_len = max(len(c.moments), max_momet_len)
            for m in c.moments:
                for o in m.operations:
                    if isinstance(o.gate, cirq.PhasedXZGate):
                        phXZ_options.add(o.gate)
                    else:
                        non_phXZ_options.add(o.gate)

        assert (max_momet_len <= 8)
        assert ((len(phXZ_options) + len(non_phXZ_options)) <= 15)

        self.__num_phXZ_gates = len(phXZ_options)
        ind_to_gate = {}
        for i, op in enumerate(phXZ_options):
            ind_to_gate[i] = Utils.phXZ_to_tuple(op)
        for i, op in enumerate(non_phXZ_options):
            if isinstance(op, cirq.ISwapPowGate):
                ind_to_gate[self.__num_phXZ_gates + i] = op

        self.__num_phXZ_gates = len(phXZ_options)

        encoded_gate_val_list = [0] * 16
        for ind, gate in ind_to_gate.items():
            if ind < self.__num_phXZ_gates:
                encoded_gate_val_list[ind] = self.py_encode_phXZ_gate(gate)
            else:
                encoded_gate_val_list[ind] = 1

        self.__gate_to_ind_map = {v:k for k,v in ind_to_gate.items()}
        self.__encoded_gates_vals_list = encoded_gate_val_list
        return encoded_gate_val_list

    def py_setup_encode_circuit(self, circuit) -> int:
        ''' Takes a ciruit and encode it to integer/s. (All the moments and pulses).
            return: int '''
        _int_max = int('1'*32, 2)
        encoded_moments = [_int_max] * self.num_qubits
        for i, moment in enumerate(circuit.moments):
            # resolve the number associated with the current operation
            for op in moment.operations:
                if len(op.qubits) == 1:
                    assert (len(op.qubits) == 1)
                    qubit_num, qubit_op = op.qubits[0].x, op.gate
                    if isinstance(qubit_op, cirq.PhasedXZGate):
                        gate_to_play_number = self.__gate_to_ind_map[Utils.phXZ_to_tuple(qubit_op)]
                    else:
                        raise NotImplementedError
                    encoded_moments[qubit_num] = encoded_moments[qubit_num] & \
                                                 ((~(((~gate_to_play_number) & 15) << (4 * i))) & _int_max)
                elif len(op.qubits) == self.num_qubits:
                    if isinstance(op.gate, cirq.ISwapPowGate):
                        gate_to_play_number = self.__gate_to_ind_map[op.gate]
                    else:
                        raise NotImplementedError
                    for i in range(self.num_qubits):
                        encoded_moments[i] = encoded_moments[i] & (gate_to_play_number << (4 * i))
                else:
                    assert 0, "unsupported"

        encoded_moment_reg = []
        for m in encoded_moments:
            if 0 <= m <= 2**31 -1:
                encoded_moment_reg.append(m)
            else:
                encoded_moment_reg.append((m & 2**31-1) - 2**31)
        return encoded_moment_reg

    def py_setup_matrices_and_pulses(self, source_type) -> Tuple[List, List]:
        source_dict_mat_to_pulse = {}
        if source_type == 'pauli':
            for ind, vec in enumerate(self.paulis_compilation_data["alphas"]):
                vec_int = Utils.py_bin_vector_to_int32(vec)
                encoded_circuit = self.py_setup_encode_circuit(self.paulis_compilation_data["circuits"][ind])
                source_dict_mat_to_pulse[vec_int] = encoded_circuit
        else:
            for ind, mat in enumerate(self.symplectic_compilation_data["symplectics"]):
                mat_int = Utils.py_bin_matrix_to_int32(mat.astype(int))
                encoded_circuit = self.py_setup_encode_circuit(self.symplectic_compilation_data["circuits"][ind])
                source_dict_mat_to_pulse[mat_int] = encoded_circuit

        pulses_list = [[]] * self.num_qubits
        values_list = sorted(list(source_dict_mat_to_pulse.keys()))

        for mat in values_list:
            pulses_list[0].append(source_dict_mat_to_pulse[mat][0])
            if self.num_qubits == 2:
                pulses_list[1].append(source_dict_mat_to_pulse[mat][1])

        return values_list, pulses_list



from configuration import *
class Test:

    def __init__(self):
        self.qmm = QuantumMachinesManager(host="localhost", port=9510)
        return

    def run_simulation(self, _program, vars_to_save):
        job_sim = self.qmm.simulate(config, _program, SimulationConfig(10000))
        res = job_sim.result_handles
        while not res.wait_for_all_values():
            pass
        job_sim.get_simulated_samples().con1.plot()
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

    def test_general(self):
        with program() as p:
            a = declare(int, value=15)
            c = declare(fixed)
            # assign(c, (a+b) >> 1)
            assign(c, Cast.unsafe_cast_fixed(a << 28))
            save(c, "c")
        print(self.run_simulation(p, ["c"])['c'][0])
        return


def run_experiment(num_qubits, num_experiments=1, min_sequence_length=1, max_sequence_length=None):
    rb = RandomizedBenchmarkProgramBuilder(num_experiments=num_experiments, num_qubits=num_qubits,
                                           min_sequence_length=min_sequence_length,
                                           max_sequence_length=max_sequence_length)
    p = rb.build_qua_program()
    tester = Test()
    d =tester.run_simulation(p, ['exp_length', 'pulse_seq_q0', "moment_ind", "c"])
    for k,v in d.items():
        print(f"{k:<20}, {[i for i in v]}")


if __name__ == '__main__':
    run_experiment(1,1, 6, 7)

    # rb.py_setup_gates_unique_ids()
    # rb.py_setup_translation_to_gate()
    # tester = Test()
    # tester.test_general()
    # tester.test_binary_search()
    # import pickle
    # with open("symplectic_compilation_XZ.pkl", 'rb') as f:
    #     s = pickle.load(f)
    # print()
    # check_axz(s["circuits"])
    # print()