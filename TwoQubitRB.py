import functools
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

    def __init__(self, num_experiments, num_qubits, bake_cnot=True, bake_swap=True,
                 min_sequence_length=1, max_sequence_length=None):
        self.num_qubits = num_qubits
        assert num_qubits in [1,2], "Only supports 1 or 2 qubits. "
        if num_qubits == 1:
            self._NUM_PAULIS = 4
            self._NUM_CLIFFORDS = 6
        self.bake_cnot = bake_cnot
        self.bake_swap = bake_swap
        self.num_experiments = num_experiments
        self.min_seq_length = min_sequence_length
        self.max_sequence_length = self._NUM_CLIFFORDS if max_sequence_length is None else min(max_sequence_length, self._NUM_CLIFFORDS)
        self.symplectic_compilation_data = self.get_cliffords_data()
        self.paulis_compilation_data = self.get_paulis_data()
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


    def build_qua_program_gaps(self):
        py_pauli_values_list = self.py_setup_matrices_and_pulses("pauli")
        py_clifford_values_list = self.py_setup_matrices_and_pulses("clifford")

        with program() as rb_program:
            helpers = self.QuaHelpers()
            then_helpers = self.QuaThenHelpers()
            inv_helpers = self.QuaInverseHelpers()
            _rand = Random()
            random_tracker = declare(int, value=[0] * 736*2)
            experiment_ind = declare(int, value=0)
            experiment_length = declare(int, value=0)

            current_clifford_ind = declare(int, value=0)
            current_g = declare(int, value=0)
            current_alpha = declare(int, value=0)
            prev_g = declare(int, value=0)
            prev_alpha = declare(int, value=0)
            temp_g12 = declare(int, value=0)
            temp_alpha12 = declare(int, value=0)

            lamb = declare(int, value=18450)
            N = declare(int, value=0)
            N_inv = declare(int, value=0)

            pauli_mat_values_list = declare(int, value=py_pauli_values_list)
            clifford_mat_values_list = declare(int, value=py_clifford_values_list)

            inverse_g_ind = declare(int, value=0)
            inverse_alpha_ind = declare(int, value=0)

            class_case = declare(int, value=0)
            cliffords_12_inds = declare(int, value=0)
            c1_q1 = declare(int, value=0)
            c1_q2 = declare(int, value=0)
            c1_q12 = declare(int, value=0)
            s1_q1 = declare(int, value=0)
            s1_q2 = declare(int, value=0)
            class_s1_q12 = declare(int, value=0)
            pauli_q1 = declare(int, value=0)
            pauli_q2 = declare(int, value=0)

            with for_(experiment_ind, cond=(experiment_ind < self.num_experiments), update=(experiment_ind+1)):
                # run the experiment several times
                assign(experiment_length,
                       _rand.rand_int(self.max_sequence_length - self.min_seq_length) + self.min_seq_length)

                _rand.set_seed(0)

                with for_(current_clifford_ind, init=0, cond=(current_clifford_ind < experiment_length),
                          update=(current_clifford_ind + 1)):
                    assign(N, _rand.rand_int(11520))
                    assign(random_tracker[current_clifford_ind], N / 16)
                    assign(random_tracker[current_clifford_ind + 1], (N & 15) + 720)

                    assign(current_g, clifford_mat_values_list[N / 16])
                    assign(current_alpha, pauli_mat_values_list[N & 15])
                    qua_simple_tableau.then(current_g, current_alpha, prev_g, prev_alpha,
                                            temp_g12, temp_alpha12, then_helpers)
                    assign(prev_g, current_g)
                    assign(prev_alpha, current_alpha)
                    assign(current_g, temp_g12)
                    assign(current_alpha, temp_alpha12)
                    # Compute the g and alpha
                #Compute the inverse
                qua_simple_tableau.inverse(current_g, current_alpha, prev_g, prev_alpha, lamb, inv_helpers)
                self.qua_insert_clifford_2_g_index(prev_g, clifford_mat_values_list, inverse_g_ind)
                self.qua_insert_clifford_2_alpha_index(prev_alpha, pauli_mat_values_list, inverse_alpha_ind)
                assign(N_inv, inverse_g_ind * inverse_alpha_ind)

                with for_(current_clifford_ind, init=0, cond=(current_clifford_ind < experiment_length + 1),
                          update=(current_clifford_ind+1)):
                    # Loop to play clifford pulses
                    # Get random indices
                    assign(N, _rand.rand_int(11520))
                    assign(N, Util.cond(current_clifford_ind == experiment_length, N_inv, N))
                    assign(cliffords_12_inds, Util.cond((N >= 576) & (N <10944) ,N / 9, N ))
                    assign(class_case, Util.cond((N >= 576) & (N <5670), 1, 0))
                    assign(class_case, Util.cond((N >= 5670) & (N <10944), 2, class_case))
                    assign(class_case, Util.cond((N >= 10944), 3, class_case))
                    assign(N, Util.cond(class_case == 1, N - 576, N))
                    assign(N, Util.cond(class_case == 2, N - 5670, N))
                    assign(N, Util.cond(class_case == 3, N - 10944, N))

                    assign(c1_q1, cliffords_12_inds / 24 / 6)
                    assign(c1_q2, (cliffords_12_inds -  (cliffords_12_inds / 24) * 24) / 6)

                    # -----------------------------------------------------------
                    assign(c1_q12, 6*c1_q1 + c1_q2)
                    with switch_(c1_q12, unsafe=True):
                        for i in range(36):
                            with case_(i):
                                play(f"c1_{i % 6}", "qe0")
                                # frame_rotation_2pi(conf.c1_phxz[f"c1_{i}"][2], "qe0")
                                play(f"c1_{i //6}", "qe1")
                                # frame_rotation_2pi(conf.c1_phxz[f"c1_{i}"][2], "qe0")
                    # -----------------------------------------------------------
                    # with switch_(c1_q1):
                    #     for i in range(6):
                    #         with case_(i):
                    #             play(f"c1_{i}", "qe0")
                    #             # frame_rotation_2pi(conf.c1_phxz[f"c1_{i}"][2], "qe0")
                    # with switch_(c1_q2):
                    #     for i in range(6):
                    #         with case_(i):
                    #             play(f"c1_{i}", "qe1")
                    #             # frame_rotation_2pi(conf.c1_phxz[f"c1_{i}"][2], "qe1")
                    # -----------------------------------------------------------

                    assign(pauli_q1, (cliffords_12_inds/24) & 3)
                    assign(pauli_q2, (cliffords_12_inds -  (cliffords_12_inds / 24) * 24) & 3)
                    assign(s1_q1, (N / 3) - (N/9) * 3)
                    assign(s1_q2, N - (N / 3) * 3)

                    with switch_(class_case):
                        with case_(1):
                            self._add_qua_cnot_chain()
                            with switch_(s1_q1):
                                for i in range(3):
                                    with case_(i):
                                        play(f"s1_{i}", "qe0")
                                        # frame_rotation_2pi(conf.s1_phxz[f"s1_{i}"][2], "qe0")
                            with switch_(s1_q1):
                                for i in range(3):
                                    with case_(i):
                                        play(f"s1_{i}", "qe1")
                                        # frame_rotation_2pi(conf.s1_phxz[f"s1_{i}"][2], "qe1")
                        with case_(2):
                            self._add_qua_swap_chain()
                            with switch_(s1_q1):
                                for i in range(3):
                                    with case_(i):
                                        play(f"s1_{i}", "qe0")
                                        # frame_rotation_2pi(conf.s1_phxz[f"s1_{i}"][2], "qe0")
                            with switch_(s1_q1):
                                for i in range(3):
                                    with case_(i):
                                        play(f"s1_{i}", "qe1")
                                        # frame_rotation_2pi(conf.s1_phxz[f"s1_{i}"][2], "qe1")
                        with case_(3):
                            play("pi", "qe2")
                            play("pi", "qe2")


                    with switch_(pauli_q1, unsafe=True):
                        for i in range(4):
                            with case_(i):
                                play(f"pauli_{i}", "qe0")
                                # frame_rotation_2pi(conf.pauli_phxz[f"pauli_{i}"][2], "qe0")
                    with switch_(pauli_q2, unsafe=True):
                        for i in range(4):
                            with case_(i):
                                play(f"pauli_{i}", "qe1")
                                # frame_rotation_2pi(conf.pauli_phxz[f"pauli_{i}"][2], "qe1")

        return rb_program


    def build_qua_program(self):
        py_pauli_values_list = self.py_setup_matrices_and_pulses("pauli")
        py_clifford_values_list = self.py_setup_matrices_and_pulses("clifford")

        with program() as rb_program:
            then_helpers = self.QuaThenHelpers()
            inv_helpers = self.QuaInverseHelpers()
            _rand = Random()
            random_tracker = declare(int, value=[0] * 737*2)
            experiment_ind = declare(int, value=0)
            experiment_length = declare(int, value=0)

            current_clifford_ind = declare(int, value=0)
            current_g = declare(int, value=0)
            current_alpha = declare(int, value=0)
            prev_g = declare(int, value=0)
            prev_alpha = declare(int, value=0)
            temp_g12 = declare(int, value=0)
            temp_alpha12 = declare(int, value=0)

            lamb = declare(int, value=18450)
            N = declare(int, value=0)

            pauli_mat_values_list = declare(int, value=py_pauli_values_list)
            clifford_mat_values_list = declare(int, value=py_clifford_values_list)

            inverse_g_ind = declare(int, value=0)
            inverse_alpha_ind = declare(int, value=0)


            with for_(experiment_ind, cond=(experiment_ind < self.num_experiments), update=(experiment_ind+1)):
                # run the experiment several times
                assign(experiment_length,
                       _rand.rand_int(self.max_sequence_length - self.min_seq_length) + self.min_seq_length)

                _rand.set_seed(0)

                with for_(current_clifford_ind, init=0, cond=(current_clifford_ind < 2*experiment_length),
                          update=(current_clifford_ind + 2)):
                    assign(N, _rand.rand_int(11520))
                    assign(random_tracker[current_clifford_ind], N / 16)
                    assign(random_tracker[current_clifford_ind + 1], (N & 15) + 720)

                    assign(current_g, clifford_mat_values_list[N / 16])
                    assign(current_alpha, pauli_mat_values_list[N & 15])
                    # Compute the g and alpha
                    qua_simple_tableau.then(current_g, current_alpha, prev_g, prev_alpha,
                                            temp_g12, temp_alpha12, then_helpers)
                    assign(prev_g, current_g)
                    assign(prev_alpha, current_alpha)
                    assign(current_g, temp_g12)
                    assign(current_alpha, temp_alpha12)
                #Compute the inverse
                qua_simple_tableau.inverse(current_g, current_alpha, prev_g, prev_alpha, lamb, inv_helpers)
                self.qua_insert_clifford_2_g_index(prev_g, clifford_mat_values_list, inverse_g_ind)
                self.qua_insert_clifford_2_alpha_index(prev_alpha, pauli_mat_values_list, inverse_alpha_ind)
                save(current_clifford_ind, "clif_ind_end")
                assign(random_tracker[current_clifford_ind], (inverse_g_ind*inverse_alpha_ind) / 16)
                assign(random_tracker[current_clifford_ind + 1], (inverse_alpha_ind & 15) + 720)

                with for_(current_clifford_ind, init=0, cond=(current_clifford_ind < 2 * (experiment_length + 1)),
                          update=(current_clifford_ind+1)):
                    # Loop to play clifford pulses
                    assign(N, random_tracker[current_clifford_ind])

                    with switch_(N, unsafe=True):
                        for i in range(736):
                            with case_(i):
                                self._resolve_case(i)

        return rb_program

    def _resolve_case(self, i):
        if i < 36:
            self._insert_case_single_qubit_gates(*divmod(i, 6))
        elif 36 <= i < 360:
            i -= 36
            self._insert_case_CNOT(*divmod(i // 9, 6), i//3 %3, i % 3)
        elif 360 <= i < 684:
            i -= 360
            self._insert_case_SWAP(*divmod(i // 9, 6), i//3 %3, i % 3)
        elif 684 <= i < 720:
            i -= 684
            self._insert_case_ISWAP(*divmod(i // 9, 6))
        elif i >= 720:
            i -= 720
            self._insert_case_pauli(*divmod(i, 4))
        return

    def _insert_case_single_qubit_gates(self, c1_0, c1_1):
        play(f"c1_{c1_0}", self.resolve_elem(0, "xy", 0))
        play(f"c1_{c1_1}", self.resolve_elem(1, "xy", 0))
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_0}"][2], self.resolve_elem(0, "xy", 0))
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_1}"][2], self.resolve_elem(1, "xy", 0))

    def _insert_case_CNOT(self, c1_0, c1_1, s1_0, s1_1):
        q0_xy = "qubit0_xy_p0"
        q1_xy = "qubit1_xy_p0"

        play(f"c1_{c1_0}", q0_xy)
        play(f"c1_{c1_1}", q1_xy)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_0}"][2], q0_xy)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_1}"][2], q1_xy)
        if self.bake_cnot:
            play("", q0_xy)
            play("", q1_xy)
            play("", "qubit0_z")
            play("", "qubit1_z")
            play("", "coupler")
        else:
            play("cnot_0_0", q0_xy)
            play("cnot_0_1", q1_xy)

            play("coupler_pulse", "qubit0_z")
            play("coupler_pulse", "qubit1_z")
            play("coupler_pulse", "coupler")

            play("cnot_2_0",q0_xy)
            play("cnot_2_1", q1_xy)

            play("coupler_pulse", "qubit0_z")
            play("coupler_pulse", "qubit1_z")
            play("coupler_pulse", "coupler")

            play("cnot_4_0", q0_xy)
            play("cnot_4_1", q1_xy)
        # TODO: Need another frame rotation
        play(f"s1_{s1_0}", q0_xy)
        play(f"s1_{s1_1}", q1_xy)
        frame_rotation_2pi(conf.s1_phxz[f"s1_{s1_0}"][2], q0_xy)
        frame_rotation_2pi(conf.s1_phxz[f"s1_{s1_1}"][2], q1_xy)

    def _insert_case_SWAP(self, c1_0, c1_1, s1_0, s1_1):
        q0_xy = "qubit0_xy_p1"
        q1_xy = "qubit1_xy_p1"
        play(f"c1_{c1_0}", q0_xy)
        play(f"c1_{c1_1}", q1_xy)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_0}"][2], q0_xy)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_1}"][2], q1_xy)
        if self.bake_swap:
            play("", q0_xy)
            play("",q1_xy)
            play("", "qubit0_z")
            play("", "qubit1_z")
            play("", "coupler")
        else:
            play("swap_0_0", q0_xy)
            play("swap_0_1",q1_xy)

            play("coupler_pulse", "qubit0_z")
            play("coupler_pulse", "qubit1_z")
            play("coupler_pulse", "coupler")

            play("swap_2_0", q0_xy)
            play("swap_2_1",q1_xy)

            play("coupler_pulse", "qubit0_z")
            play("coupler_pulse", "qubit1_z")
            play("coupler_pulse", "coupler")

            play("swap_4_0", q0_xy)
            play("swap_4_1",q1_xy)

            play("coupler_pulse", "qubit0_z")
            play("coupler_pulse", "qubit1_z")
            play("coupler_pulse", "coupler")

            play("swap_6_0", q0_xy)
            play("swap_6_1",q1_xy)

        # TODO: Need another frame rotation
        play(f"s1_{s1_0}", q0_xy)
        play(f"s1_{s1_1}", q1_xy)
        frame_rotation_2pi(conf.s1_phxz[f"s1_{s1_0}"][2], q0_xy)
        frame_rotation_2pi(conf.s1_phxz[f"s1_{s1_1}"][2], q1_xy)

    def _insert_case_ISWAP(self, c1_0, c1_1):
        q0_xy = "qubit0_xy_p1"
        q1_xy = "qubit1_xy_p1"
        play(f"c1_{c1_0}", q0_xy)
        play(f"c1_{c1_1}", q1_xy)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_0}"][2], q0_xy)
        frame_rotation_2pi(conf.c1_phxz[f"c1_{c1_1}"][2], q1_xy)
        play("coupler_pulse", "qubit0_z")
        play("coupler_pulse", "qubit1_z")
        play("coupler_pulse", "coupler")

        play("coupler_pulse", "qubit0_z")
        play("coupler_pulse", "qubit1_z")
        play("coupler_pulse", "coupler")

    def _insert_case_pauli(self, p0, p1):
        q0_xy = "qubit0_xy_p1"
        q1_xy = "qubit1_xy_p1"

        play(f"pauli_{p0}", q0_xy)
        play(f"pauli_{p1}", q1_xy)
        frame_rotation_2pi(conf.pauli_phxz[f"pauli_{p0}"][2], q0_xy)
        frame_rotation_2pi(conf.pauli_phxz[f"pauli_{p1}"][2], q1_xy)


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



from conf import *
from conf import _config
import time
class Test:

    def __init__(self):
        self.qmm = QuantumMachinesManager(host="localhost", port=9510)
        return

    def run_simulation(self, _program, vars_to_save):
        job_sim = self.qmm.simulate(_config, _program, SimulationConfig(100000//4))
        res = job_sim.result_handles
        while not res.wait_for_all_values():
            time.sleep(0.5)
            print(".", end="")

        # pplot.subplots(3, 1)
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
            a = declare(int, value=15)
            b = declare(int, value=15)
            save(a, "a")
            play("CW", "qe")
            frame_rotation_2pi(0.35, "qe")
            frame_rotation_2pi(0.35, "qe")
            frame_rotation_2pi(0.35, "qe")
            frame_rotation_2pi(0.35, "qe")
            play("CW", "qe")
            save(b, "b")

        print(self.run_simulation(p, ["a", "b"]))
        return


def run_experiment(num_qubits, num_experiments=1, min_sequence_length=1, max_sequence_length=None):
    rb = RandomizedBenchmarkProgramBuilder(num_experiments=num_experiments, num_qubits=num_qubits,
                                           min_sequence_length=min_sequence_length,
                                           max_sequence_length=max_sequence_length)
    p = rb.build_qua_program()
    tester = Test()
    d =tester.run_simulation(p, ["clif_ind_end"])
    for k,v in d.items():
        print(f"{k:<20}, {[i for i in v]}")


if __name__ == '__main__':
    # Test()
    # run_experiment(2,1, 10,11)

    # rb.py_setup_gates_unique_ids()
    # rb.py_setup_translation_to_gate()
    tester = Test()
    tester.test_general()
    # tester.test_all_simple_tablue()
    # tester.test_general()
    # tester.test_binary_search()
    # import pickle
    # with open("symplectic_compilation_XZ.pkl", 'rb') as f:
    #     s = pickle.load(f)
    # print()
    # check_axz(s["circuits"])
    # print()