import random

import qua_simple_tableau as qst
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
from qm import LoopbackInterface
import numpy as np
from configuration import *
from qm.simulate.credentials import create_credentials  # only when simulating
import simple_tableau


LUT_4BIT_ADD_TO_MOD = [bin(i).strip("0b").count("1") % 2 for i in range(16)]


class TestQUASimpleTableau:

    def __init__(self, num_qubits):
        self.qmm = QuantumMachinesManager(host="localhost", port=9510)
        self.num_qubits = num_qubits
        self.pauli_vectors_1_qubit = {"I": np.array([0,0]),
                              "X": np.array([1,0]),
                              "Z": np.array([0,1]),
                              "Y": np.array([1,1]),
                              }
        return


    def run_qmm_simulation(self, _program, var_to_save):
        job_sim = self.qmm.simulate(config, _program, SimulationConfig(100000))
        res = job_sim.result_handles
        res.__getattribute__(var_to_save).wait_for_all_values()
        return res.__getattribute__(var_to_save).fetch_all()[0][0]

    def array_to_int32(self, arr):
        return int(sum([i * 2 ** e for e, i in enumerate(arr)]))

    def matrix_to_int32(self, mat):
        r = 0
        for row_i, row in enumerate(mat):
            rep_num = sum([i * 2 ** e for e, i in enumerate(row)])
            r |= rep_num << (row_i * 4)
        return int(r)

    def test_matXvec(self, mat=None, vec=None):
        if not all([mat, vec]):
            mat = np.random.randint(0, 2, [self.num_qubits*2]*2)
            vec = np.random.randint(0,2, [self.num_qubits*2])
        vec_ref = np.matmul(mat, vec) % 2

        vec_ref_int32 = self.array_to_int32(vec_ref)
        mat_int = self.matrix_to_int32(mat)
        vec_int = self.array_to_int32(vec)

        with program() as p:
            prod_lut = declare(int, value=LUT_4BIT_ADD_TO_MOD)
            _mat = declare(int, value=mat_int)
            _vec = declare(int, value = vec_int)
            prod = declare(int, value=0)
            qst.calc_bin_matXvec(_mat, _vec, prod, prod_lut)
            save(prod, "prod_vec")

        qua_vec = self.run_qmm_simulation(p, "prod_vec")
        if qua_vec == vec_ref_int32:
            print("Test Passed!")
        else:
            print(f"Test Failed!! {mat=},  {vec=}")
        return



    def run_single_beta_calc(self, v,u):
        v_int = self.array_to_int32(v)
        u_int = self.array_to_int32(u)
        with program() as p:
            v1 = declare(int, value=v_int)
            v2 = declare(int, value=u_int)
            beta = declare(int, value=0)
            qst.beta_qua(v1, v2, beta)
            save(beta, "beta")

        beta = self.run_qmm_simulation(p, "beta")

        ref_beta = simple_tableau._beta(v,u)
        return beta, ref_beta

    def get_random_pauli(self):
        def _get_random_pauli_1_qubit():
            return self.pauli_vectors_1_qubit[random.choice(list(self.pauli_vectors_1_qubit.keys()))]

        pauli_vec = np.zeros(self.num_qubits*2, dtype=np.int32)
        for n in range(self.num_qubits):
            pauli_vec[2*n: 2*n+2] = _get_random_pauli_1_qubit()
        return pauli_vec

    def get_random_symplectic_matrix(self):
        mat = np.random.randint(low=0, high=2, size=[self.num_qubits*2]*2, dtype=int)
        while not simple_tableau._is_symplectic(mat, self.num_qubits):
            mat = np.random.randint(low=0, high=2, size=[self.num_qubits*2]*2, dtype=int)

        return mat

    def test_beta(self, v=None, u=None, num_to_run=10):
        if v is not None and u is not None:
            beta, beta_ref = self.run_single_beta_calc(v, u)
            if beta==beta_ref:
                print("Test Passed!")
            else:
                print("Test Failed!!")
        else:
            num_fail, num_pass = 0,0
            for i in range(num_to_run):
                v = self.get_random_pauli()
                u = self.get_random_pauli()
                beta, beta_ref = self.run_single_beta_calc(v, u)
                if beta == beta_ref:
                    num_pass += 1
                else:
                    num_fail += 1
                    print(f"Failed!!!: {v=},  {u=}")
            if num_pass == num_to_run:
                print("All Tests Passed!")
        return

    def run_single_bi_calc(self, g1, g2, i):
        g1_int = self.matrix_to_int32(g1)
        g2_int = self.matrix_to_int32(g2)
        with program() as p:
            _g1 = declare(int, value=g1_int)
            _g2 = declare(int, value=g2_int)
            b_i = declare(int, value=0)
            qst.qua_calc_bi(_g1, _g2, i, b_i)
            save(b_i, "b_i")

        bi = self.run_qmm_simulation(p, "b_i")

        ref_bi = simple_tableau._calc_b_i(g1, g2, i)
        return bi, ref_bi

    def test_bi(self, _g1=None, _g2=None, num_to_run=10):
        num_passed, num_failed = 0,0
        for _n in range(num_to_run):
            if _g1 is None and _g2 is None:
                g1 = self.get_random_symplectic_matrix()
                g2 = self.get_random_symplectic_matrix()
            else:
                g1, g2 = _g1, _g2
            print(f"========== \n Running matrices: \n{g1=}\n {g2=}")
            for i in range(self.num_qubits*2):
                b_i, b_i_ref = self.run_single_bi_calc(g1, g2, i)
                if b_i == b_i_ref:
                    print(f"\nTest Passed ({i=} {b_i=}, {b_i_ref=})")
                    num_passed += 1
                else:
                    num_failed += 1
                    print(f"\nTest Failed!! ({i=} {b_i=}, {b_i_ref=})")
            if _g1 is None and _g2 is None:
                return

        return

tester = TestQUASimpleTableau(2)
# for i in range(10):
#     tester.test_matXvec()

print("Testing b_i")
# tester.test_bi()



def get_g(name):
    g = np.identity(4, dtype=np.int32)
    alpha = np.zeros(4)
    simple_tableau._embed_two_qubit_gate(alpha, g, name, (0,1))
    return g, alpha




def bin_mat_to_int32(mat):
    r = 0
    for row_i, row in enumerate(mat):
        rep_num = sum([i* 2**e for e,i in enumerate(row)])
        r |= rep_num << (row_i*4)
    return r

def bin_array_to_int32(arr):
    return sum([i * 2 ** e for e, i in enumerate(arr)])

def simulate_beta(v,u):
    lut = [0, 0, 0, 0, 0, 0, 3, 1, 0, 1, 0, 3, 0, 3, 1, 0]
    beta = 0
    for i in range(2):
        mask =  3 << i * 2
        v_ind = ((((v & mask) >> (i*2)) & 2) >> 1) + ((v & mask) >> (i*2) & 1) * 2
        u_ind = ((((u & mask) >> (i*2)) & 2) >> 1) + ((u & mask) >> (i**2) & 1) * 2
        beta += (lut[4*v_ind + u_ind])
    return beta



def test_all_beta():
    possible_vectors = {"I": 0,
                        "X": 2,
                        "Z": 1,
                        "Y": 3}
    vectors_to_symbols = {v:k for k,v in possible_vectors.items()}
    pauli_vectors = list(possible_vectors.keys())
    first_pauli = random.choice(pauli_vectors)
    second_pauli = random.choice(pauli_vectors)
    non_phase = possible_vectors[first_pauli] ^ possible_vectors[second_pauli]

    phase = complex(0, 1) ** (simulate_beta(possible_vectors[first_pauli], possible_vectors[second_pauli]))

    print(f"{first_pauli} * {second_pauli} = {phase} {vectors_to_symbols[non_phase]}")


def py__mat_coords_to_int32_bitmap(row, col):
    ''' assuming 0 <= row, col < 4 '''
    return (2 ** col) << (4 * row)


def simulate_b_calc(g1,g2, ii):
    b_i = 0

    e_i_list = [1, 2, 1 << 2, 2 << 2]  # All the e_i vectors (e[i] = e_i) (x_0, z_0, x_1, z_1).
    for k in range(0, 4, 2):
        beta_arg1 = simulate_matXvec(g2, e_i_list[k])
        beta_arg2 = simulate_matXvec(g2, e_i_list[k + 1])
        beta = simulate_beta(beta_arg1, beta_arg2)

        gki_mask = py__mat_coords_to_int32_bitmap(k, ii)
        gki_p1_mask = py__mat_coords_to_int32_bitmap(k + 1, ii)

        gki_mask_revert_shift_amount = 4 * k + ii
        gki_p1_mask_revert_shift_amount = 4 * (k + 1) + ii
        b_i = b_i + \
                     (
                             (((g1 & gki_mask) >> gki_mask_revert_shift_amount) *
                              ((g1 & gki_p1_mask) >> gki_p1_mask_revert_shift_amount)) * (1 + beta)
                     )

        b_i = b_i & 3  # Performing mod4 (b_i is NOT binary number).

        beta=0  # reset beta

    return b_i

def simulate_matXvec(mat, vec):
    temp_prod = 0

    res_vec = 0
    row_mask = int(b'1111', 2)
    for row_ind in range(4):
        temp_prod = (((mat & row_mask) >> (4*row_ind)) & vec)
        res_vec = res_vec | (LUT_4BIT_ADD_TO_MOD[temp_prod] << (row_ind))
        row_mask = row_mask << 4

    # for b in format(res_vec, '08b')[-1:-5: -1]:
    #     print(b)
    return res_vec


mat =np.array([[0, 1, 1, 0],
               [0, 1, 0, 1],
               [1, 0, 1, 1],
               [1, 1, 1, 1]])

mat2 =np.array([[1, 0, 0, 0],
               [0, 1, 1, 0],
               [0, 0, 1, 0],
               [1, 0, 0, 1]])

tester.test_beta(np.matmul(mat2, [1,0,0,0]), np.matmul(mat2, [0,1,0,0]), 1)
tester.test_beta(np.matmul(mat2, [0,0,1,0]), np.matmul(mat2, [0,0,0,1]), 1)
tester.test_bi(mat, mat2, 1)
exit()
# mat = [[1,1,1,0], [1,0,0,1],[1,1,1,1],[1,1,1,1]]
# mat = [[1,1,1,0], [1,0,0,1],[1,1,1,1],[1,1,1,1]]
# mat2 =[[1,0,1,1], [0,0,0,1],[1,1,0,0],[0,1,0,1]]
mat_bin = bin_mat_to_int32(mat)
mat2_bin = bin_mat_to_int32(mat2)

vec = bin_array_to_int32([1,1,1,1])

simulate_matXvec(mat2_bin, 15)
simulate_matXvec(mat2_bin, 2)

# simulate_matXvec(mat, vec)
b_i = simulate_b_calc(mat_bin, mat2_bin, 2)

from simple_tableau import _calc_b_i

b_i_ref = _calc_b_i(np.array(mat), np.array(mat2), 2)
# test_all_beta()
# exit()











exit()



with program() as test:
    # mat1 = declare(int, value=33825)
    # mat2 = declare(int, value=33825)
    # mat3 = declare(int, value=0)
    # col_mask = declare(int, value=4369)
    # mat3Stream = declare_stream()
    # save(mat3, mat3Stream)
    #
    # qst.mat_mul_qua(mat1, mat2, mat3)
    # save(mat3, mat3Stream)
    # with stream_processing():
    #     mat3Stream.with_timestamps().save_all('mat3')

    v1 = declare(int, value=3)
    v2 = declare(int, value=2)
    beta = declare(int, value=0)
    qst.beta_qua(v1,v2,beta)
    save(beta, "beta")


# simulate_beta(3,2)

# qmm = QuantumMachinesManager(host="localhost", port=9510, credentials=create_credentials())
qmm = QuantumMachinesManager(host="localhost", port=9510)
job_sim = qmm.simulate(config, test, SimulationConfig(100000))

res = job_sim.result_handles
res.beta.wait_for_all_values()
print(res.beta.fetch_all())