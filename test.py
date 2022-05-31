import random

import qua_simple_tableau as qst
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
from qm import LoopbackInterface
import numpy as np
from configuration import *
from qm.simulate.credentials import create_credentials  # only when simulating

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

def simulate_b_calc():
    ...

def simulate_matXvec(mat, vec):
    temp_prod = 0
    product_lut = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]

    res_vec = 0
    row_mask = int(b'1111', 2)
    for row_ind in range(4):
        temp_prod = (((mat & row_mask) >> (4*row_ind)) & vec)
        res_vec = res_vec | (product_lut[temp_prod] << (row_ind))
        row_mask = row_mask << 4

    for b in format(res_vec, '08b')[-1:-5: -1]:
        print(b)

mat = int(b'1010_0011_1101_0011', 2)
vec = int(b'1010', 2)

mat = bin_mat_to_int32([[1,1,1,0], [1,0,0,1],[1,1,1,1],[1,1,1,1]])
vec = bin_array_to_int32([1,1,1,1])
simulate_matXvec(mat, vec)

# test_all_beta()
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