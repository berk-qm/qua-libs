from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
from qm import LoopbackInterface
import numpy as np
from configuration import *
from qm.simulate.credentials import create_credentials # only when simulating


def _col_to_row(mat_col, index_of_col, wcol, col_mask):
    assign(col_mask, 4369)
    with switch_(index_of_col):
        with case_(0):
            assign(wcol, ((mat_col & col_mask) & 1) ^ (((mat_col & col_mask) & 16) >> 3) ^ (((mat_col & col_mask) & 256) >> 6) ^
                   (((mat_col & col_mask) & 4096) >> 9))
        with case_(1):
            assign(col_mask, col_mask << 1)
            assign(wcol, (((mat_col & col_mask) & 2) >> 1) ^ (((mat_col & col_mask) & 32) >> 4) ^ (
                        ((mat_col & col_mask) & 512) >> 7) ^
                   (((mat_col & col_mask) & 8192) >> 10))
        with case_(2):
            assign(col_mask, col_mask << 2)
            assign(wcol, (((mat_col & col_mask) & 4) >> 2) ^ (((mat_col & col_mask) & 64) >> 5) ^ (
                        ((mat_col & col_mask) & 1024) >> 8) ^
                   (((mat_col & col_mask) & 16384) >> 11))
        with case_(3):
            assign(col_mask, col_mask << 3)
            assign(wcol, (((mat_col & col_mask) & 8) >> 3) ^ (((mat_col & col_mask) & 128) >> 6) ^ (
                        ((mat_col & col_mask) & 2048) >> 9) ^
                   (((mat_col & col_mask) & 32768) >> 12))
    save(col_mask, 'col_mask')


def _get_row(mat_row, index_of_row, wrow, row_mask):
    assign(row_mask, 15)
    with switch_(index_of_row):
        with case_(0):
            assign(wrow, mat_row & row_mask)
        with case_(1):
            assign(wrow, ((mat_row & (row_mask << 4)) >> 4))
        with case_(2):
            assign(wrow, ((mat_row & (row_mask << 8)) >> 8))
        with case_(3):
            assign(wrow, ((mat_row & (row_mask << 12)) >> 12))


def _rowXcol(rowt, colt, prodnumt):
    assign(prodnumt, (rowt & colt))

def calc_bin_matXvec(mat, vec, res_vec, product_lut):
    temp_prod = declare(int, value=0)
    assign(res_vec, 0)
    row_mask = int(b'1111', 2)
    for row_ind in range(4):
        assign(temp_prod, ((mat & row_mask) >> (4 * row_ind)) & vec)
        assign(res_vec, res_vec | (product_lut[temp_prod] << row_ind))
        row_mask = row_mask << 4

def mat_mul_qua(m1, m2, m3):
    product_lut = declare(int, value=[0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])  # TODO: check if true
    col_mask = declare(int, value=4369)
    row_mask = declare(int, value=15)
    transposed = declare(int, value=0)
    bin_transpose(m2, transposed, col_mask)
    assign(m3, m3 ^ ((product_lut[(m1 & row_mask) & (transposed & row_mask)]) ^
                     (product_lut[((m1 & row_mask) & ((transposed & (row_mask << 4)) >> 4))] << 1) ^
                     (product_lut[((m1 & row_mask) & ((transposed & (row_mask << 8)) >> 8))] << 2) ^
                     (product_lut[((m1 & row_mask) & ((transposed & (row_mask << 12)) >> 12))] << 3) ^
                     (product_lut[(((m1 & (row_mask << 4)) >> 4) & (transposed & row_mask))] << 4) ^
                     (product_lut[(((m1 & (row_mask << 4)) >> 4) & ((transposed & (row_mask << 4)) >> 4))] << 5) ^
                     (product_lut[(((m1 & (row_mask << 4)) >> 4) & ((transposed & (row_mask << 8)) >> 8))] << 6) ^
                     (product_lut[(((m1 & (row_mask << 4)) >> 4) & ((transposed & (row_mask << 12)) >> 12))] << 7) ^
                     (product_lut[(((m1 & (row_mask << 8)) >> 8) & (transposed & row_mask))] << 8) ^
                     (product_lut[(((m1 & (row_mask << 8)) >> 8) & ((transposed & (row_mask << 4)) >> 4))] << 9) ^
                     (product_lut[(((m1 & (row_mask << 8)) >> 8) & ((transposed & (row_mask << 8)) >> 8))] << 10) ^
                     (product_lut[(((m1 & (row_mask << 8)) >> 8) & ((transposed & (row_mask << 12)) >> 12))] << 11) ^
                     (product_lut[(((m1 & (row_mask << 12)) >> 12) & (transposed & row_mask))] << 12) ^
                     (product_lut[(((m1 & (row_mask << 12)) >> 12) & ((transposed & (row_mask << 4)) >> 4))] << 13) ^
                     (product_lut[(((m1 & (row_mask << 12)) >> 12) & ((transposed & (row_mask << 8)) >> 8))] << 14) ^
                     (product_lut[(((m1 & (row_mask << 12)) >> 12) & ((transposed & (row_mask << 12)) >> 12))] << 15)))




def qua_compose_alpha(c1, a1, c2, a2, a12):
    pass


def bin_transpose(mat, transposed, col_mask):
    assign(mat, mat & 65535)
    assign(col_mask, 4369)

    assign(transposed, ((mat & col_mask) & 1) ^ (((mat & col_mask) & 16) >> 3) ^ (((mat & col_mask) & 256) >> 6) ^
           (((mat & col_mask) & 4096) >> 9))

    assign(transposed, transposed ^ (((((mat & (col_mask << 1)) & 2) >> 1) ^ (((mat & (col_mask << 1)) & 32) >> 4) ^ (
                        ((mat & (col_mask << 1)) & 512) >> 7) ^
                   (((mat & (col_mask << 1)) & 8192) >> 10)) << 4))

    assign(transposed, transposed ^ (((((mat & (col_mask << 2)) & 4) >> 2) ^ (((mat & (col_mask << 2)) & 64) >> 5) ^ (
                        ((mat & (col_mask << 2)) & 1024) >> 8) ^
                   (((mat & (col_mask << 2)) & 16384) >> 11)) << 8))

    assign(transposed, transposed ^ (((((mat & (col_mask << 3)) & 8) >> 3) ^ (((mat & (col_mask << 3)) & 128) >> 6) ^ (
                        ((mat & (col_mask << 3)) & 2048) >> 9) ^
                   (((mat & (col_mask << 3)) & 32768) >> 12)) << 12))


def py__mat_coords_to_int32_bitmap(row, col):
    ''' assuming 0 <= row, col < 4 '''
    return (2 ** col) << (4 * row)


def beta_qua(v, u, beta):
    ''' v, u are binary vectors (int32 each). '''
    lut = declare(int, value=[0, 0, 0, 0, 0, 0, 3, 1, 0, 1, 0, 3, 0, 3, 1, 0])
    mask = declare(int, value=0)
    v_ind = declare(int, value=0)
    u_ind = declare(int, value=0)
    assign(beta, 0)
    for i in range(2):
        assign(mask, 3 << (i * 2)) # binary mask to extract the 2 bits of current v and u.
        assign(v_ind, (((v & mask) >> (i*2)) & 1) + ((((v & mask) >> (i*2) & 2) >> 1) * 2))
        assign(u_ind, (((u & mask) >> (i*2)) & 1) + ((((u & mask) >> (i*2) & 2) >> 1) * 2))
        assign(beta, beta + (lut[4*v_ind + u_ind]))


def qua_calc_bi(g1, g2, ii, b_i):
    assign(b_i, 0)
    product_lut = declare(int, value=[0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])  # TODO: check if true

    e_i_list = [1, 2, 1 << 2, 2 << 2]  # All the e_i vectors (e[i] = e_i) (x_0, z_0, x_1, z_1).
    beta = declare(int, value=0)
    beta_arg1 = declare(int, value=0)
    beta_arg2 = declare(int, value=0)
    for k in range(0, 4, 2):
        calc_bin_matXvec(g2, e_i_list[k], beta_arg1, product_lut)
        calc_bin_matXvec(g2, e_i_list[k+1], beta_arg2, product_lut)
        beta_qua(beta_arg1, beta_arg2, beta)
        save(beta, f"beta{k}")
        gki_mask = py__mat_coords_to_int32_bitmap(k, ii)
        gki_p1_mask = py__mat_coords_to_int32_bitmap(k+1, ii)

        gki_mask_revert_shift_amount = 4 * k + ii
        gki_p1_mask_revert_shift_amount = 4 * (k+1) + ii
        t1 = declare(int, value=0)
        t2 = declare(int, value=0)
        assign(t1, ((g1 & gki_mask) >> gki_mask_revert_shift_amount))
        assign(t2, ((g1 & gki_p1_mask) >> gki_p1_mask_revert_shift_amount))

        save(t1, "t1")
        save(t2, "t2")
        assign(b_i, (b_i +
                     (
                (((g1 & gki_mask) >> gki_mask_revert_shift_amount) *
                ((g1 & gki_p1_mask) >> gki_p1_mask_revert_shift_amount)) * (1 + beta))
                     )
               )

        assign(b_i, b_i & 3) # Performing mod4 (b_i is NOT binary number).

        assign(beta, 0) # reset beta


        # beta_qua(current, ((((g1_transpose & (row_mask << ii*4)) >> ii*4) & (spe << j)) >> j) &
        #                             ((g2 & (row_mask << j*4)) >> j*4)))) # remember to %4
        # perform mod4 (&3).

        # ---------------------------------------------------------
        # b_i = (b_i + beta_qua(current, g1[j, ii] * g2[:, j])) % 4
        # assign(current, (current ^ g1[j, ii] & g2[:, j]))  # remeber to % 2
        # current = (current + g1[j, ii] * g2[:, j]) % 2


def qua_calc_inverse_alpha(g, a, inv_a, lamb, inv_g):
    pass


def inverse(g, alpha, inversed_g, inversed_alpha):
    lamb = declare(int, value=18450)
    left_prod = declare(int)
    col_mask = declare(int, value=4369)
    row_mask = declare(int, value=15)
    assign(left_prod, 0)
    gt = declare(int)
    assign(gt, 0)

    bin_transpose(g, gt, col_mask)

    mat_mul_qua(lamb, gt, left_prod)
    mat_mul_qua(left_prod, lamb, inversed_g)

    qua_calc_inverse_alpha(g, alpha, inversed_alpha, lamb, inversed_g)


def then(clifford1, clifford2):
    g12 = declare(int, 0)
    alpha12 = declare(int, 0)
    mat_mul_qua(clifford1 & 65535, clifford2 & 65535, g12)
    qua_compose_alpha(clifford1 & 65535, clifford1 & 983040, clifford2 & 65535, clifford2 & 983040, alpha12)








