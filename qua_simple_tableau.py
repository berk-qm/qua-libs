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


def mat_mul_qua(m1, m2, m3):
    product_lut = declare(int, value=[0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0])  # TODO: check if true
    col_mask = declare(int, value=4369)
    row_mask = declare(int, value=15)
    place = declare(int, value=0)
    up = declare(int, value=1)
    i1 = declare(int)
    i2 = declare(int)
    prodt = declare(int, value=0)
    prodnum = declare(int, value=0)
    col = declare(int, value=0)
    row = declare(int, value=0)
    with for_(i1, 0, i1 < 4, i1+1):
        _get_row(m1, i1, row, row_mask)
        with for_(i2, 0, i2 < 4, i2+1):
            _col_to_row(m2, i2, col, col_mask)
            _rowXcol(row, col, prodnum)
            assign(prodt, product_lut[prodnum])
            assign(m3, m3 ^ (prodt << place))
            assign(place, place+up)
            save(prodnum, "prodnum")


def qua_compose_alpha(c1, a1, c2, a2, a12):
    pass


def bin_transpose(mat, transposed, col_mask):
    assign(mat, mat & 65535)
    assign(col_mask, 4369)
    save(transposed, 'transposed')
    assign(transposed, ((mat & col_mask) & 1) ^ (((mat & col_mask) & 16) >> 3) ^ (((mat & col_mask) & 256) >> 6) ^
           (((mat & col_mask) & 4096) >> 9))
    save(transposed, 'transposed')
    assign(transposed, transposed ^ (((((mat & col_mask) & 2) >> 1) ^ (((mat & col_mask) & 32) >> 4) ^ (
                        ((mat & col_mask) & 512) >> 7) ^
                   (((mat & col_mask) & 8192) >> 10)) << 4))
    save(transposed, 'transposed')
    assign(transposed, transposed ^ (((((mat & col_mask) & 4) >> 2) ^ (((mat & col_mask) & 64) >> 5) ^ (
                        ((mat & col_mask) & 1024) >> 8) ^
                   (((mat & col_mask) & 16384) >> 11)) << 8))
    save(transposed, 'transposed')
    assign(transposed, transposed ^ (((((mat & col_mask) & 8) >> 3) ^ (((mat & col_mask) & 128) >> 6) ^ (
                        ((mat & col_mask) & 2048) >> 9) ^
                   (((mat & col_mask) & 32768) >> 12)) << 12))
    save(transposed, 'transposed')


def qua_calc_inverse_alpha(g, a, inv_a):
    pass


def inverse(g, alpha, inversed_g, inversed_alpha):
    lamb = declare(int, value=18450)
    left_prod = declare(int)
    col_mask = declare(int, value=4369)
    row_mask = declare(int, value=15)
    assign(left_prod, 0)
    ct = declare(int)
    assign(ct, 0)

    bin_transpose(g, ct, col_mask)

    mat_mul_qua(lamb, ct, left_prod)
    mat_mul_qua(left_prod, lamb, inversed_g)

    qua_calc_inverse_alpha(g, alpha, inversed_alpha, lamb, inversed_g)


def then(clifford1, clifford2):
    g12 = declare(int, 0)
    alpha12 = declare(int, 0)
    mat_mul_qua(clifford1 & 65535, clifford2 & 65535, g12)
    qua_compose_alpha(clifford1 & 65535,clifford1 & 983040, clifford2 & 65535,clifford2 & 983040, alpha12)
