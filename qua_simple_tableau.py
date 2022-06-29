from qm.qua import *


def _get_row(mat_row, index_of_row):
    row_mask = int(b'1111', 2) << 4*index_of_row
    row = declare(int, value=0)
    assign(row, (mat_row& row_mask) >> (4*index_of_row))
    return row


def _get_col(mat, i, res_vector):
    assign(res_vector, 0)
    i_th_col_mask = int(b"0001_0001_0001_0001", 2) << i
    col_values = declare(int, value = 0)
    assign(col_values,  mat & i_th_col_mask)
    for row in range(4):
        single_val_mask = (2**i) << (4*row)
        assign(res_vector, res_vector | (((col_values & single_val_mask) >> (4 * row) >> i) << row))


def _rowXcol(rowt, colt, prodnumt):
    assign(prodnumt, (rowt & colt))


def calc_dot_product_over_z(v1, v2, res):
    assign(res, 0)
    for i in range(4):
        assign(res, res + (((v1 & (2**i)) >> i) & ((v2 &(2**i)) >> i))) # 2**i is bit mask


def calc_bin_matXvec(mat, vec, res_vec, product_lut):
    temp_prod = declare(int, value=0)
    assign(res_vec, 0)
    row_mask = int(b'1111', 2)
    for row_ind in range(4):
        assign(temp_prod, ((mat & row_mask) >> (4 * row_ind)) & vec)
        assign(res_vec, res_vec | (product_lut[temp_prod] << row_ind))
        row_mask = row_mask << 4


def qua_mat_mul_over_z2(m1, m2, m3, product_lut, row_mask, m2_transposed):
    assign(m3, 0)
    assign(m2_transposed, 0)
    bin_transpose(m2, m2_transposed)

    assign(m3, ((product_lut[(m1 & row_mask) & (m2_transposed & row_mask)]) ^
                (product_lut[((m1 & row_mask) & ((m2_transposed & (row_mask << 4)) >> 4))] << 1) ^
                 (product_lut[((m1 & row_mask) & ((m2_transposed & (row_mask << 8)) >> 8))] << 2) ^
                 (product_lut[((m1 & row_mask) & ((m2_transposed & (row_mask << 12)) >> 12))] << 3) ^
                 (product_lut[(((m1 & (row_mask << 4)) >> 4) & (m2_transposed & row_mask))] << 4) ^
                 (product_lut[(((m1 & (row_mask << 4)) >> 4) & ((m2_transposed & (row_mask << 4)) >> 4))] << 5) ^
                 (product_lut[(((m1 & (row_mask << 4)) >> 4) & ((m2_transposed & (row_mask << 8)) >> 8))] << 6) ^
                 (product_lut[(((m1 & (row_mask << 4)) >> 4) & ((m2_transposed & (row_mask << 12)) >> 12))] << 7) ^
                 (product_lut[(((m1 & (row_mask << 8)) >> 8) & (m2_transposed & row_mask))] << 8) ^
                 (product_lut[(((m1 & (row_mask << 8)) >> 8) & ((m2_transposed & (row_mask << 4)) >> 4))] << 9) ^
                 (product_lut[(((m1 & (row_mask << 8)) >> 8) & ((m2_transposed & (row_mask << 8)) >> 8))] << 10) ^
                 (product_lut[(((m1 & (row_mask << 8)) >> 8) & ((m2_transposed & (row_mask << 12)) >> 12))] << 11) ^
                 (product_lut[(((m1 & (row_mask << 12)) >> 12) & (m2_transposed & row_mask))] << 12) ^
                 (product_lut[(((m1 & (row_mask << 12)) >> 12) & ((m2_transposed & (row_mask << 4)) >> 4))] << 13) ^
                 (product_lut[(((m1 & (row_mask << 12)) >> 12) & ((m2_transposed & (row_mask << 8)) >> 8))] << 14) ^
                 (product_lut[(((m1 & (row_mask << 12)) >> 12) & ((m2_transposed & (row_mask << 12)) >> 12))] << 15)))


def calc_inverse_g(g, lamd, inv_g, product_lut, row_mask, m2_transposed):
    gt = declare(int, value=0)
    bin_transpose(g, gt)
    temp_prod = declare(int, value=0)
    qua_mat_mul_over_z2(gt, lamd, temp_prod, product_lut, row_mask, m2_transposed)
    qua_mat_mul_over_z2(lamd, temp_prod, inv_g, product_lut, row_mask, m2_transposed)


def bin_transpose(mat, transposed):
    col_mask = declare(int, value=int(b'0001_0001_0001_0001', 2))

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


def qua_calc_dot_bin_vec_reg_vec(bin_vec, reg_vec):
    return (bin_vec & (2**0)) * reg_vec[0] + \
                        ((bin_vec & (2**1)) >> 1) * reg_vec[1] + \
                        ((bin_vec & (2**2)) >> 2) * reg_vec[2] + \
                        ((bin_vec & (2**3)) >> 3) * reg_vec[3]


def py__mat_coords_to_int32_bitmap(row, col):
    ''' assuming 0 <= row, col < 4 '''
    return (2 ** col) << (4 * row)


def beta_qua(v, u, beta, beta_lut):
    ''' v, u are binary vectors (int32 each). '''
    mask = declare(int, value=0)
    v_ind = declare(int, value=0)
    u_ind = declare(int, value=0)
    assign(beta, 0)
    for i in range(2):
        assign(mask, 3 << (i * 2)) # binary mask to extract the 2 bits of current v and u.
        assign(v_ind, (((v & mask) >> (i*2)) & 1) + ((((v & mask) >> (i*2) & 2) >> 1) * 2))
        assign(u_ind, (((u & mask) >> (i*2)) & 1) + ((((u & mask) >> (i*2) & 2) >> 1) * 2))
        assign(beta, beta + (beta_lut[4*v_ind + u_ind]))


def qua_calc_bi(g1, g2, i, b_i, beta_lut):
    assign(b_i, 0)
    g1_ith_col = declare(int, value=0)
    temp_beta = declare(int, value=0)
    _get_col(g1, i, g1_ith_col)

    for j in range(0, 4, 2):
        assign(b_i, b_i + (((g1_ith_col & (2**j)) >> j) & ((g1_ith_col & (2** (j+1))) >> (j+1))))

    assign(b_i, b_i & 3)

    current = declare(int, value=0)
    g2_jth_col = declare(int, value=0)
    for j in range(4):
        with if_(~(( g1_ith_col & (2**j)) == 0)):
            _get_col(g2, j, g2_jth_col)
            beta_qua(current, g2_jth_col, temp_beta, beta_lut)
            assign(current, current ^ g2_jth_col)
        with else_():
            assign(temp_beta, 0)
        assign(b_i, (b_i + temp_beta)) # mod4
        assign(b_i, b_i & 3)


def qua_compose_alpha(g1, alpha1, g2, alpha2, a12_res, temp_bi, two_alpha_21, g1_ith_col, temp_g1_dot_alpha2,
                      beta_lut):
    assign(temp_bi, 0)
    assign(g1_ith_col, 0)
    assign(temp_g1_dot_alpha2, 0)
    for i in range(4):
        qua_calc_bi(g1, g2, i, temp_bi, beta_lut)
        _get_col(g1, i, g1_ith_col)
        calc_dot_product_over_z(g1_ith_col, alpha2, temp_g1_dot_alpha2)
        assign(two_alpha_21[i], 0)
        assign(two_alpha_21[i], (2 * ((alpha1 & (2**i))>> i) + 2 * temp_g1_dot_alpha2 + temp_bi) & 3)
        assign(a12_res, a12_res | (((two_alpha_21[i] >> 1) & (1)) << i)) # divide by 2.


def qua_calc_inverse_alpha(g, alpha, inv_g, res_inv_a, b, temp_res_inv_a, temp_bi,
                           inv_g1_transpose, temp_2alpha_plus_b, beta_lut):
    assign(inv_g1_transpose ,0)

    for i in range(4):
        qua_calc_bi(g, inv_g, i, temp_bi, beta_lut)
        assign(b[i], temp_bi)

    bin_transpose(inv_g, inv_g1_transpose)
    for i in range(4):
        assign(temp_2alpha_plus_b[i], (2 * ((alpha & (2**i)) >> i) + b[i]))

    for i in range(4):
        assign(temp_res_inv_a[i], (qua_calc_dot_bin_vec_reg_vec(_get_row(inv_g1_transpose, i), temp_2alpha_plus_b)))
        assign(res_inv_a, res_inv_a | (((3 - ((temp_res_inv_a[i] -1) & 3)) >> 1) << i))


def inverse(g, alpha, inversed_g, inversed_alpha, lamda_mat, inverse_helpers):
    calc_inverse_g(g, lamda_mat, inversed_g, inverse_helpers.product_lut,
                           inverse_helpers.row_mask, inverse_helpers.m2_transposed)
    qua_calc_inverse_alpha(g, alpha, inversed_g, inversed_alpha, inverse_helpers.b, inverse_helpers.temp_res_inv_a,
                           inverse_helpers.temp_bi, inverse_helpers.inv_g1_transpose,
                           inverse_helpers.temp_2alpha_plus_b, inverse_helpers.beta_lut)


def then(g1, alpha1, g2, alpha2, g12, alpha12, then_helpers):
    qua_mat_mul_over_z2(g2, g1, g12, then_helpers.product_lut, then_helpers.row_mask, then_helpers.m2_transposed)
    qua_compose_alpha(g1, alpha1, g2, alpha2, alpha12, then_helpers.temp_bi, then_helpers.two_alpha_21,
                      then_helpers.g1_ith_col, then_helpers.temp_g1_dot_alpha2, then_helpers.beta_lut)

