from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
from qm import LoopbackInterface
import numpy as np
from configuration import *
from qm.simulate.credentials import create_credentials # only when simulating


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
    for i11 in range(4):
        assign(i1, i11)
        _get_row(m1, i1, row, row_mask)
        for i22 in range(4):
            assign(i2, i22)
            _col_to_row(m2, i2, col, col_mask)
            _rowXcol(row, col, prodnum)
            assign(prodt, product_lut[prodnum])
            assign(m3, m3 ^ (prodt << place))
            assign(place, place+up)
            save(prodnum, "prodnum")