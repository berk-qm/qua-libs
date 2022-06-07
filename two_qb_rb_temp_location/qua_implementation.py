"""
observations:
time to randomize a number between 0 and 719 is about 240 nsec.
"""
import pickle
from itertools import combinations, product

import cirq
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from matplotlib import pyplot as plt

from config import config

cirq_qua_map = {
    cirq.LineQubit(0): "qe1",
    cirq.LineQubit(1): "qe2",
}


def play_phased_xz(gate: cirq.PhasedXZGate, qubit: cirq.LineQubit):
    qe = cirq_qua_map[qubit]
    frame_rotation_2pi(gate.axis_phase_exponent / 2, qe)
    play("pi" * amp(gate.x_exponent), qe)
    frame_rotation_2pi(-gate.axis_phase_exponent / 2, qe)
    frame_rotation_2pi(gate.z_exponent, qe)


with open('symplectic_compilation_XZ.pkl', 'rb') as f:
    clifford_data = pickle.load(f)

q1, q2 = cirq.LineQubit.range(2)
pauli_I = cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0)
pauli_X = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0)
pauli_Z = cirq.PhasedXZGate(x_exponent=0, z_exponent=1, axis_phase_exponent=0)
pauli_Y = cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0.5)

two_qubit_paulis = [cirq.Circuit([op1(q1), op2(q2)])
                    for op1, op2 in product([pauli_I, pauli_X, pauli_Z, pauli_Y], repeat=2)]

n_runs = 10
symplectic_circuits = clifford_data['circuits']

with program() as prog:
    rand = Random()
    rand.set_seed(0)
    p_rand = declare(int, value=0)
    c_rand = declare(int, value=0)
    n = declare(int, value=0)

    with for_(n, 0, n < n_runs, n + 1):
        # play Pauli
        assign(p_rand, rand.rand_int(4))
        with switch_(p_rand, unsafe=True):
            for i in range(4):
                with case_(i):
                    gate = two_qubit_paulis[i].operation_at(q2, 0).gate
                    play_phased_xz(gate, q2)

        # # play Symplectic Clifford
        assign(c_rand, rand.rand_int(6))
        moment = 0
        op_in_moment = 1  # this is because 0 to 6 are on qubit 1 (counting from 0)
        with switch_(c_rand, unsafe=True):
            for i in range(6):
                with case_(i):
                    gate = symplectic_circuits[i][moment].operations[op_in_moment].gate
                    qubit = symplectic_circuits[i][moment].operations[op_in_moment].qubits[0]
                    play_phased_xz(gate, qubit)

qmm = QuantumMachinesManager(host='lior-e0914222.dev.quantum-machines.co', port=443, credentials=create_credentials())

qm = qmm.open_qm(config)

job = qmm.simulate(config, prog, SimulationConfig(duration=1000))

job.get_simulated_samples().con1.plot()
plt.show()
