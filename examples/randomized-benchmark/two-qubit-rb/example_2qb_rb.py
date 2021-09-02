from rb_2qb import *
from qm.QuantumMachinesManager import SimulationConfig
from configuration import config
from typing import Optional

qmm = QuantumMachinesManager()

"""
For the required two qubit gates to generate the 4 classes (see Supplementary info of this paper:
https://arxiv.org/pdf/1210.7011.pdf), we require the user to complete the following macros below according
to their own native set of qubit gates, that is perform the appropriate decomposition and convert the pulse sequence
in a sequence of baking play statements (amounts to similar structure as QUA, just add the prefix b. before every statement)
Example :
in QUA you would have for a CZ operation:
    play("CZ", "coupler")
in the macros below you write instead:
    b.play("CZ", "coupler")
"""
# Define here quantum elements required to compute the macros
q0 = "q0"
q1 = "q1"
coupler = "coupler"
# Baking Macros required for two qubit gates

# Here is my assumed native two qubit gate


def CZ(b_seq: Baking):
    b_seq.align(q0, q1, coupler)
    b_seq.play("CZ", coupler)
    b_seq.align(q0, q1, coupler)


def CNOT(b_seq: Baking, ctrl: str = "q0", tgt: str = "q1"):
    # Map your pulse sequence for performing a CNOT using baking play statements
    #

    # Option 1: simple play statement for single qubit gate

    b_seq.play("-Y/2", tgt)
    CZ(b_seq)
    b_seq.play("Y/2", tgt)
    # Option 2: macro required for single qubit gate

    # mY_2(b_seq, tgt)
    # b_seq.align(tgt, coupler)
    # b_seq.play("CZ", coupler)
    # b_seq.align(ctrl, tgt, coupler)
    # Y_2(b_seq, tgt)


def iSWAP(b_seq: Baking):

    b_seq.play("-X/2", q1)
    CNOT(b_seq, q1, q0)
    b_seq.play("-X/2", q1)
    CZ(b_seq)
    b_seq.play("Y/2", q0)
    b_seq.play("X/2", q1)


def SWAP(b_seq: Baking):
    CNOT(b_seq, q0, q1)
    CNOT(b_seq, q1, q0)
    CNOT(b_seq, q0, q1)


"""
In what follows, q_tgt should be the main target qubit for which should be played the single qubit gate.
qe_set can be a set of additional quantum elements that might be needed to actually compute the gate
(e.g fluxline, trigger, ...). It is then up to the user to use a name convention for elements allowing him to perform
the correct gate to the right target qubit and its associated elements
"""


def I(b: Baking, q_tgt, *qe_set: str):
    pass


def X(b: Baking, q_tgt, *qe_set: str):
    pass


def Y(b: Baking, q_tgt, *qe_set: str):
    pass


def X_2(b: Baking, q_tgt, *qe_set: str):
    pass


def Y_2(b: Baking, q_tgt, *qe_set: str):
    pass


def mX_2(b: Baking, q_tgt, *qe_set: str):
    pass


def mY_2(b: Baking, q_tgt, *qe_set: str):
    pass


two_qb_gate_macros = {
    "CNOT": CNOT,
    "iSWAP": iSWAP,
    "SWAP": SWAP
}

single_qb_gate_macros = {
    "I": I,
    "X": X,
    "Y": Y,
    "X/2": X_2,
    "-X/2": mX_2,
    "Y/2": Y_2,
    "-Y/2": mY_2
}


def qua_prog(b_seq: Baking, N_shots: int):
    with program() as prog:
        n = declare(int)
        th1 = declare(fixed, value=0.)
        th2 = declare(fixed, value=0.)
        stream1 = declare_stream()
        stream2 = declare_stream()
        state1 = declare(bool)
        state2 = declare(bool)
        I1 = declare(fixed)
        I2 = declare(fixed)
        d1 = declare(fixed)
        d2 = declare(fixed)
        d3 = declare(fixed)
        d4 = declare(fixed)
        with for_(n, 0, n < N_shots, n+1):
            b_seq.run()
            align()
            measure('readout', "rr1", None, demod.full('integW1', d1, 'out1'),
                    demod.full('integW2', d2, 'out2'))
            measure('readout', "rr2", None, demod.full('integW1', d3, 'out1'),
                    demod.full('integW2', d4, 'out2'))

            assign(I1, d1+d2)
            assign(I2, d3+d4)
            assign(state1, I1 > th1)
            assign(state2, I2 > th2)
            save(state1, stream1)
            save(state2, stream2)
        with stream_processing():
            stream1.boolean_to_int().average().save("state1")
            stream2.boolean_to_int().average().save("state2")

    return prog


nCliffords = range(1, 180, 2)
s = RBTwoQubits(qmm=qmm, config=config,
                N_Clifford=nCliffords, K=1,
                two_qb_gate_baking_macros=two_qb_gate_macros,
                quantum_elements=("q0", "q1"))
sequences = s.sequences
s1 = sequences[0].full_sequence
# for h in s1:
#     print(len(h), h)

baked_reference = s.baked_reference
print(baked_reference.get_Op_length("q0"))
print("starting simulation")
job = qmm.simulate(config=config, program=qua_prog(baked_reference, 100), simulate=SimulationConfig(5000))

samples = job.get_simulated_samples()
samples.con1.plot()
