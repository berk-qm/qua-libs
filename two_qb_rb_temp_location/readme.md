## 2 qubit randomized benchmarking in native QUA

This is an implementation of the 2 qubit randomized benchmarking protocol
performed fully in QUA on the OPX.

The implementation builds on the theory and package in [CliffordTableau](https://github.com/liorella-qm/CliffordTableau).
In principle, the native gates supported are CX, CZ, CNOT and iSWAP. Currently only iSWAP is actively developed.