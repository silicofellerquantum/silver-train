# quantum_runs - results

All 8 textbook notebooks run at their maximum qubit count. The notebooks in the repository
root were not modified: each run executes a copy with Section 3 parameter overrides only.
At the full 156-qubit width every hardware result is dominated by hardware noise - the
expected outcome of a scaling run on current devices.

| Run | Qubits | Where | Status | IBM jobs | QPU s | Result |
|---|---|---|---|---|---|---|
| `qft_ibm_156q` | 156 | IBM ibm_kingston | hardware result complete; analysis limit | 1 | 6 | 4000 shots -> 4000 distinct outcomes (ideal: 8 peaks); notebook analysis stops above 63 qubits (OverflowError, 64-bit integers) - see qft_ibm_63q |
| `qft_ibm_63q` | 63 | IBM ibm_marrakesh | completed | 1 | 4 | 0% of shots on the 8 expected peaks (ideal 100%) |
| `grover_ibm_156q` | 156 | IBM ibm_kingston | completed | 1 | 21 | 1 iteration: P(marked) 0/2000, theory ~0 |
| `vqe_ibm_156q` | 156 | IBM ibm_kingston, ibm_marrakesh | completed | 5 | 23 | E = -43.65 vs exact E0 = -165.04 (relative error 74%) |
| `qaoa_ibm_156q` | 156 | IBM ibm_fez, ibm_kingston, ibm_marrakesh | completed | 6 | 24 | <C> = 71.98 on a 156-node ring; exact max cut 156 (notebook's local search: 140) -> ratio 0.46, below random (0.5) |
| `qml_ibm_156q` | 156 | IBM ibm_kingston, ibm_marrakesh | completed | 6 | 17 | test accuracy 50% (= random guessing), training 100% on 2 images |
| `shor_sim_21q` | 21 | CPU simulator | completed | 0 | 0 | 119 = 7 x 17; 94% of shots yield the factors |
| `hhl_sim_19q` | 19 | CPU simulator | completed | 0 | 0 | solution fidelity 0.9952 |
| `noise_sim_1000q` | 1000 | CPU simulator | completed | 0 | 0 | GHZ, depolarizing p = 0.01: Hellinger fidelity 0.0003 |

## qft_ibm_156q

Notebook: `results/qft_ibm_156q/QFT.ipynb`

    CUDA-Q target     : ibm  (ibm QPU (REAL HARDWARE - needs credentials, may cost money))

Errors:

    OverflowError: Python int too large to convert to C long
    NameError: name 'on_peaks' is not defined

| time (UTC) | account S.No | backend | job id | status | QPU s | chip qubits | depth | 2q gates |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09T05:23:46+00:00 | 1 | ibm_kingston | `dbnlm5c4148s73918770` | COMPLETED | 6 | 156 | 18981 | 67552 |

## qft_ibm_63q

Notebook: `results/qft_ibm_63q/QFT.ipynb`

    CUDA-Q target     : ibm  (ibm QPU (REAL HARDWARE - needs credentials, may cost money))
    Algorithm                           : Quantum Fourier Transform
    Number of qubits                    : 63
    CUDA-Q target                       : ibm
    Number of shots                     : 4000
    Total wall time of execution cell   : 930.0203 s
    Gates in the QFT circuit            : 2109 (depth 128)
    Expected result                     : 8 peaks at multiples of 1,152,921,504,606,846,976, P = 0.1250 each
    Actual result                       : 4,000 distinct outcomes; most frequent 001001111000110000000010001001000000000010100100001000000011001 (y=1,424,827,508,935,757,849) with 0.0003

| time (UTC) | account S.No | backend | job id | status | QPU s | chip qubits | depth | 2q gates |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09T05:40:14+00:00 | 1 | ibm_marrakesh | `dboli88qtr0c738d1jk0` | COMPLETED | 4 | 64 | 4295 | 11029 |

## grover_ibm_156q

Notebook: `results/grover_ibm_156q/Grover.ipynb`

    Number of qubits N          : 156
    CUDA-Q target     : ibm  (ibm QPU (REAL HARDWARE - needs credentials, may cost money))
    Expected result
    Gates in the main circuit : 930   (per iteration: 774)
    Gates in the sweep        : 0
    Algorithm                         : Grover's algorithm (quantum search)
    Number of qubits                  : 156
    CUDA-Q target                     : ibm
    Number of shots                   : 2000
    Total wall time of execution cell : 18.2986 s
    Expected result                   : P(marked) = 0.000000 -> about 0 of 2000 shots
    Error |P_measured - P_theory|     : 0.0000 (1-sigma shot noise: 0.0000)

| time (UTC) | account S.No | backend | job id | status | QPU s | chip qubits | depth | 2q gates |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09T05:23:50+00:00 | 7 | ibm_kingston | `dbnlm5k4148s73918780` | COMPLETED | 21 | 156 | 274020 | 121332 |

## vqe_ibm_156q

Notebook: `results/vqe_ibm_156q/VQE.ipynb`

    CUDA-Q target     : ibm  (ibm QPU (REAL HARDWARE - needs credentials, may cost money))
    VQE energy      : -43.6452500000
    Error           : 1.214e+02 (relative 73.555%)
    Algorithm                             : VQE (variational quantum eigensolver)
    Number of qubits                      : 156
    CUDA-Q target                         : ibm
    Number of shots                       : 4000
    Total wall time of execution cell     : 1247.1123 s
    Expected result                       : E0 = -165.0422569082 (exact (free-fermion solution))
    Actual result                         : E_VQE = -43.6452500000
    Error E_VQE - E0                      : 1.214e+02 (relative 73.5551%)

| time (UTC) | account S.No | backend | job id | status | QPU s | chip qubits | depth | 2q gates |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09T05:23:45+00:00 | 13 | ibm_kingston | `dbnlmitt7fqc73ajrg6g` | COMPLETED | 5 | 156 | 1410 | 1039 |
| 2026-11-09T05:43:29+00:00 | 13 | ibm_marrakesh | `dbolhp0qtr0c738d1iu0` | COMPLETED | 5 | 156 | 1410 | 1039 |
| 2026-11-09T05:43:54+00:00 | 13 | ibm_marrakesh | `dbolr0oou4as73csad70` | COMPLETED | 5 | 156 | 1410 | 1039 |
| 2026-11-09T05:44:17+00:00 | 13 | ibm_marrakesh | `dbolr6s4148s739t7ol0` | COMPLETED | 5 | 156 | 1410 | 1039 |
| 2026-11-09T05:44:29+00:00 | 13 | ibm_marrakesh | `dbolrcgou4as73csadu0` | COMPLETED | 3 | 156 | 1410 | 1039 |

## qaoa_ibm_156q

Notebook: `results/qaoa_ibm_156q/QAOA.ipynb`

    CUDA-Q target     : ibm  (ibm QPU (REAL HARDWARE - needs credentials, may cost money))
    Algorithm                             : QAOA (Max-Cut)
    Number of qubits                      : 156
    CUDA-Q target                         : ibm
    Number of shots                       : 4000
    Total wall time of execution cell     : 1281.0433 s
    Expected result                       : max cut = 140 (heuristic local search (best known, NOT guaranteed optimal))

| time (UTC) | account S.No | backend | job id | status | QPU s | chip qubits | depth | 2q gates |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09T05:23:40+00:00 | 19 | ibm_kingston | `dbnlmi8qtr0c73ad9l50` | COMPLETED | 4 | 156 | 4762 | 1650 |
| 2026-11-09T05:23:42+00:00 | 19 | ibm_fez | `dbnltvgou4as73b4q4fg` | COMPLETED | 4 | 156 | 4762 | 1650 |
| 2026-11-09T05:43:38+00:00 | 19 | ibm_marrakesh | `dbolhntt7fqc73881qjg` | COMPLETED | 4 | 156 | 4762 | 1650 |
| 2026-11-09T05:44:11+00:00 | 19 | ibm_marrakesh | `dbolr2oou4as73csadbg` | COMPLETED | 4 | 156 | 4762 | 1650 |
| 2026-11-09T05:44:33+00:00 | 19 | ibm_marrakesh | `dbolrb5t7fqc73882ca0` | COMPLETED | 4 | 156 | 4762 | 1650 |
| 2026-11-09T05:44:56+00:00 | 19 | ibm_marrakesh | `dbolrgk4148s739t7p30` | COMPLETED | 4 | 156 | 4762 | 1650 |

## qml_ibm_156q

Notebook: `results/qml_ibm_156q/QML.ipynb`

    CUDA-Q target: ibm
    QCNN on 156 qubits: 8 layers, 460 blocks, 32 trainable parameters, output qubit 155
    Training finished: 3 evaluations in 1114.4 s (Powell)
    Test accuracy     : 50.00%   (random guessing: 50%)
    Algorithm                      : QCNN classifier (horizontal vs vertical lines)
    CUDA-Q target                  : ibm
    Number of qubits               : 156
    QCNN layers / parameters       : 8 / 32
    Expected result                : accuracy clearly above 50 % (random guessing)
    Test accuracy                  : 50.00%

| time (UTC) | account S.No | backend | job id | status | QPU s | chip qubits | depth | 2q gates |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09T05:23:40+00:00 | 25 | ibm_kingston | `dbnlmhlt7fqc73ajrg40` | COMPLETED | 2 | 156 | 797 | 3160 |
| 2026-11-09T05:24:48+00:00 | 25 | ibm_marrakesh | `dbnltv5t7fqc73ajrr4g` | COMPLETED | 3 | 156 | 797 | 3160 |
| 2026-11-09T05:37:48+00:00 | 25 | ibm_marrakesh | `dboli8oqtr0c738d1jkg` | COMPLETED | 3 | 156 | 797 | 3160 |
| 2026-11-09T05:43:22+00:00 | 25 | ibm_marrakesh | `dbolobc4148s739t7km0` | COMPLETED | 3 | 156 | 797 | 3160 |
| 2026-11-09T05:43:45+00:00 | 25 | ibm_marrakesh | `dbolquoou4as73csad3g` | COMPLETED | 3 | 156 | 797 | 3160 |
| 2026-11-09T05:44:08+00:00 | 25 | ibm_marrakesh | `dbolr4gou4as73csadg0` | COMPLETED | 3 | 156 | 797 | 3160 |

## shor_sim_21q

Notebook: `results/shor_sim_21q/Shor.ipynb`

    CUDA-Q target     : qpp-cpu  (qpp-cpu)
    Factors by trial division (reference): (7, 17)
    Algorithm                           : Shor's algorithm (integer factorization)
    Number of qubits                    : 21
    CUDA-Q target                       : qpp-cpu
    Number of shots                     : 1000
    Total wall time of execution cell   : 229.6732 s
    Expected result                     : 119 = 7 x 17 (trial division)
    Actual result                       : 119 = 7 x 17

## hhl_sim_19q

Notebook: `results/hhl_sim_19q/HHL.ipynb`

    CUDA-Q target     : qpp-cpu  (qpp-cpu)
    Expected result
    Algorithm                              : HHL (quantum linear systems)
    Number of qubits                       : 19
    CUDA-Q target                          : qpp-cpu
    Number of shots                        : 20000
    Total wall time of execution cell      : 1773.2204 s

## noise_sim_1000q

Notebook: `results/noise_sim_1000q/Noise_Simulation.ipynb`

    Algorithm                         : Quantum noise simulation
    Number of qubits                  : 1000
    CUDA-Q target                     : stim
    Number of shots                   : 10000
    Total wall time of execution cell : 4.2593 s
    Actual result                     : Hellinger F = 0.0003, population = 0.0003, X parity = 0.0088
