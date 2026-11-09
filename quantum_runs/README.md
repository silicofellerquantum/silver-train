# quantum_runs

The 8 textbook notebooks (repository root) run at their **maximum qubit count**: on
**IBM Quantum hardware** (156-qubit IBM Heron chips) where the notebook supports hardware,
and on the CPU simulator where it does not. Nothing outside this folder was changed: each run
executes an in-memory copy of a notebook with only its Section 3 parameters overridden, and
saves the executed copy here.

Results: [`results/summary.md`](results/summary.md), one folder per run in `results/`.

## Status: complete (2026-11-09)

All runs finished; headline results are in [`results/summary.md`](results/summary.md).
IBM usage: 20 jobs, 95 QPU seconds in total (each account has 600 s per 28 days).

| Run | Qubits | Result |
|---|---|---|
| `qft_ibm_156q` | 156 | hardware result complete (4000 distinct outcomes from 4000 shots); the notebook's analysis section stops above 63 qubits (64-bit integers) |
| `qft_ibm_63q` | 63 | full notebook analysis: 0% of shots on the 8 expected peaks (ideal 100%) |
| `grover_ibm_156q` | 156 | 1 iteration: P(marked) 0/2000, theory ~0 |
| `vqe_ibm_156q` | 156 | E = -43.65 vs exact -165.04 |
| `qaoa_ibm_156q` | 156 | approximation ratio 0.46 (exact max cut 156), random assignment 0.5 |
| `qml_ibm_156q` | 156 | test accuracy 50% (random guessing) |
| `shor_sim_21q` | 21 | 119 = 7 × 17, 94% of shots yield the factors |
| `shor_sim_24q` | 24 | 221 = 13 × 17, 95.6% of shots yield the factors |
| `hhl_sim_19q` | 19 | solution fidelity 0.9952 |
| `noise_sim_1000q` | 1000 | GHZ with 1% depolarizing noise per gate: Hellinger fidelity 0.0003 |

At the full width of the chip every hardware result is dominated by noise: the circuits have
roughly 1,000-116,000 two-qubit gates, each failing ~0.3-1% of the time. The same pipeline
gives correct answers on small circuits (checked on a noisy and a noiseless local copy of the
chip), and the simulator results are correct, so the hardware results show where current
devices stop, not a problem in the code. A qubit sweep (e.g. 4-32 qubits) on the same chips
would show where each algorithm still works.

Runs were paused on 2026-11-07 and resumed on 2026-11-09; the five jobs in flight finished at
IBM during the pause and their results were reused, not resubmitted.

## Runs

| Run | Notebook | Where | Qubits | Settings that differ from the notebook defaults |
|---|---|---|---|---|
| `qft_ibm_156q` | QFT | IBM | 156 | `N_QUBITS=156` |
| `qft_ibm_63q` | QFT | IBM | 63 | `N_QUBITS=63` (largest size the notebook's analysis section supports) |
| `grover_ibm_156q` | Grover | IBM | 156 | `GROVER_ITERATIONS=1` (the optimal ~π/4·2^78 iterations cannot be built), no iteration sweep |
| `vqe_ibm_156q` | VQE | IBM | 156 | `OPTIMIZER="Powell"`, 5 energy evaluations, 4000 shots per term group |
| `qaoa_ibm_156q` | QAOA | IBM | 156 | `GRAPH="ring"` (156 edges), 6 evaluations, 4000 shots |
| `qml_ibm_156q` | QML | IBM | 156 | 12 × 13 images, 2 train / 2 test images, `OPTIMIZER="Powell"`, 3 iterations |
| `shor_sim_21q` | Shor | CPU simulator | 21 | factor 119 = 7 × 17 |
| `shor_sim_24q` | Shor | CPU simulator | 24 | factor 221 = 13 × 17 |
| `hhl_sim_19q` | HHL | CPU simulator | 19 | random 256 × 256 system, 10 clock qubits |
| `noise_sim_1000q` | Noise_Simulation | CPU (stabilizer) | 1000 | `N_QUBITS=1000` |

Simulator sizes are the largest that finish in about an hour on the 4-core machine used (Shor at 24 qubits took 59 min): each
extra qubit doubles the simulation time, and the notebooks' own estimates were ~5.4 days per
attempt for Shor at 27 qubits (391 = 17 × 23, the largest that fits in 16 GB RAM), ~7.9 h at
24 qubits, and ~4.6 days for HHL at 23 qubits. On a GPU (the notebooks' intended target) the
same notebooks reach larger sizes.

Why three notebooks are not on IBM: they raise an error for `BACKEND="hardware"` by design.
Shor and HHL build their circuits from exact custom unitaries (permutation matrices and matrix
exponentials) that hardware cannot execute, and Noise_Simulation simulates noise models
(on hardware the noise is physical). Every IBM job uses all 156 qubits of the chip, so the
results at this size are dominated by hardware noise - this is the expected outcome and the
purpose of the scaling run.

Optimizer budgets are small because every optimizer step is one IBM job that waits in IBM's
queue; COBYLA (the notebook default) first evaluates once per parameter, which would be
~470 hardware jobs for VQE at 156 qubits, so Powell (also offered by the notebooks) is used.

## How it works

| File | Purpose |
|---|---|
| `run_all.py` | the run list above; starts all runs in parallel, each with its own group of IBM accounts |
| `run_notebook.py` | executes one notebook with parameter overrides (like `dev/notebook_builders/run_variant.py`) and adds one first cell that installs `ibm_shim` |
| `ibm_shim.py` | makes `cudaq.has_target("ibm")` true and sends `cudaq.sample` / `cudaq.observe` to IBM; rotates API keys; logs and caches every job |
| `kernel_trace.py` | records a CUDA-Q kernel call as a Qiskit circuit (exact angles, any size) |
| `cudaq_export.py` | CUDA-Q compiler route to OpenQASM 2; used to cross-check `kernel_trace` on small circuits |
| `make_summary.py` | writes `results/summary.md` |

The IBM submission uses the CUDA-Q → IBM bridge
[cudaq-ibm-backend](https://github.com/brunopacheco1/cudaq-ibm-backend): its `to_isa`
(transpile for the chip), `IBMQuantumClient` (IBM REST API) and `samples_to_counts`
(IBM results → CUDA-Q bit order). Its own kernel exporter (`cudaq.synthesize` +
`cudaq.translate(format="openqasm2")`) fails for these notebooks in CUDA-Q 0.16 (kernels with
list arguments, sub-kernels and large loops), which is why `kernel_trace.py` exists.
`cudaq.observe` on hardware is computed from samples: Pauli terms are grouped into
qubit-wise-commuting sets, each set is measured in its basis, and the expectation is the
coefficient-weighted parity average.

### API keys

Keys are read in memory from the spreadsheet given by `QR_KEYS_XLSX` (columns S.No, Email,
API Key, CRN). They are never printed, logged or written to disk; only the S.No appears in
logs. Accounts are used in S.No order inside each run's group; when IBM reports an account's
QPU time is used up, the next account is used. Rows that share a CRN share one allowance.
The `.gitignore` here blocks spreadsheets and `.env` files from being committed.

### Re-running

```bash
pip install "cuda-quantum-cu12==0.16.0" --no-deps   # CPU-only install; plus: astpretty numpy scipy requests
pip install qiskit qiskit-ibm-runtime qiskit-aer matplotlib psutil nbformat nbclient ipykernel openpyxl python-dotenv
pip install --no-deps -e /path/to/cudaq-ibm-backend
QR_KEYS_XLSX=/path/to/API_KEYS.xlsx python quantum_runs/run_all.py            # everything
QR_KEYS_XLSX=/path/to/API_KEYS.xlsx python quantum_runs/run_all.py qft_ibm_156q
python quantum_runs/make_summary.py
```

Completed IBM results are cached in `results/ibm_raw/`, so a re-run never submits the same
circuit twice. `QR_MODE=fake` runs the same pipeline on a noisy local copy of the chip and
`QR_MODE=transpile` only reports circuit sizes (no IBM calls).
