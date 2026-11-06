# CUDA-Q Quantum Algorithm Benchmarking Suite

Educational, executable Jupyter notebooks that demonstrate, visualize and benchmark quantum algorithms with
**NVIDIA CUDA-Q**. Every notebook explains the problem and the mathematics, lets you change all inputs in one cell,
runs on CPU, NVIDIA GPU(s) or quantum-hardware targets, monitors RAM and GPU memory live, and ends with a benchmark
report built only from real measurements.

> **Status:** all 8 notebooks are complete. They are committed **with their outputs** from a full run at default
> settings on a CPU-only test machine (see [Validation results](#validation-results-cpu-only-test-machine)); running
> them on your hardware replaces those outputs with your own measurements.

| # | Notebook | Default problem | Status |
|---|---|---|---|
| 1 | `VQE.ipynb` | Ground state of a 15-qubit transverse-field Ising chain (H2 molecule as built-in example) | done |
| 2 | `QAOA.ipynb` | Max-Cut on a 15-node random graph | done |
| 3 | `Shor.ipynb` | Factor N = 15 (12 qubits) + resource / reverse estimation for larger N | done |
| 4 | `Grover.ipynb` | Find 1 marked item among 2^15 | done |
| 5 | `HHL.ipynb` | Solve a 4x4 linear system A x = b (7 qubits) | done |
| 6 | `QML.ipynb` | QCNN (quantum convolutional neural network) classifying 2x4-pixel line images (8 qubits; 3x5 = 15 qubits on a GPU) | done |
| 7 | `Noise_Simulation.ipynb` | 15-qubit GHZ state with depolarizing / bit-flip / phase-flip / amplitude-damping noise | done |
| 8 | `QFT.ipynb` | QFT of a 15-qubit periodic state, compared with NumPy's FFT | done |

## Teaching material

* [`presentations/QFT_Lecture.pdf`](presentations/QFT_Lecture.pdf) - a 55-slide lecture (16:9, vector graphics) that
  explains the Quantum Fourier Transform to first-year engineering students: Fourier analysis, complex numbers,
  qubits, the QFT definition, the circuit derived step by step, worked examples, CUDA-Q results and exercises with
  answers. Source: `presentations/qft_source/` (`python make_figures.py` regenerates the figures with CUDA-Q, then
  `latexmk -pdf qft_lecture.tex` builds the slides; needs TeX Live with beamer and quantikz).
* [`book/QFT_Chapter.pdf`](book/QFT_Chapter.pdf) - book chapter "QFT": a 23-page step-by-step course from roots of
  unity to Shor's algorithm, with worked examples, common traps, strategies, a map from every assessment question to
  the section that teaches it, and practice problems with answers. Source: `book/qft_chapter_source/qft_chapter.tex`.
* [`assessments/QFT_Question_Paper.pdf`](assessments/QFT_Question_Paper.pdf) and the separate
  [`assessments/QFT_Answer_Book.pdf`](assessments/QFT_Answer_Book.pdf) - an 80-question test
  on the QFT algorithm: 4 sections x 20 questions (A Mathematics for the QFT, B Quantum Mechanics, C QFT
  Basics, D Advanced QFT), every question single-correct with 4 options, marking +3 / 0 / -1, 3 hours, 240 marks.
  Formats: single-correct MCQ, matching lists, Assertion-Reason, Statement I/II and paragraph questions. The paper
  ends with an OMR-style response sheet; the Answer Book has the key, topic map, quick-marking grid and the
  "Answer Bank": for every question the concepts needed, a full step-by-step derivation, why each wrong option is wrong, and a takeaway (explanations in `bank_explain.py`). Source: `assessments/qft_test_source/` - edit the questions in `bank_*.py`,
  check them with `python verify_answers.py --cudaq` (recomputes every key with NumPy and CUDA-Q) and rebuild both
  PDFs with `python generate_papers.py` (also writes `answer_key.csv`).
* [`presentations/QAOA_Lecture.pdf`](presentations/QAOA_Lecture.pdf) - a 58-slide lecture on the Quantum Approximate
  Optimization Algorithm in the same style: optimization and Max-Cut, spins and the cost Hamiltonian, Pauli algebra and
  gates, the QAOA circuit, the exact $p = 1$ analysis, sign conventions, a line-by-line walkthrough of `QAOA.ipynb`,
  recorded and re-executed results, limits, noise and current research, exercises with answers. Source:
  `presentations/qaoa_source/` (`python make_figures.py` recomputes every figure and number with CUDA-Q, then
  `latexmk -pdf qaoa_lecture.tex`).
* [`book/QAOA_Chapter.pdf`](book/QAOA_Chapter.pdf) - book chapter "QAOA" with worked examples, common traps, a map from
  every assessment question to its section, and practice problems with answers. Source: `book/qaoa_chapter_source/`.
* [`assessments/QAOA_Question_Paper.pdf`](assessments/QAOA_Question_Paper.pdf) and
  [`assessments/QAOA_Answer_Book.pdf`](assessments/QAOA_Answer_Book.pdf) - an 80-question QAOA test in the QFT format
  (A Mathematics and Graph Theory for QAOA, B Quantum Mechanics, C QAOA Basics, D Advanced QAOA and Implementation;
  4 x 20 single-correct questions, +3 / 0 / -1, 3 hours, 240 marks, OMR response sheet). The Answer Book adds a
  topic-wise map and, for every question, the concepts required, relevant formulae, a step-by-step explanation, why
  each wrong option is wrong and a key takeaway. Source: `assessments/qaoa_test_source/` - check with
  `python verify_answers.py --cudaq` (176 checks, including a rebuild of the notebook's CUDA-Q kernel) and rebuild with
  `python generate_papers.py`.
* [`presentations/VQE_Lecture.pdf`](presentations/VQE_Lecture.pdf) - a lecture on the Variational Quantum Eigensolver in the
  same style: eigenvalue problems, molecular Hamiltonians and second quantisation, Jordan-Wigner / Bravyi-Kitaev, Pauli
  decomposition and measurement, the variational principle, Hartree-Fock / UCC / hardware-efficient ansaetze, optimizers,
  worked H$_2$ and Ising examples, a line-by-line walkthrough of `VQE.ipynb`, recorded and re-executed results, shots, noise
  and current research. Source: `presentations/vqe_source/` (`python make_figures.py` recomputes every figure and number
  with CUDA-Q; the H$_2$ dissociation curve also needs PySCF + OpenFermion).
* [`book/VQE_Chapter.pdf`](book/VQE_Chapter.pdf) - book chapter "VQE" with worked examples, common traps, a question map and
  practice problems. Source: `book/vqe_chapter_source/`.
* [`assessments/VQE_Question_Paper.pdf`](assessments/VQE_Question_Paper.pdf) and
  [`assessments/VQE_Answer_Book.pdf`](assessments/VQE_Answer_Book.pdf) - an 80-question VQE test in the QFT format
  (A Mathematical and Quantum Chemistry Foundations, B Quantum Computing Foundations, C VQE Fundamentals, D Advanced VQE and
  Implementation; +3 / 0 / -1, 3 hours, 240 marks, OMR sheet; Answer Book with key, topic map, quick-marking grid and full
  explanations). Source: `assessments/vqe_test_source/` - check with `python verify_answers.py --chem --cudaq` (157 checks:
  NumPy, PySCF/OpenFermion and the notebook's CUDA-Q kernels) and rebuild with `python generate_papers.py`.
  Note: the notebook's H$_2$ coefficients correspond to a bond length of 0.740 Angstrom (verified with PySCF); the notebook
  text, which previously said 0.7414 Angstrom, has been corrected.

## Structure of the notebooks

`QML.ipynb` is deliberately **short and easy to modify**: one input cell, one small CUDA-Q kernel driven by a Python
layout function, a plain training loop, plots and a short report - without the shared utility cell and without
memory monitoring. The other seven notebooks follow this full structure:

1. Problem statement and mathematical explanation (LaTeX)
2. Import libraries (with installation instructions and a version check)
3. **Input parameters** - the only cell you need to edit
4. Example inputs (a working default plus a table of alternatives)
5. Benchmark utilities and backend initialization
6. Expected output *before* execution (exact expectations, memory formula, calibrated time estimate)
7. Quantum circuit generation (drawing limited to 10 qubits, full-circuit gate statistics, export)
8. Execution cell with live memory monitoring
9. Results and visualization
10. Final benchmark report (also saved as JSON + PNG in `exports/<notebook>/`)

## Installation

Requirements: Linux x86_64/aarch64 (GPU acceleration) or macOS on Apple silicon (CPU only), Python >= 3.11, pip >= 24.

```bash
python3 -m venv cudaq-env
```

```bash
source cudaq-env/bin/activate
```

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

```bash
jupyter lab
```

A GPU is not required: CPU simulators (`qpp-cpu`, `density-matrix-cpu`, `stim`) are always available.

### CUDA-Q setup, NVIDIA drivers and CUDA compatibility

* `pip install cudaq` detects the CUDA runtime/driver on the machine at install time and installs the matching
  binary package: `cuda-quantum-cu12` (CUDA 12.x) or `cuda-quantum-cu13` (CUDA 13.x). If no CUDA installation is
  detected it installs the CUDA 13 build. Do not install two variants side by side.
* Driver requirement: CUDA 12.x builds need an NVIDIA Linux driver >= 525.60.13; CUDA 13.x builds need driver R580
  or newer. Check with `nvidia-smi` (top-right shows the highest CUDA version the driver supports). If CUDA-Q reports
  "CUDA driver version is insufficient", install the CUDA 12 build (`pip uninstall cudaq cuda-quantum-cu13` then
  `pip install cuda-quantum-cu12`) or update the driver.
* CUDA 13 no longer supports some older GPU architectures (Maxwell, Pascal, Volta); use the CUDA 12 build on those.
* Verify the installation:

```bash
python -c "import cudaq; print(cudaq.__version__); print('GPUs:', cudaq.num_available_gpus())"
```

* **Multi-GPU / multi-node** (state vector distributed over GPUs) needs MPI. The CUDA-Q package README recommends a
  conda environment with `openmpi>=5.0.3` and `mpi4py`, then running CUDA-Q's
  `distributed_interfaces/activate_custom_mpi.sh` (inside the installed package) once, and setting
  `export OMPI_MCA_opal_cuda_support=true OMPI_MCA_btl='^openib'`.

## Selecting a backend

In Section 3 of each notebook:

```python
BACKEND = "auto"              # "auto" | "cpu" | "gpu" | "multi-gpu" | "hardware"
SIMULATION_METHOD = "statevector"
GPU_PRECISION = "fp64"        # "fp32" halves GPU memory
```

| Choice | CUDA-Q target | Notes |
|---|---|---|
| `cpu` + `statevector` | `qpp-cpu` | OpenMP, fp64 |
| `cpu` + `density_matrix` | `density-matrix-cpu` | mixed states / noise, memory 16 x 4^N bytes |
| `cpu` + `stabilizer` | `stim` | Clifford circuits + Pauli noise only (noise notebook) |
| `gpu` + `statevector` | `nvidia` (option `fp32`/`fp64`) | cuStateVec, one GPU |
| `gpu` + `trajectory` | `nvidia` + noise model | Monte-Carlo noise trajectories (noise notebook) |
| `gpu` + `mps` | `tensornet-mps` | matrix product state |
| `gpu` + `tensornet` | `tensornet` | exact tensor-network contraction |
| `multi-gpu` + `statevector` | `nvidia` (option `mgpu,fp64`) | one state vector split over GPUs, requires `mpiexec` |
| `hardware` | `quantinuum`, `ionq`, `oqc`, `anyon`, `infleqtion`, ... | `HARDWARE_EMULATE = True` runs the provider's flow locally (no account, no cost) |

`"auto"` uses a GPU when CUDA-Q sees one, otherwise the CPU. Unsupported combinations stop with an explanation
(for example MPS on a CPU). Each notebook only offers the methods that make sense for it.

### Running on several GPUs or nodes

A Jupyter kernel is a single process, so it uses one GPU. To pool several GPUs, export the notebook as a script and
launch one MPI rank per GPU (with `BACKEND = "multi-gpu"` set in the script):

```bash
jupyter nbconvert --to script Grover.ipynb
```

```bash
mpiexec -np 4 python Grover.py
```

Every rank prints its own node and per-GPU memory lines, so multi-node memory use is reported per node.

## Changing the number of qubits

Edit Section 3. The parameter is `N_QUBITS` (VQE, QAOA, Grover, Noise, QFT), `N_TO_FACTOR` (Shor: qubits = 3 x bits
of N), `NUM_CLOCK_QUBITS` + matrix size (HHL) or `IMAGE_ROWS` x `IMAGE_COLS` (QML: one qubit per pixel).
Memory needed for the state:

| Method | Memory | 15 qubits | 30 qubits | 40 qubits |
|---|---|---|---|---|
| state vector fp64 | 16 x 2^N bytes | 0.5 MB | 17.2 GB | 17.6 TB |
| state vector fp32 | 8 x 2^N bytes | 0.26 MB | 8.6 GB | 8.8 TB |
| density matrix | 16 x 4^N bytes | 17.2 GB | - | - |

Section 6 of every notebook computes the memory exactly, compares it with free RAM / GPU memory and refuses runs
that exceed `MEMORY_SAFETY_FRACTION`. It also prints a calibrated execution-time estimate (rough, about +-3x).
For 40+ qubits use `SIMULATION_METHOD = "mps"` where the notebook offers it (QFT, Grover, QAOA, VQE) or multiple GPUs.

## Executing a notebook

Open it in Jupyter and run all cells (*Kernel -> Restart & Run All*), or headless:

```bash
jupyter nbconvert --to notebook --execute VQE.ipynb --output VQE_executed.ipynb --ExecutePreprocessor.timeout=3600
```

Outputs (circuit drawings as `.txt`/`.png`, the full circuit as QIR `.ll`, the benchmark report as `.json`/`.png`)
are written to `exports/<notebook>/` (ignored by git).

## What is measured and how

* **Timing:** circuit construction (Python -> Quake IR), compilation (MLIR passes + LLVM JIT) and quantum execution
  are separated with CUDA-Q's built-in span tracer on the first kernel launch; later launches are timed with a wall
  clock. If the tracer is unavailable, the report says so.
* **Memory:** a separate lightweight sampler process records host RAM (all processes), the notebook's own RSS
  (`psutil`) and every NVIDIA GPU's memory (NVML), so peaks are captured even while CUDA-Q holds the Python GIL.
  NVML values are device-wide (they include other processes on the GPU).
* Nothing is estimated or invented in the reports: unavailable measurements are reported as unavailable.

## Validation results (CPU-only test machine)

These are the outputs now stored in the notebooks: a full run of all 8 notebooks at their default settings on a
CPU-only container (4 vCPU Intel Xeon @ 2.1 GHz, 16.9 GB RAM, CUDA-Q 0.16.0, simulator `qpp-cpu` / `stim`, no GPU).
Your numbers will differ; GPU runs are much faster.

| Notebook | Result | Execution time |
|---|---|---|
| VQE (TFIM, 15 qubits, h = 0.5) | E = -15.05339 vs exact -15.08252 (0.19 % above, as the variational principle requires) | 70 s |
| QAOA (15 nodes, 30 edges, p = 2) | optimal cut 25 found, approximation ratio 0.82, P(optimal split) = 0.06 | 41 s |
| Shor (N = 15, a = 7) | 15 = 3 x 5 verified, 77 % of shots yield the factors | 0.07 s |
| Grover (15 qubits) | marked item in 2000/2000 shots; iteration sweep matches sin^2((2k+1)theta) to 8e-13 | 67 s |
| HHL (4x4) | fidelity 1.0000000000, relative error 1.7e-15; shot read-out within statistical error | 0.2 s |
| QML / QCNN (8 qubits, 20 + 20 images) | training accuracy 90 %, test accuracy 85 % (random guessing 50 %), loss 1.12 -> 0.67 | 3.4 s training |
| Noise (15-qubit GHZ, depolarizing p = 0.01, stim) | population 0.894 (lower bound 0.860), X-basis parity 0.691; noiseless check 1.000 | 0.25 s |
| QFT (15 qubits, period 8) | state fidelity 1.000000000000 vs NumPy FFT, max amplitude error 7.5e-16 | 0.9 s |

Also checked during development: VQE with the H2 example reaches chemical accuracy (error 1.3e-10 Ha in 22
evaluations); the noise notebook's density-matrix path (amplitude damping, mirror circuit) and the phase-flip case
(Z-basis population stays 1 while the X-basis parity drops to 0.56) behave as expected.

GPU, multi-GPU, MPS/tensor-network and real-hardware paths are implemented following the CUDA-Q 0.16 APIs but
could not be executed on the CPU-only test machine.

## Known limitations

* Classical simulation memory is exponential: state vectors need 16 x 2^N bytes (fp64), density matrices 16 x 4^N.
* **Shor:** modular multiplication is implemented as exact permutation unitaries (custom gates). This is correct
  and efficient for small N on simulators but is not a gate-level fault-tolerant circuit, and it cannot run on
  hardware targets. The notebook includes resource-estimation modes for larger N; it does not claim that simulators
  can factor large integers.
* **HHL:** uses the exact matrix exponential e^{iAt} as custom gates and amplitude encoding of b (simulators only).
  The full solution vector is read from the simulated state; on hardware only |x_i|^2 can be sampled.
* **Noise:** the noiseless state-vector simulators ignore noise models, so the noise notebook offers only
  `stabilizer` (stim, Pauli noise only), `density_matrix` (CPU) and `trajectory` (GPU). With CUDA-Q 0.16, noisy
  *sampling* on `density-matrix-cpu` was very slow (about 115 s for 10 qubits), so the notebook computes the noisy
  density matrix once and draws shot counts from its diagonal (statistically identical). General Kraus channels such
  as amplitude damping are much slower than Pauli channels on the density-matrix simulator.
* CUDA-Q 0.16 specifics handled in the code: `cudaq.draw("latex", kernel, args)` fails when arguments are given (ASCII
  drawings are used); OpenQASM 2 export fails for kernels with sub-kernels or list arguments (QIR export is used);
  in the kernel builder an amplitude-initialized `qalloc(vector)` placed after other registers received extra qubits
  (HHL initializes one combined register); selecting the `tensornet` target on a machine without a GPU aborts the
  Python process, so GPU targets are never selected when no GPU is visible.
* `multi-gpu` from inside Jupyter runs on one GPU; real pooling needs `mpiexec` (see above).
* Hardware targets support sampling only (no state vectors); real submissions need provider credentials and may cost money.
* Execution-time estimates are rough (calibrated per backend, extrapolated with 2^N or 4^N scaling).
* **QML / QCNN:** training runs the circuit once per image per optimizer step, so simulation cost dominates. In a test
  at 15 qubits (3x5 images, 16 training images, 60 COBYLA evaluations) each circuit took ~0.35 s on the 4-core test
  CPU and the accuracy reached only 69 %: the deeper 15-qubit QCNN needs more optimizer steps. That is why the QML
  default is 8 qubits; use a GPU (`TARGET = "nvidia"`) and raise `MAX_ITERATIONS` for 15+ qubits.

## Open work

* Validate the GPU, multi-GPU (MPI) and real-hardware paths on a machine with NVIDIA GPUs (the test machine had none).

## Development: regenerating the notebooks

The notebooks were generated from the scripts in `dev/notebook_builders/`. Shared code is defined once there
(`common_imports.py`, `common_utils.py`, `common_md.py`) and copied into every notebook so each notebook is
self-contained. You can edit the notebooks directly, but **regenerating overwrites them**:

```bash
cd dev/notebook_builders && python build_grover.py ../../Grover.ipynb
```

There is one builder per notebook (`build_vqe.py`, `build_qaoa.py`, `build_shor.py`, `build_grover.py`, `build_hhl.py`,
`build_qml.py`, `build_noise.py`, `build_qft.py`). Regenerated notebooks contain no outputs until they are run.

`run_variant.py` executes a notebook with overridden Section 3 parameters (useful for tests), for example:

```bash
python dev/notebook_builders/run_variant.py Grover.ipynb /tmp/grover_test.ipynb N_QUBITS=8
```
