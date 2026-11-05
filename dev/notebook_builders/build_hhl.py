from nbgen import md, code, common_imports, common_utils, write_notebook
from common_md import (title_md, INSTALL_MD, params_md, common_input_block, UTILS_MD,
                       backend_init_code, EXECUTION_MD)
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "HHL.ipynb"
ALLOWED = ["statevector", "density_matrix"]

cells = []
cells.append(md(title_md(
    5, "HHL - Quantum Linear Systems Algorithm", "Solving A x = b with quantum phase estimation in NVIDIA CUDA-Q",
    "This notebook implements the Harrow-Hassidim-Lloyd (HHL) algorithm: it validates the linear system, "
    "estimates the qubits needed, prepares |b>, runs phase estimation and eigenvalue inversion, recovers the "
    "solution from the quantum state and compares it with the classical solution - with real timing and memory "
    "measurements.")))

cells.append(md(r"""
---
## Section 1 - Problem statement and mathematical explanation

### What problem does the algorithm solve?
Solve a linear system $A\,x = b$ for $x$, where $A$ is a $d\times d$ matrix and $b$ a vector of length $d = 2^{n_b}$.
HHL does not output the numbers $x_i$; it prepares the **quantum state**

$$|x\rangle = \frac{A^{-1}|b\rangle}{\lVert A^{-1}|b\rangle\rVert},$$

whose amplitudes are proportional to the solution vector.

### Why is this problem important?
Linear systems are everywhere: simulations (finite elements, fluid dynamics), optimisation, machine learning
(least squares, kernel methods), network analysis. HHL is the prototype of quantum linear-algebra algorithms; for
sparse, well-conditioned systems it runs in time $\mathrm{poly}(\log d,\ \kappa)$ instead of $\mathrm{poly}(d)$,
an exponential speed-up in the dimension *if* only a property of $x$ (e.g. an expectation value $x^\dagger M x$) is needed.

### What is the classical approach?
Gaussian elimination costs $O(d^3)$; iterative methods (conjugate gradient) cost about $O(d\,s\,\kappa)$ for sparsity
$s$ and condition number $\kappa = |\lambda|_{\max}/|\lambda|_{\min}$. NumPy's `np.linalg.solve` is the reference here.

### How does the quantum algorithm solve it?
Write $|b\rangle = \sum_j \beta_j |u_j\rangle$ in the eigenbasis of the Hermitian matrix $A$ ($A|u_j\rangle = \lambda_j|u_j\rangle$). Then
$A^{-1}|b\rangle = \sum_j \frac{\beta_j}{\lambda_j}|u_j\rangle$. HHL builds exactly this with three registers:
**system** ($n_b$ qubits, holds $|b\rangle$), **clock** ($n_c$ qubits, eigenvalue estimates), **ancilla** (1 qubit).

1. **State preparation:** $|b\rangle$ is loaded into the system register (amplitude encoding).
2. **Quantum phase estimation** with $U = e^{iAt}$: Hadamards on the clock, controlled $U^{2^j}$, inverse QFT, giving
   $\sum_j \beta_j |u_j\rangle|k_j\rangle$ with $k_j \approx \lambda_j t\,2^{n_c}/(2\pi)$ (negative eigenvalues wrap
   around: values $k \ge 2^{n_c-1}$ mean $k - 2^{n_c}$).
3. **Eigenvalue inversion:** for every clock value $k$, a controlled rotation
   $R_Y(\theta_k)$, $\theta_k = 2\arcsin(C/\tilde\lambda_k)$ puts amplitude $C/\tilde\lambda_k$ on ancilla state $|1\rangle$.
   $C = 2\pi/(2^{n_c} t)$ is the smallest representable eigenvalue, so $|C/\tilde\lambda| \le 1$.
4. **Uncomputation:** inverse phase estimation returns the clock to $|0\rangle$.
5. **Post-selection:** measuring the ancilla in $|1\rangle$ (probability $P_{\text{success}} = C^2\lVert A^{-1}|b\rangle\rVert^2$)
   leaves the system in $|x\rangle$.

### Conditions required for HHL
* $A$ must be **Hermitian** - otherwise the notebook embeds it as $\begin{pmatrix}0 & A\\ A^\dagger & 0\end{pmatrix}$ (dimension doubles).
* $A$ must be **invertible** and **well conditioned**: cost and failure probability grow with $\kappa$.
* $e^{iAt}$ must be implementable efficiently (sparse or structured $A$). Here we use the **exact** matrix exponential as a
  custom gate - fine for simulating small systems, but on hardware it would be replaced by Hamiltonian simulation.
* $|b\rangle$ must be preparable efficiently, and the eigenvalues must be resolvable by $n_c$ clock qubits.
* The useful output is a **property** of $|x\rangle$; reading out all $d$ amplitudes needs about $d/\epsilon^2$ shots
  (tomography), which would cancel the speed-up.

### Measurement and solution recovery
* **State-vector simulation:** we read the amplitudes of the post-selected branch directly; then
  $x = \lVert b\rVert\,C^{-1}\times(\text{amplitudes})$ recovers the full vector, norm included.
* **Sampling (as on hardware):** shots with ancilla $=1$ and clock $=0$ give $|x_i|^2/\lVert x\rVert^2$ (magnitudes only;
  signs/phases need measurements in other bases). Their fraction estimates $P_{\text{success}}$.

### What is the expected solution?
Default example: a $4\times4$ symmetric matrix with eigenvalues $1, 2, 3, 4$ ($\kappa = 4$) and $b = (1, 0, 0, 0)$. With
$n_c = 4$ clock qubits and $t = \pi/8$ the eigenvalues map exactly onto clock values $k = 1, 2, 3, 4$, so HHL is exact:
fidelity 1 and $x = (0.520833, 0.145833, 0.229167, 0.104167)$.
"""))

cells.append(md(INSTALL_MD))
cells.append(common_imports("from scipy.linalg import expm                       # exact matrix exponential exp(i A t)\n"
                            "    import hashlib                                    # unique names for the custom gates"))

cells.append(md("""
---
## Section 3 - Input parameters

This is the **only cell you need to edit**. Every later cell reads these variables.
"""))
cells.append(code(r'''
# =====================================================================================
# Section 3 - INPUT PARAMETERS  (edit only this cell, then re-run the notebook)
# =====================================================================================

# ---- Problem definition: the linear system A x = b ---------------------------------------
MATRIX_A = "example"             # "example" (4x4, eigenvalues 1,2,3,4) | "random" | your matrix, e.g. [[2, 1], [1, 3]]
VECTOR_B = "example"             # "example" (= [1, 0, 0, 0]) | your vector with the same length as A, e.g. [1, 0]
RANDOM_SIZE = 4                  # "random": dimension d (padded to the next power of 2)
RANDOM_CONDITION_NUMBER = 4.0    # "random": condition number kappa of the random positive-definite matrix

# ---- Algorithm parameters ----------------------------------------------------------------
NUM_CLOCK_QUBITS = 4             # phase-estimation qubits n_c: eigenvalue resolution 2*pi/(2^n_c * t)
EVOLUTION_TIME = "auto"          # "auto" (smallest |eigenvalue| -> clock value 1) or a number t for U = exp(i A t)
''' + common_input_block(ALLOWED, "statevector", 20000, "exports/hhl")))

cells.append(md(params_md(r"""
| `MATRIX_A` / `VECTOR_B` | The linear system. | Any square real or complex matrix; non-Hermitian matrices are embedded (one extra qubit), non-power-of-2 sizes are padded with an identity block. Larger matrices need more system qubits ($n_b = \log_2 d$). |
| `RANDOM_SIZE` / `RANDOM_CONDITION_NUMBER` | Random symmetric positive-definite test system. | Larger $\kappa$ means smaller eigenvalues relative to the largest: more clock qubits are needed and the success probability drops (~$1/\kappa^2$). |
| `NUM_CLOCK_QUBITS` | Precision $n_c$ of phase estimation. | Each extra qubit halves the eigenvalue resolution and doubles the number of controlled rotations ($2^{n_c}-1$). Too few qubits -> eigenvalues are rounded -> lower fidelity. |
| `EVOLUTION_TIME` | Time $t$ in $U = e^{iAt}$. | `"auto"` maps the smallest $|\lambda|$ to clock value 1 (exact when eigenvalues are integer multiples of it). $|\lambda| t$ must stay below $\pi$ or eigenvalues alias. |
| `SHOTS` | Measurements for the sampling-based read-out. | Only shots with ancilla = 1 (fraction $P_{\text{success}}$) are useful; their statistical error ~ $1/\sqrt{P_{\text{success}}\cdot\text{SHOTS}}$. |
""")))

cells.append(md(r"""
---
## Section 4 - Example inputs

**Default example (runs as is):**

$$A = \begin{pmatrix} 2.5 & -0.5 & -1 & 0\\ -0.5 & 2.5 & 0 & -1\\ -1 & 0 & 2.5 & -0.5\\ 0 & -1 & -0.5 & 2.5\end{pmatrix},\qquad
b = \begin{pmatrix}1\\0\\0\\0\end{pmatrix},$$

built as $A = H\,\mathrm{diag}(1,2,3,4)\,H$ with the orthogonal $4\times4$ Hadamard-type matrix $H$, so its eigenvalues
are exactly 1, 2, 3, 4.

| Goal | Setting in Section 3 |
|---|---|
| 2x2 system | `MATRIX_A = [[1, -1/3], [-1/3, 1]]`, `VECTOR_B = [1, 0]` (eigenvalues 2/3 and 4/3) |
| Random well-conditioned system | `MATRIX_A = "random"`, `VECTOR_B = [1, 1, 1, 1]`, `RANDOM_CONDITION_NUMBER = 4` |
| Ill-conditioned system | `MATRIX_A = "random"`, `RANDOM_CONDITION_NUMBER = 20`, `NUM_CLOCK_QUBITS = 6` |
| Non-Hermitian matrix | `MATRIX_A = [[1, 2], [0, 1]]`, `VECTOR_B = [1, 1]` (embedded automatically) |
| More precision / qubits | `NUM_CLOCK_QUBITS = 8` (11 qubits in total for the 4x4 example) |
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 4 - Build, validate and condition the linear system; estimate the qubits
# ---------------------------------------------------------------------------------
H4 = 0.5 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])   # orthogonal (H4 @ H4 = I)
rng = np.random.default_rng(SEED)
if isinstance(MATRIX_A, str) and MATRIX_A.strip().lower() == "example":
    A_input = H4 @ np.diag([1.0, 2.0, 3.0, 4.0]) @ H4                      # eigenvalues exactly 1, 2, 3, 4
elif isinstance(MATRIX_A, str) and MATRIX_A.strip().lower() == "random":
    d = int(RANDOM_SIZE)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))                            # random orthogonal eigenvectors
    eigenvalues = np.linspace(1.0, float(RANDOM_CONDITION_NUMBER), d)        # condition number = max/min
    A_input = Q @ np.diag(eigenvalues) @ Q.T
else:
    A_input = np.array(MATRIX_A, dtype=complex)
if isinstance(VECTOR_B, str) and VECTOR_B.strip().lower() == "example":
    b_input = np.zeros(A_input.shape[0]); b_input[0] = 1.0
else:
    b_input = np.array(VECTOR_B, dtype=complex).ravel()

# ---- validation ------------------------------------------------------------------------------
if A_input.ndim != 2 or A_input.shape[0] != A_input.shape[1]:
    raise ValueError(f"A must be a square matrix, got shape {A_input.shape}.")
if b_input.shape[0] != A_input.shape[0]:
    raise ValueError(f"Dimension mismatch: A is {A_input.shape[0]}x{A_input.shape[1]} but b has length {b_input.shape[0]}.")
if np.linalg.norm(b_input) == 0:
    raise ValueError("b must not be the zero vector.")
if abs(np.linalg.det(A_input)) < 1e-12 or np.linalg.cond(A_input) > 1e12:
    raise ValueError("A is singular (or numerically singular): A x = b has no unique solution.")
x_classical = np.linalg.solve(A_input, b_input)                             # reference solution

# ---- make the system HHL-ready: Hermitian and power-of-2 dimension ---------------------------
hermitian = np.allclose(A_input, A_input.conj().T, atol=1e-10)
A_work, b_work, solution_slice, notes = A_input.astype(complex), b_input.astype(complex), slice(0, A_input.shape[0]), []
if not hermitian:                                                           # [[0, A], [A^dag, 0]] [y; x] = [b; 0]
    d0 = A_work.shape[0]
    A_work = np.block([[np.zeros((d0, d0)), A_work], [A_work.conj().T, np.zeros((d0, d0))]])
    b_work = np.concatenate([b_work, np.zeros(d0)])
    solution_slice = slice(d0, 2 * d0)                                       # x is the second half of the solution
    notes.append("A is not Hermitian -> embedded into [[0, A], [A^dagger, 0]] (dimension doubled)")
d_work = A_work.shape[0]
d_padded = 1 << (d_work - 1).bit_length()                                    # next power of 2
if d_padded != d_work:                                                       # pad with an identity block
    A_pad = np.eye(d_padded, dtype=complex)
    A_pad[:d_work, :d_work] = A_work
    A_work = A_pad
    b_work = np.concatenate([b_work, np.zeros(d_padded - d_work)])
    notes.append(f"dimension {d_work} is not a power of 2 -> padded to {d_padded} with an identity block")
n_b = int(np.log2(d_padded))
b_norm = float(np.linalg.norm(b_work))
lam = np.linalg.eigvalsh(A_work)                                             # real eigenvalues of the Hermitian matrix
lam_abs_min, lam_abs_max = float(np.min(np.abs(lam))), float(np.max(np.abs(lam)))
kappa = lam_abs_max / lam_abs_min

# ---- phase-estimation parameters ------------------------------------------------------------
n_c = int(NUM_CLOCK_QUBITS)
if n_c < 2:
    raise ValueError("NUM_CLOCK_QUBITS must be at least 2.")
k_max = 2 ** (n_c - 1) - 1                                                   # largest positive clock value (signed)
if isinstance(EVOLUTION_TIME, str) and EVOLUTION_TIME.strip().lower() == "auto":
    if kappa <= k_max:
        t_evolution = 2 * np.pi / (2 ** n_c * lam_abs_min)                   # smallest |lambda| -> clock value 1
        t_note = "auto: smallest |eigenvalue| mapped to clock value 1"
    else:
        t_evolution = 2 * np.pi * k_max / (2 ** n_c * lam_abs_max)           # largest |lambda| -> largest clock value
        t_note = (f"auto: kappa = {kappa:.2f} > {k_max} representable levels -> largest |eigenvalue| mapped to {k_max}; "
                  "small eigenvalues are clipped (increase NUM_CLOCK_QUBITS)")
else:
    t_evolution = float(EVOLUTION_TIME)
    t_note = "user-defined"
C_constant = 2 * np.pi / (2 ** n_c * t_evolution)                            # smallest representable |eigenvalue|
clock_values = lam * t_evolution * 2 ** n_c / (2 * np.pi)                    # ideal (real-valued) clock readings
total_qubits = n_c + n_b + 1

print("A (as given) =")
print(np.array2string(np.real_if_close(A_input), precision=4, suppress_small=True))
print(f"b (as given) = {np.real_if_close(b_input)}")
for note in notes:
    print("NOTE:", note)
print(f"\nHermitian: {hermitian},  dimension used: {d_padded} -> n_b = {n_b} system qubits")
print(f"Eigenvalues of A (used): {np.round(lam, 6).tolist()}")
print(f"Condition number kappa: {kappa:.4f}")
print(f"Evolution time t: {t_evolution:.6f} ({t_note});  C = {C_constant:.6f}")
print("Ideal clock values k_j = lambda_j t 2^n_c / (2 pi): "
      + ", ".join(f"{k:+.3f}{' (exact)' if abs(k - round(k)) < 1e-9 else ' (rounded -> error)'}" for k in clock_values))
if np.any(np.abs(clock_values) > k_max + 0.5):
    print("WARNING: some |k_j| exceed the representable range -> eigenvalues alias; reduce EVOLUTION_TIME.")
print(f"\nQubit requirement: n_b + n_c + 1 = {n_b} + {n_c} + 1 = {total_qubits} qubits")
print(f"Classical solution x = A^-1 b = {np.round(np.real_if_close(x_classical), 6)}")
'''))

cells.append(md(UTILS_MD))
cells.append(common_utils())
cells.append(code(backend_init_code(ALLOWED) + r'''
if backend_cfg["device"] == "hardware":
    raise ValueError("This HHL implementation uses exact matrix exponentials as custom gates and amplitude encoding, "
                     "which hardware targets cannot execute. Use a simulator backend.")
can_read_state = True                                                    # both allowed methods provide get_state()
'''))

cells.append(md(r"""
---
## Section 6 - Expected output (before execution)

**Expected graphs (Section 9):** quantum vs. classical solution components (normalised, plus the norm-recovered
vector from the state), sampled $|x_i|^2$ vs. exact, and the memory timeline.

**Expected numbers:** the classical solution (NumPy), the success probability
$P_{\text{success}} = \sum_j |\beta_j|^2 (C/\lambda_j)^2$ (exact when every $k_j$ is an integer) and a fidelity of 1 in that case.
If some $k_j$ are not integers, phase estimation rounds them and the fidelity drops below 1 - the cell below says which.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6 - Expected results, memory requirement and time estimate
# ---------------------------------------------------------------------------------
eigvals, eigvecs = np.linalg.eigh(A_work)
beta = eigvecs.conj().T @ (b_work / b_norm)                                 # |b> in the eigenbasis
p_success_ideal = float(np.sum(np.abs(beta) ** 2 * (C_constant / eigvals) ** 2))
x_expected_state = x_classical / np.linalg.norm(x_classical)
exact_mapping = bool(np.all(np.abs(clock_values - np.round(clock_values)) < 1e-9))
print("Expected result")
print(f"  Normalised solution |x>     : {np.round(np.real_if_close(x_expected_state), 6)}")
print(f"  Success probability (ideal) : {p_success_ideal:.6f}  (expected useful shots: {p_success_ideal * SHOTS:.0f} of {SHOTS})")
print(f"  Expected fidelity           : {'1 (all eigenvalues exactly representable)' if exact_mapping else '< 1 (eigenvalues are rounded by phase estimation)'}")

state_bytes = estimate_memory_bytes(total_qubits, backend_cfg["method"], backend_cfg["precision"])
print("\nExpected memory")
check_memory(state_bytes, backend_cfg, MEMORY_SAFETY_FRACTION, extra_copies=1)
rotations = 2 ** n_c - 1
gates = 2 * n_c + 2 * n_c + n_c * (n_c - 1) + 2 * (n_c // 2) + rotations * (1 + n_c)   # H, controlled U, QFTs, rotations
seconds_per_gate, timing_note = measure_gate_time(backend_cfg, total_qubits)
print("\nExpected execution time (ESTIMATE)")
print(f"  ~{gates} gates ({rotations} multi-controlled eigenvalue-inversion rotations, {2 * n_c} controlled exp(+-iAt 2^j))")
print(f"  Calibration: {timing_note}")
if seconds_per_gate is not None:
    print(f"  Estimated  : ~{format_duration(2 * gates * seconds_per_gate)} for the state run + the sampling run")
'''))

cells.append(md(r"""
---
## Section 7 - Quantum circuit generation

The circuit is assembled with CUDA-Q's **kernel builder** because every controlled power $U^{2^j} = e^{iAt2^j}$ and
its inverse is a separate custom gate (registered with `cudaq.register_operation`; CUDA-Q expects these matrices with
the first target qubit as the most significant bit, matching our convention).

Qubit layout (one register, qubit 0 first): `clock[0..n_c-1]`, `system[0..n_b-1]`, `ancilla`. The initial state
$|0\ldots0\rangle_{\text{clock}}\otimes|b\rangle\otimes|0\rangle_{\text{anc}}$ is loaded in one step with `qalloc(vector)`.
(In CUDA-Q 0.16 an amplitude-initialised register allocated *after* other registers in the builder received extra
qubits, so the whole register is initialised at once.)
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7a - HHL circuit construction with the CUDA-Q kernel builder
# ---------------------------------------------------------------------------------
gate_uid = hashlib.sha1(np.r_[A_work.real.ravel(), A_work.imag.ravel(), t_evolution, n_c].tobytes()).hexdigest()[:10]
U_step = expm(1j * A_work * t_evolution)                                     # U = exp(i A t), exact (unitary)

def build_hhl_kernel(clock_qubits=None, measure=False, for_statistics=False):
    """HHL circuit; returns the builder kernel. clock_qubits < n_c is only used for drawings.
    for_statistics=True replaces the amplitude encoding by plain qubits (cudaq.estimate_resources cannot
    count amplitude-initialised registers); all other gates are identical."""
    nc = n_c if clock_qubits is None else clock_qubits
    t_used = t_evolution if clock_qubits is None else t_evolution * 2 ** (n_c - nc)   # keep C for drawings
    n_tot = nc + n_b + 1
    U = U_step if clock_qubits is None else expm(1j * A_work * t_used)
    kernel = cudaq.make_kernel()
    initial = np.kron(np.kron(np.eye(2 ** nc)[0], b_work / b_norm), np.array([1.0, 0.0]))   # |0>_clock |b> |0>_anc
    reg = kernel.qalloc(n_tot) if for_statistics else kernel.qalloc(bit_reverse_order(initial.astype(complex), n_tot))
    clock = [reg[i] for i in range(nc)]
    system = [reg[nc + i] for i in range(n_b)]
    ancilla = reg[n_tot - 1]
    # ---- 1) quantum phase estimation -------------------------------------------------------------
    for q in clock:
        kernel.h(q)                                                          # clock superposition
    for j in range(nc):
        name = f"hhl_{gate_uid}_{nc}_u{j}"
        cudaq.register_operation(name, np.linalg.matrix_power(U, 2 ** j))   # U^(2^j) = exp(i A t 2^j)
        getattr(kernel, name)(clock[nc - 1 - j], *system)                    # controlled by the clock bit of weight 2^j
    for j in range(nc // 2):                                                 # inverse QFT on the clock register
        kernel.swap(clock[j], clock[nc - 1 - j])
    for jj in range(nc):
        j = nc - 1 - jj
        for kk in range(nc - 1 - j):
            k = nc - 1 - kk
            kernel.cr1(-np.pi / 2 ** (k - j), clock[k], clock[j])
        kernel.h(clock[j])
    # ---- 2) eigenvalue inversion: R_Y(2 arcsin(C / lambda_k)) on the ancilla for every clock value k ---
    c_used = 2 * np.pi / (2 ** nc * t_used)
    for k_val in range(1, 2 ** nc):
        signed = k_val - 2 ** nc if k_val >= 2 ** (nc - 1) else k_val       # two's complement: negative eigenvalues
        lam_k = 2 * np.pi * signed / (2 ** nc * t_used)
        theta = 2 * np.arcsin(np.clip(c_used / lam_k, -1.0, 1.0))
        zero_bits = [i for i in range(nc) if not (k_val >> (nc - 1 - i)) & 1]   # clock[0] is the most significant bit
        for i in zero_bits:
            kernel.x(clock[i])                                               # select |k> with X "polarity" gates
        kernel.cry(theta, clock, ancilla)                                    # multi-controlled rotation
        for i in zero_bits:
            kernel.x(clock[i])
    # ---- 3) uncompute phase estimation (QFT, inverse controlled powers, H) ------------------------
    for j in range(nc):
        kernel.h(clock[j])
        for k in range(j + 1, nc):
            kernel.cr1(np.pi / 2 ** (k - j), clock[k], clock[j])
    for j in range(nc // 2):
        kernel.swap(clock[j], clock[nc - 1 - j])
    for jj in range(nc):
        j = nc - 1 - jj
        name = f"hhl_{gate_uid}_{nc}_uinv{j}"
        cudaq.register_operation(name, np.linalg.matrix_power(U.conj().T, 2 ** j))   # U^-(2^j)
        getattr(kernel, name)(clock[nc - 1 - j], *system)
    for q in clock:
        kernel.h(q)
    if measure:
        kernel.mz(reg)                                                       # bitstring = clock | system | ancilla
    return kernel


def postselected_amplitudes(state_vector):
    """Amplitudes of the system register in the branch clock = 0, ancilla = 1 (system qubit 0 = MSB)."""
    tensor = np.asarray(state_vector).reshape([2] * total_qubits).transpose(list(range(total_qubits))[::-1])
    return tensor[(0,) * n_c + (slice(None),) * n_b + (1,)].reshape(-1)

print("HHL circuit builder defined.")
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7b - Circuit visualization (at most 10 qubits) and full-circuit statistics
# ---------------------------------------------------------------------------------
max_draw = min(int(DRAW_QUBITS), 10)
draw_clock = min(n_c, max_draw - n_b - 1, 3)                                # at most 3 clock qubits keep it readable
if draw_clock < 1:
    print(f"Drawing skipped: the system register alone needs {n_b} qubits.")
else:
    print(f"Representative drawing with {draw_clock} clock qubits (the executed circuit has {n_c}); "
          "'hhl_..._u' boxes are the controlled exp(iAt 2^j) gates.")
    show_circuit(build_hhl_kernel(clock_qubits=draw_clock, measure=True), (), draw_clock + n_b + 1, total_qubits,
                 title="HHL circuit.", export=EXPORT_CIRCUIT, formats=EXPORT_FORMATS, out_dir=EXPORT_DIR,
                 basename="hhl", full_kernel=build_hhl_kernel(measure=True), full_args=())
stats = circuit_statistics(build_hhl_kernel(measure=True, for_statistics=True))
print(f"\nFULL circuit statistics ({total_qubits} qubits; the amplitude encoding of |b> is not counted):")
for key, value in stats.items():
    print(f"  {key:<32}: {value}")
'''))

cells.append(md(r"""
---
## Section 8 - Execution with live memory monitoring

Running the next cell:
1. initialises the selected backend,
2. builds the HHL circuit (construction time = builder calls + custom-gate registration),
3. runs it twice: once without measurement to read the full state (amplitude read-out), once with `SHOTS`
   measurements (read-out as on hardware, with post-selection),
4. monitors host RAM and GPU memory during the whole cell,
5. prints a short summary.
""" + EXECUTION_MD))
cells.append(code(r'''
# =====================================================================================
# >>> EXECUTION CELL - HHL with live memory monitoring <<<
# =====================================================================================
execution_ok, execution_error = False, None
monitor = MemoryMonitor(MEMORY_SAMPLE_INTERVAL_S, live=LIVE_MEMORY_DISPLAY)
monitor.start()
cell_start = time.perf_counter()
try:
    # Step 1 - initialise the backend
    initialize_backend(backend_cfg, seed=SEED)
    # Step 2 - build the circuits (construction measured directly: builder + matrix registration)
    t0 = time.perf_counter()
    kernel_state = build_hhl_kernel(measure=False)
    kernel_sample = build_hhl_kernel(measure=True)
    construction_time = time.perf_counter() - t0
    # Step 3a - full state (amplitude read-out); first launch traced
    state, phases = profile_first_launch(lambda: cudaq.get_state(kernel_state))
    state_vector = np.array(state)
    if state_vector.ndim == 2:                                               # density-matrix method: pure state -> vector
        evals, evecs = np.linalg.eigh(state_vector)
        state_vector = evecs[:, -1] * np.sqrt(max(evals[-1], 0.0))
    del state
    # Step 3b - sampling read-out with post-selection on ancilla = 1 and clock = 0
    t0 = time.perf_counter()
    counts = cudaq.sample(kernel_sample, shots_count=SHOTS)
    sampling_time = time.perf_counter() - t0
    execution_ok = True
except Exception as exc:
    execution_error = exc
    print(f"\nEXECUTION FAILED: {type(exc).__name__}: {exc}")
    print("Hints: reduce NUM_CLOCK_QUBITS or the matrix size, or choose another BACKEND (README).")
finally:
    memory = monitor.stop()
    total_wall = time.perf_counter() - cell_start

if execution_ok:
    phases["construction_s"] = construction_time
    phases["total_wall_s"] = total_wall
    amplitudes = postselected_amplitudes(state_vector)                       # proportional to x (padded/embedded system)
    p_success_state = float(np.linalg.norm(amplitudes) ** 2)
    x_quantum_full = amplitudes * b_norm / C_constant                         # norm recovery: x = ||b|| / C * amplitudes
    x_quantum = x_quantum_full[solution_slice]                                # undo embedding / padding
    if backend_cfg["method"] == "density_matrix":                            # rho has no global phase: align it for display
        overlap = np.vdot(x_quantum, x_classical)
        x_quantum = x_quantum * (overlap / abs(overlap) if abs(overlap) > 0 else 1.0)
        print("Note: a density matrix carries no global phase; the recovered vector's phase is aligned with NumPy's.")
    x_quantum_unit = x_quantum / np.linalg.norm(x_quantum)
    fidelity = float(abs(np.vdot(x_expected_state, x_quantum_unit)) ** 2)
    relative_error = float(np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical))
    post = {}                                                                # shots with clock = 0 and ancilla = 1
    for bits, c in counts.items():
        if bits[:n_c] == "0" * n_c and bits[-1] == "1":
            post[bits[n_c:n_c + n_b]] = post.get(bits[n_c:n_c + n_b], 0) + c
    accepted = sum(post.values())
    p_success_shots = accepted / SHOTS
    print(f"\nSuccess probability: state {p_success_state:.6f}, shots {p_success_shots:.4f} ({accepted} of {SHOTS}), ideal {p_success_ideal:.6f}")
    print(f"Fidelity |<x_classical|x_quantum>|^2 : {fidelity:.10f}")
    print(f"Relative error of the recovered vector: {relative_error:.3e}")
    print(f"Timing: construction {construction_time:.4f} s, compilation {fmt_seconds(phases['compilation_s'])}, "
          f"state run {fmt_seconds(phases['execution_s'])}, sampling run {sampling_time:.4f} s; total {total_wall:.2f} s\n")
print_memory_summary(memory)
'''))

cells.append(md(r"""
---
## Section 9 - Results and visualization

**Interpretation.** The quantum computer produces the *state* $|x\rangle$, proportional to the solution. On a
simulator we can read its amplitudes directly and even restore the norm. On real hardware we only get samples: the
histogram of post-selected shots estimates $|x_i|^2/\lVert x\rVert^2$ with statistical error ~ $1/\sqrt{\text{accepted shots}}$,
and recovering the signs/phases and the norm needs additional measurements (other bases, swap tests, or the
success probability). Reading all $d$ components requires about $d/\epsilon^2$ shots - which is why HHL is useful for
properties of $x$, not for printing $x$.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 9 - Quantum vs classical solution
# ---------------------------------------------------------------------------------
if not execution_ok:
    print("Skipped: the execution cell failed (see the error message above).")
else:
    d_out = x_classical.shape[0]
    exact_probs = np.abs(x_classical) ** 2 / np.sum(np.abs(x_classical) ** 2)
    # sampled probabilities for the original components (embedding/padding removed)
    sampled_full = np.zeros(2 ** n_b)
    for bits, c in post.items():
        sampled_full[int(bits, 2)] = c / max(accepted, 1)                    # system qubit 0 = most significant bit
    sampled = sampled_full[solution_slice]
    sampled = sampled / sampled.sum() if sampled.sum() > 0 else sampled
    print(f"{'i':>3} {'x classical':>22} {'x quantum (state)':>22} {'|x_i|^2 exact':>14} {'|x_i|^2 shots':>14}")
    for i in range(d_out):
        xc, xq = np.real_if_close(x_classical[i]), np.real_if_close(x_quantum[i])
        print(f"{i:>3} {str(np.round(xc, 6)):>22} {str(np.round(xq, 6)):>22} {exact_probs[i]:>14.4f} {sampled[i]:>14.4f}")
    stat_err = 1.0 / np.sqrt(max(accepted, 1))
    print(f"\nShot-based read-out: {accepted} accepted shots -> statistical error about +-{stat_err:.3f} per probability")

    fig = plt.figure(figsize=(12, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.85], hspace=0.45, wspace=0.28)
    ax1 = fig.add_subplot(grid[0, 0])
    idx = np.arange(d_out)
    ax1.bar(idx - 0.2, np.real(x_classical), width=0.38, color=INK_MUTED, label="classical (NumPy)")
    ax1.bar(idx + 0.2, np.real(x_quantum), width=0.38, color=BLUE, label="quantum (state read-out)")
    ax1.axhline(0, color=AXIS_COLOR, linewidth=1)
    ax1.set_xticks(idx, [f"x{i}" for i in idx])
    ax1.set_ylabel("solution component (real part)")
    ax1.set_title(f"Solution of A x = b: fidelity {fidelity:.6f}", loc="left")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="x", visible=False)
    span = max(np.max(np.abs(np.real(x_classical))), np.max(np.abs(np.real(x_quantum))))
    ax1.set_ylim(min(0, np.min(np.real(x_classical)), np.min(np.real(x_quantum))) * 1.2 - 0.05 * span, span * 1.45)
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.bar(idx - 0.2, exact_probs, width=0.38, color=INK_MUTED, label="exact |x_i|^2/||x||^2")
    ax2.bar(idx + 0.2, sampled, width=0.38, color=ORANGE, label=f"post-selected shots ({accepted})")
    ax2.set_xticks(idx, [f"x{i}" for i in idx])
    ax2.set_ylabel("probability")
    ax2.set_ylim(0, max(exact_probs.max(), sampled.max()) * 1.35)
    ax2.set_title("Hardware-style read-out (magnitudes only)", loc="left")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="x", visible=False)
    ax3 = fig.add_subplot(grid[1, :])
    monitor.plot(ax=ax3, title="Memory during the execution cell")
    plt.show()
'''))

cells.append(md(r"""
---
## Section 10 - Final benchmark report
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 10 - Final benchmark report
# ---------------------------------------------------------------------------------
if not execution_ok:
    print(f"No report: execution failed with {type(execution_error).__name__}: {execution_error}")
else:
    report = build_report("HHL (quantum linear systems)", backend_cfg, total_qubits, SHOTS, phases, memory,
                          extra_times={"Sampling run (execution)": sampling_time})
    report["System"] = f"{x_classical.shape[0]}x{x_classical.shape[0]} matrix, kappa = {kappa:.3f}" + (f" ({'; '.join(notes)})" if notes else "")
    report["Registers"] = f"{n_c} clock + {n_b} system + 1 ancilla = {total_qubits} qubits; t = {t_evolution:.5f}, C = {C_constant:.5f}"
    report["Expected result"] = f"x = {np.round(np.real_if_close(x_classical), 6).tolist()} (NumPy), P_success = {p_success_ideal:.4f}"
    report["Actual result"] = f"x = {np.round(np.real_if_close(x_quantum), 6).tolist()} (state read-out), P_success = {p_success_state:.4f} (shots: {p_success_shots:.4f})"
    report["Fidelity of |x>"] = f"{fidelity:.10f}"
    report["Relative error ||x_q - x_c|| / ||x_c||"] = f"{relative_error:.3e}"
    report["Shot read-out max |error| of |x_i|^2"] = f"{np.max(np.abs(sampled - exact_probs)):.4f} (statistical scale {stat_err:.3f})"
    final_fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(idx - 0.2, np.real(x_classical), width=0.38, color=INK_MUTED, label="classical")
    ax.bar(idx + 0.2, np.real(x_quantum), width=0.38, color=BLUE, label="HHL")
    ax.axhline(0, color=AXIS_COLOR, linewidth=1)
    ax.set_xticks(idx, [f"x{i}" for i in idx])
    ax.set_title(f"HHL {x_classical.shape[0]}x{x_classical.shape[0]}, {total_qubits} qubits on {cudaq.get_target().name}: fidelity {fidelity:.6f}", loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    final_fig.tight_layout()
    show_report(report, save=SAVE_REPORT, out_dir=EXPORT_DIR, name="hhl_benchmark_report", figure=final_fig)
    plt.show()
'''))

cells.append(md(r"""
---
### Things to try next
* `NUM_CLOCK_QUBITS = 3`: the eigenvalue 4 no longer fits ($\kappa = 4 > 3$ levels), the time $t$ is rescaled and the
  fidelity drops - phase-estimation precision limits HHL.
* `MATRIX_A = "random"`, `RANDOM_CONDITION_NUMBER = 20`: the success probability falls roughly like $1/\kappa^2$ and more
  clock qubits are needed.
* `MATRIX_A = [[1, 2], [0, 1]]`: a non-Hermitian matrix is embedded automatically (one extra system qubit).
"""))

write_notebook(OUT, cells, "HHL - CUDA-Q benchmark")
print("wrote", OUT)
