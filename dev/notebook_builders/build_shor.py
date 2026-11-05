from nbgen import md, code, common_imports, common_utils, write_notebook
from common_md import (title_md, INSTALL_MD, params_md, common_input_block, UTILS_MD,
                       backend_init_code, EXECUTION_MD)
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "Shor.ipynb"
ALLOWED = ["statevector", "density_matrix"]   # custom unitary gates verified on these simulators

cells = []
cells.append(md(title_md(
    3, "Shor's Algorithm - Integer Factorization", "Quantum order finding and factoring with NVIDIA CUDA-Q",
    "This notebook factors a small composite integer with Shor's algorithm: quantum phase estimation finds the "
    "order of a modular multiplication, and classical number theory turns the order into factors. It explains "
    "and computes the qubit requirements, provides resource-estimation modes for larger integers, and "
    "benchmarks the simulation.")))

cells.append(md(r"""
---
## Section 1 - Problem statement and mathematical explanation

### What problem does the algorithm solve?
**Integer factorization:** given a composite integer $N$, find non-trivial factors $p, q$ with $N = p\,q$.

### Why is this problem important?
The security of RSA public-key cryptography rests on factoring being hard. Shor's algorithm (1994) factors in
polynomial time on a fault-tolerant quantum computer, which is the main reason for today's migration to
post-quantum cryptography. It is also the textbook example of an exponential quantum speed-up.

### What is the classical approach?
Trial division costs $O(\sqrt N)$; the best known classical algorithm, the general number field sieve, runs in
sub-exponential time $\exp\!\big(O((\log N)^{1/3}(\log\log N)^{2/3})\big)$ - RSA-2048 is far out of reach.

### How does the quantum algorithm solve it?
**Reduction to order finding.** Pick $a$ with $1 < a < N$ and $\gcd(a, N) = 1$. The **order** $r$ is the smallest
positive integer with $a^r \equiv 1 \pmod N$. If $r$ is even and $a^{r/2} \not\equiv -1 \pmod N$, then

$$p = \gcd\!\left(a^{r/2} - 1,\; N\right),\qquad q = \gcd\!\left(a^{r/2} + 1,\; N\right)$$

are non-trivial factors. Finding $r$ is the hard part, and this is what the quantum computer does.

**Quantum phase estimation (QPE).** The unitary $U_a|y\rangle = |a\,y \bmod N\rangle$ (for $y < N$) permutes basis
states. Its eigenvalues are $e^{2\pi i\,s/r}$, $s = 0, \dots, r-1$, and the state $|1\rangle$ is an equal
superposition of the corresponding eigenvectors. The circuit uses two registers:

* **work register** - $n = \lceil \log_2 (N+1) \rceil$ qubits, initialised to $|1\rangle$;
* **counting register** - $t = 2n$ qubits in uniform superposition (enough precision that the continued-fraction
  step below recovers any $r \le N$, since $2^t \ge N^2$).

Counting qubit $j$ (weight $2^j$) controls $U_a^{2^j} = U_{a^{2^j} \bmod N}$, computed classically by repeated
squaring, so only $t$ controlled multiplications are needed. An **inverse QFT** on the counting register then
concentrates the probability at $y \approx 2^t\, s/r$.

**Classical post-processing.** Each measured $y$ gives a phase $y/2^t \approx s/r$; the **continued-fraction
expansion** (best rational approximation with denominator $\le N$) yields $r$ (or a divisor of it, so small
multiples are tested too). We check $a^r \equiv 1 \pmod N$ and compute the gcds above.

### Measurement
Only the counting register is measured. The outcomes cluster at the $r$ values $y \approx k\,2^t/r$; if $r$ divides
$2^t$ they are exact (e.g. $N = 15$, $a = 7$: $r = 4$, peaks at $y = 0, 64, 128, 192$ for $t = 8$).

### Qubit requirements
This notebook needs $t + n = 3n$ qubits ($N = 15$: $4 + 8 = 12$). The modular multiplications are implemented as exact
*permutation unitaries* (custom gates given to CUDA-Q as matrices). This is efficient for simulation and exact, but
it is **not** a gate-level fault-tolerant circuit: compiled circuits (e.g. Beauregard, $2n+3$ qubits) need many more gates.

### What is the expected solution?
For the default $N = 15$, $a = 7$: order $r = 4$, $7^2 = 49 \equiv 4$, so $\gcd(3, 15) = 3$ and $\gcd(5, 15) = 5$:
**$15 = 3 \times 5$**.

> **Important:** a classical simulator needs memory exponential in the number of qubits ($2^{3n}$ amplitudes here),
> so it can only "factor" tiny integers. The resource-estimation modes in Section 6 show what larger integers would
> need. Nothing in this notebook suggests that a simulator can factor large integers.
"""))

cells.append(md(INSTALL_MD))
cells.append(common_imports("from fractions import Fraction                    # continued fractions (exact rational arithmetic)"))

cells.append(md("""
---
## Section 3 - Input parameters

This is the **only cell you need to edit**. Every later cell reads these variables.
"""))
cells.append(code(r'''
# =====================================================================================
# Section 3 - INPUT PARAMETERS  (edit only this cell, then re-run the notebook)
# =====================================================================================

# ---- Problem definition -----------------------------------------------------------------
N_TO_FACTOR = 15                 # odd composite integer, not a prime power. Qubits needed = 3 x bits(N):
                                 # 15 -> 12 qubits, 21 -> 15, 33/35 -> 18, 55/57 -> 18, 91 -> 21, 143 -> 24
BASE_A = 7                       # 1 < a < N with gcd(a, N) = 1; None = random (Shor's original recipe, seeded)
NUM_COUNTING_QUBITS = None       # None = 2n (precision needed for continued fractions); or an integer
MAX_ATTEMPTS = 5                 # retries with a new random a if an attempt fails (odd order, trivial factor)
MAX_SIMULATION_QUBITS = 26       # never simulate circuits wider than this (estimation-only above)

# ---- Resource-estimation modes (no simulation needed) ---------------------------------------
ESTIMATE_BITS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]   # integer sizes (bits) to tabulate
QUBIT_BUDGET = 100               # reverse mode: largest integer that fits into this many (logical) qubits
MEMORY_BUDGET_GB = 80.0          # reverse mode: largest integer THIS notebook could simulate with this memory
''' + common_input_block(ALLOWED, "statevector", 1000, "exports/shor")))

cells.append(md(params_md(r"""
| `N_TO_FACTOR` | The integer $N$ to factor. | Qubits grow as $3\lceil\log_2(N+1)\rceil$, simulation memory as $16 \cdot 2^{3n}$ bytes: $N=15$: 12 qubits (65 KB), $N=143$: 24 qubits (268 MB), $N=1023$: 30 qubits (17 GB). Even numbers, primes and prime powers are detected and handled classically. |
| `BASE_A` | The base $a$ of the modular exponentiation. | Different $a$ have different orders $r$ (different peak patterns). Some $a$ fail (odd $r$ or $a^{r/2} \equiv -1$); then a new random $a$ is tried. If $\gcd(a,N) > 1$ the notebook switches to a random valid $a$. |
| `NUM_COUNTING_QUBITS` | Precision $t$ of the phase estimate. | Fewer than $2n$ qubits can make the continued-fraction step fail; more qubits sharpen the peaks but cost memory. |
| `MAX_ATTEMPTS` | Number of $a$ values tried. | Each attempt is one quantum run; the success probability per attempt is at least ~1/2 for most $N$. |
| `MAX_SIMULATION_QUBITS` | Safety limit for simulation. | Above it the notebook only shows resource estimates. |
| `ESTIMATE_BITS` | Integer sizes for the resource table. | Display only (no simulation). |
| `QUBIT_BUDGET` / `MEMORY_BUDGET_GB` | Inputs of the reverse estimation mode. | Shows the largest integer representable with a qubit budget, and the largest one this notebook could simulate with that memory. |
""")))

cells.append(md(r"""
---
## Section 4 - Example inputs

**Default example (runs as is):** `N_TO_FACTOR = 15`, `BASE_A = 7` - the classic first demonstration of Shor's algorithm.

| Example | Setting in Section 3 | Qubits | Order r | Factors |
|---|---|---|---|---|
| Classic | `N_TO_FACTOR = 15`, `BASE_A = 7` | 12 | 4 | 3 x 5 |
| 15 qubits, order not a power of 2 | `N_TO_FACTOR = 21`, `BASE_A = 2` | 15 | 6 | 3 x 7 |
| Larger | `N_TO_FACTOR = 35`, `BASE_A = 3` | 18 | 12 | 5 x 7 |
| Random base | `BASE_A = None` | - | - | Shor's original randomised recipe |
| Estimation only | `N_TO_FACTOR = 2**61 - 1` or any large number | - | - | qubit/memory estimates, no simulation |

The cell below validates $N$, handles the classically easy cases and computes the register sizes.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 4 - Validate N, choose a, compute the qubit requirements
# ---------------------------------------------------------------------------------
def is_probable_prime(m):
    """Miller-Rabin test (deterministic for m < 3.3e24, a strong probabilistic test beyond)."""
    if m < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    for p in small_primes:
        if m % p == 0:
            return m == p
    d, s = m - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in small_primes:
        x = pow(a, d, m)
        if x in (1, m - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, m)
            if x == m - 1:
                break
        else:
            return False
    return True

def integer_root(m, k):
    """floor(m ** (1/k)) for big integers (binary search, exact)."""
    lo, hi = 1, 1 << (m.bit_length() // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= m:
            lo = mid
        else:
            hi = mid - 1
    return lo

def prime_power(m):
    """(p, k) if m = p^k with p prime and k >= 2, else None (Shor's reduction does not apply)."""
    for k in range(2, m.bit_length() + 1):
        root = integer_root(m, k)
        if root > 1 and root ** k == m and is_probable_prime(root):
            return root, k
    return None

if not isinstance(N_TO_FACTOR, (int, np.integer)) or isinstance(N_TO_FACTOR, bool) or N_TO_FACTOR < 4:
    raise ValueError("N_TO_FACTOR must be an integer >= 4.")
N = int(N_TO_FACTOR)
classical_shortcut = None                                      # set if no quantum computation is needed
if N % 2 == 0:
    classical_shortcut = f"N is even: {N} = 2 x {N // 2} (no quantum computation needed)"
elif is_probable_prime(N):
    classical_shortcut = f"N = {N} is prime: it has no non-trivial factors"
elif prime_power(N) is not None:
    base, power = prime_power(N)
    classical_shortcut = f"N = {base}^{power} is a prime power: found classically (Shor's reduction needs >= 2 distinct primes)"

n_work = N.bit_length()                                        # bits needed to store 0 ... N-1
n_count = 2 * n_work if NUM_COUNTING_QUBITS is None else int(NUM_COUNTING_QUBITS)
total_qubits = n_work + n_count
simulate = classical_shortcut is None and total_qubits <= MAX_SIMULATION_QUBITS

rng = np.random.default_rng(SEED)
def random_base():
    """Random a with 1 < a < N and gcd(a, N) = 1 (Shor's original recipe)."""
    while True:
        a = int(rng.integers(2, N - 1))
        if math.gcd(a, N) == 1:
            return a

base_note = ""
if classical_shortcut is None:
    if BASE_A is None:
        a_first = random_base()
        base_note = "random (BASE_A = None)"
    elif not 1 < int(BASE_A) < N or math.gcd(int(BASE_A), N) != 1:
        lucky = math.gcd(int(BASE_A), N)
        a_first = random_base()
        base_note = (f"BASE_A = {BASE_A} is invalid for N = {N}"
                     + (f" (gcd = {lucky}: a lucky classical factor!)" if 1 < lucky < N else "")
                     + f" -> using random a = {a_first}")
    else:
        a_first = int(BASE_A)
        base_note = "from BASE_A"

print(f"N to factor          : {N}  ({n_work} bits)")
if classical_shortcut:
    print(f"Classical shortcut   : {classical_shortcut}")
else:
    print(f"Base a               : {a_first}  ({base_note})")
    print(f"Work register        : n = {n_work} qubits (holds y = 0 ... {2 ** n_work - 1} >= N - 1)")
    print(f"Counting register    : t = {n_count} qubits (2^t = {2 ** n_count:,} >= N^2 = {N * N:,}: "
          f"{'yes' if 2 ** n_count >= N * N else 'NO - continued fractions may fail'})")
    print(f"Total qubits         : t + n = {total_qubits}")
    print(f"Simulation           : {'yes' if simulate else f'NO - {total_qubits} qubits > MAX_SIMULATION_QUBITS = {MAX_SIMULATION_QUBITS} (estimation only)'}")
'''))

cells.append(md(UTILS_MD))
cells.append(common_utils())
cells.append(code(backend_init_code(ALLOWED) + r'''
if backend_cfg["device"] == "hardware":
    raise ValueError("This notebook implements U_a as exact custom unitaries (permutation matrices), which hardware "
                     "targets cannot execute. Use a simulator backend; Section 6 gives hardware resource estimates.")
'''))

cells.append(md(r"""
---
## Section 6 - Expected output (before execution)

**Expected graphs (Section 9):** a histogram of the measured counting-register values $y$ with $r$ peaks near
$k\,2^t/r$, the factors found, and the memory timeline.

**Expected numbers:** for this tiny $N$ we can compute the order $r$ classically (by brute force - impossible for
real cryptographic sizes), predict the peak positions and the factors, and check the quantum result against them.

**Memory and time:** exact state-vector size of the $3n$-qubit circuit, and a calibrated (rough) time estimate.

Sections 6.2 and 6.3 below are the **resource-estimation** and **reverse-estimation** modes.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6.1 - Expected result for this N, memory and time
# ---------------------------------------------------------------------------------
def classical_order(a, m):
    """Order of a modulo m by brute force (only feasible for tiny m - this is the hard part Shor speeds up)."""
    value, r = a % m, 1
    while value != 1:
        value, r = (value * a) % m, r + 1
    return r

def trial_division(m):
    for p in range(3, integer_root(m, 2) + 1, 2):
        if m % p == 0:
            return p, m // p
    return None

if classical_shortcut:
    print(classical_shortcut)
else:
    r_expected = classical_order(a_first, N)
    print(f"Order of a = {a_first} modulo {N} (classical check): r = {r_expected}")
    peaks = [round(k * 2 ** n_count / r_expected) for k in range(r_expected)]
    print(f"Expected peaks of y (t = {n_count}): near {peaks[:12]}{' ...' if r_expected > 12 else ''}")
    if r_expected % 2 == 0 and pow(a_first, r_expected // 2, N) != N - 1:
        x = pow(a_first, r_expected // 2, N)
        print(f"Expected factors from this a: gcd({x} - 1, {N}) = {math.gcd(x - 1, N)}, gcd({x} + 1, {N}) = {math.gcd(x + 1, N)}")
    else:
        print("This a is an unlucky base (odd order or a^(r/2) = -1 mod N): the notebook will retry with another a.")
    if N < 10 ** 12:
        print(f"Factors by trial division (reference): {trial_division(N)}")
    if simulate:
        state_bytes = estimate_memory_bytes(total_qubits, backend_cfg["method"], backend_cfg["precision"])
        print("\nExpected memory")
        check_memory(state_bytes, backend_cfg, MEMORY_SAFETY_FRACTION)
        custom_ops = sum(1 for j in range(n_count) if pow(a_first, 2 ** j, N) != 1)
        gates = n_count + 1 + n_count * (n_count + 1) // 2 + n_count // 2 + custom_ops * 2 ** n_work
        seconds_per_gate, timing_note = measure_gate_time(backend_cfg, total_qubits)
        print("\nExpected execution time (ESTIMATE)")
        print(f"  Gates: {n_count} H + 1 X + {n_count * (n_count - 1) // 2} controlled phases + {n_count // 2} SWAPs "
              f"+ {custom_ops} controlled modular multiplications (each ~2^n = {2 ** n_work} gate-equivalents)")
        print(f"  Calibration: {timing_note}")
        if seconds_per_gate is not None:
            print(f"  Estimated time per attempt: ~{format_duration(gates * seconds_per_gate)}")
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6.2 - RESOURCE-ESTIMATION MODE: what would larger integers need?
# ---------------------------------------------------------------------------------
# Logical-qubit formulas (error correction would multiply these by ~100-1000 physical qubits each):
#   this notebook     : 3n      (2n counting + n work qubits, modular multiplication as exact unitaries)
#   Beauregard (2003) : 2n + 3  (gate-level circuit, one recycled control qubit; arXiv:quant-ph/0205095)
#   Gidney-Ekera 2019 : 3n + 0.002 n log2 n  (optimised fault-tolerant construction; arXiv:1905.09749)
def notebook_qubits(bits):
    return 3 * bits

def beauregard_qubits(bits):
    return 2 * bits + 3

def gidney_ekera_qubits(bits):
    return math.ceil(3 * bits + 0.002 * bits * math.log2(bits))

print(f"{'bits n':>7} {'largest N':>12} {'this notebook':>14} {'state-vector memory':>20} {'Beauregard 2n+3':>16} {'Gidney-Ekera':>13}")
for bits in ESTIMATE_BITS:
    largest = f"2^{bits}-1" if bits > 30 else f"{2 ** bits - 1:,}"
    memory_needed = format_bytes(16 * 2 ** notebook_qubits(bits))
    print(f"{bits:>7} {largest:>12} {notebook_qubits(bits):>14} {memory_needed:>20} {beauregard_qubits(bits):>16} {gidney_ekera_qubits(bits):>13}")
print("\nReading the table: RSA-2048 (n = 2048) needs about 6,200 LOGICAL qubits in the best known constructions;")
print("with surface-code error correction Gidney & Ekera estimated ~20 million noisy physical qubits (2019),")
print("and later work (Gidney 2025) reduced this to below one million. Simulating even n = 32 would need")
print(f"{format_bytes(16 * 2 ** 96)} of memory - classical simulation of Shor's algorithm is limited to tiny N.")
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6.3 - REVERSE ESTIMATION MODE: given a budget, what is the largest integer?
# ---------------------------------------------------------------------------------
q = int(QUBIT_BUDGET)
bits_notebook = q // 3
bits_beauregard = (q - 3) // 2
bits_ge = max([b for b in range(2, q + 1) if gidney_ekera_qubits(b) <= q], default=0)
print(f"Qubit budget: {q} logical qubits")
for label, bits in (("this notebook's construction (3n)", bits_notebook), ("Beauregard (2n + 3)", bits_beauregard),
                    ("Gidney-Ekera (3n + 0.002 n log2 n)", bits_ge)):
    if bits >= 2:
        print(f"  {label:<36}: integers up to {bits} bits, i.e. N <= {2 ** bits - 1:,}")
    else:
        print(f"  {label:<36}: budget too small")

memory_bytes = float(MEMORY_BUDGET_GB) * 1e9
sim_qubits = int(math.floor(math.log2(memory_bytes / 16)))         # fp64 state vector: 16 bytes per amplitude
sim_bits = sim_qubits // 3
print(f"\nMemory budget: {MEMORY_BUDGET_GB} GB -> a {sim_qubits}-qubit fp64 state vector fits")
print(f"  This notebook can therefore SIMULATE Shor for N up to {sim_bits} bits, i.e. N <= {2 ** sim_bits - 1:,}")
print("  (one more qubit doubles the memory; fp32 on GPUs gains one qubit).")
'''))

cells.append(md(r"""
---
## Section 7 - Quantum circuit generation

The circuit is assembled with CUDA-Q's **kernel builder** (`cudaq.make_kernel`), because each controlled power
$U_a^{2^j} = U_{a^{2^j} \bmod N}$ is a *different* custom gate:

1. `mulmod_matrix(c)` - the permutation matrix of $|y\rangle \to |c\,y \bmod N\rangle$ (identity for $y \ge N$), registered
   with `cudaq.register_operation`. CUDA-Q expects custom-gate matrices with the **first target qubit as the most
   significant bit**, which matches our convention (work[0] = most significant bit of $y$).
2. `build_shor_kernel(a)` - $X$ on the last work qubit ($|1\rangle$), $H$ on all counting qubits, controlled $U_a^{2^j}$ with
   counting qubit $t-1-j$ as control (qubit 0 is the most significant bit of $y$), inverse QFT, measurement of the
   counting register.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7a - Circuit construction with the CUDA-Q kernel builder
# ---------------------------------------------------------------------------------
registered_ops = set()                                           # names already given to cudaq.register_operation

def mulmod_matrix(c, modulus, n):
    """Permutation unitary |y> -> |c*y mod modulus> for y < modulus, identity for y >= modulus."""
    dim = 2 ** n
    matrix = np.zeros((dim, dim))
    for y in range(dim):
        matrix[(c * y) % modulus if y < modulus else y, y] = 1.0
    return matrix

def build_shor_kernel(a, modulus=None, count_qubits=None, work_qubits=None):
    """Order-finding circuit for a modulo N; returns (kernel, list of custom gates used)."""
    modulus = N if modulus is None else modulus
    t = n_count if count_qubits is None else count_qubits
    n = n_work if work_qubits is None else work_qubits
    kernel = cudaq.make_kernel()
    count = kernel.qalloc(t)                                     # counting register (phase estimate)
    work = kernel.qalloc(n)                                      # work register (holds a^x * 1 mod N)
    kernel.x(work[n - 1])                                        # |1>: the last qubit is the least significant bit
    kernel.h(count)                                              # uniform superposition over x = 0 ... 2^t - 1
    used = []
    for j in range(t):
        c = pow(a, 2 ** j, modulus)                              # a^(2^j) mod N by classical repeated squaring
        if c == 1:
            continue                                             # U_1 = identity: nothing to apply
        name = f"mulmod_{c}_mod_{modulus}"
        if name not in registered_ops:
            cudaq.register_operation(name, mulmod_matrix(c, modulus, n))
            registered_ops.add(name)
        getattr(kernel, name)(count[t - 1 - j], *[work[i] for i in range(n)])   # first argument = control
        used.append(name)
    for j in range(t // 2):                                      # ---- inverse QFT on the counting register ----
        kernel.swap(count[j], count[t - 1 - j])
    for jj in range(t):
        j = t - 1 - jj
        for kk in range(t - 1 - j):
            k = t - 1 - kk
            kernel.cr1(-np.pi / 2 ** (k - j), count[k], count[j])  # undo the phase kick-back pattern
        kernel.h(count[j])
    kernel.mz(count)                                             # measure the counting register only
    return kernel, used

print("Circuit builder defined.")
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7b - Circuit visualization (at most 10 qubits) and full-circuit statistics
# ---------------------------------------------------------------------------------
if not simulate:
    print("No circuit to draw: " + (classical_shortcut or f"{total_qubits} qubits exceed MAX_SIMULATION_QUBITS."))
else:
    max_draw = min(int(DRAW_QUBITS), 10)
    if n_work >= max_draw:
        print(f"Drawing skipped: the work register alone has {n_work} qubits (> {max_draw - 1}); see the statistics below.")
    else:
        t_draw = min(n_count, max_draw - n_work)                 # fewer counting qubits so that the drawing has <= 10 qubits
        draw_kernel, _ = build_shor_kernel(a_first, count_qubits=t_draw)
        print(f"Representative drawing with {t_draw} counting + {n_work} work qubits "
              f"(the executed circuit has {n_count} counting qubits).")
        full_kernel_vis, _ = build_shor_kernel(a_first)
        show_circuit(draw_kernel, (), t_draw + n_work, total_qubits, title="Order-finding circuit.",
                     export=EXPORT_CIRCUIT, formats=EXPORT_FORMATS, out_dir=EXPORT_DIR, basename="shor",
                     full_kernel=full_kernel_vis, full_args=())
    full_kernel_vis, used_ops = build_shor_kernel(a_first)
    stats = circuit_statistics(full_kernel_vis)
    print(f"\nFULL circuit statistics ({total_qubits} qubits, a = {a_first}):")
    for key, value in stats.items():
        print(f"  {key:<32}: {value}")
    print(f"  controlled modular multiplications : {used_ops}  ('ccustom_unitary_constant' above)")
'''))

cells.append(md(r"""
---
## Section 8 - Execution with live memory monitoring

Running the next cell:
1. initialises the selected backend,
2. builds the order-finding circuit for the current $a$ (construction time = builder + matrix registration),
3. samples it `SHOTS` times and post-processes every distinct outcome with continued fractions,
4. retries with a new random $a$ (up to `MAX_ATTEMPTS`) if no factors were found,
5. monitors host RAM and GPU memory during the whole cell and prints a short summary.
""" + EXECUTION_MD))
cells.append(code(r'''
# =====================================================================================
# >>> EXECUTION CELL - Shor's algorithm with live memory monitoring <<<
# =====================================================================================
execution_ok, execution_error = False, None
attempts, factors, phases_first, construction_total, execution_later = [], None, None, 0.0, 0.0

def candidate_orders(y, t, modulus, a):
    """Orders suggested by outcome y: continued-fraction denominator and its small multiples."""
    if y == 0:
        return None, []
    fraction = Fraction(y, 2 ** t).limit_denominator(modulus)     # best s/r with r <= N
    base = fraction.denominator
    return fraction, [base * k for k in range(1, modulus // base + 1) if pow(a, base * k, modulus) == 1][:1]

if not simulate:
    print("Quantum execution skipped: " + (classical_shortcut or f"{total_qubits} qubits > MAX_SIMULATION_QUBITS "
                                                                  f"= {MAX_SIMULATION_QUBITS}. See the resource estimates in Section 6."))
    memory = None
else:
    monitor = MemoryMonitor(MEMORY_SAMPLE_INTERVAL_S, live=LIVE_MEMORY_DISPLAY)
    monitor.start()
    cell_start = time.perf_counter()
    try:
        # Step 1 - initialise the backend
        initialize_backend(backend_cfg, seed=SEED)
        a = a_first
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Step 2 - build the circuit for this a (registering the permutation matrices)
            t0 = time.perf_counter()
            kernel, used_ops = build_shor_kernel(a)
            construction_total += time.perf_counter() - t0
            # Step 3 - execute (first launch traced) and post-process every distinct outcome
            if phases_first is None:
                counts, phases_first = profile_first_launch(lambda: cudaq.sample(kernel, shots_count=SHOTS))
            else:
                t0 = time.perf_counter()
                counts = cudaq.sample(kernel, shots_count=SHOTS)
                execution_later += time.perf_counter() - t0
            rows, found = [], None
            for bits, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                y = int(bits, 2)                                     # counting register, qubit 0 = most significant bit
                fraction, orders = candidate_orders(y, n_count, N, a)
                row = {"y": y, "bits": bits, "counts": c, "phase": y / 2 ** n_count, "fraction": fraction,
                       "r": orders[0] if orders else None, "factors": None}
                if orders and orders[0] % 2 == 0 and pow(a, orders[0] // 2, N) != N - 1:
                    x = pow(a, orders[0] // 2, N)
                    p, q = math.gcd(x - 1, N), math.gcd(x + 1, N)
                    if 1 < p < N and p * q == N:
                        row["factors"] = (min(p, q), max(p, q))
                        found = found or row["factors"]
                rows.append(row)
            attempts.append({"a": a, "counts": counts, "rows": rows, "factors": found, "custom_ops": used_ops})
            print(f"Attempt {attempt}: a = {a}, {len(counts)} distinct outcomes -> "
                  + (f"factors {found[0]} x {found[1]}" if found else "no factors (unlucky a), retrying"))
            if found:
                factors = found
                break
            a = random_base()                                       # Shor: pick another random base
        execution_ok = True
    except Exception as exc:
        execution_error = exc
        print(f"\nEXECUTION FAILED: {type(exc).__name__}: {exc}")
        print("Hints: use a smaller N_TO_FACTOR or another BACKEND / SIMULATION_METHOD (README).")
    finally:
        memory = monitor.stop()
        total_wall = time.perf_counter() - cell_start
    if execution_ok:
        phases = dict(phases_first)
        phases["construction_s"] = construction_total + (phases_first["construction_s"] or 0.0)
        phases["execution_s"] = (phases_first["execution_s"] or 0.0) + execution_later
        phases["total_wall_s"] = total_wall
        print(f"\nTiming: construction {fmt_seconds(phases['construction_s'])}, compilation {fmt_seconds(phases['compilation_s'])}, "
              f"execution {fmt_seconds(phases['execution_s'])} over {len(attempts)} attempt(s); total {total_wall:.2f} s\n")
    print_memory_summary(memory)
'''))

cells.append(md(r"""
---
## Section 9 - Results and visualization

For every measured $y$: the phase $y/2^t$, its continued-fraction approximation $s/r$, the order candidate $r$
(verified by $a^r \equiv 1 \bmod N$) and the factors it yields. Outcomes like $y = 0$ or $s/r$ with a common
divisor give no (or a trivial) answer - this is expected, which is why Shor's algorithm may need several shots.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 9 - Outcome table, factor verification and plots
# ---------------------------------------------------------------------------------
if not simulate or not execution_ok:
    print("No quantum results: " + (classical_shortcut or "simulation skipped or failed (see above)."))
else:
    last = attempts[-1]
    print(f"Outcomes of the last attempt (a = {last['a']}, t = {n_count}, {SHOTS} shots):")
    print(f"{'y':>8} {'bitstring':<{max(9, n_count)}} {'counts':>6} {'phase y/2^t':>12} {'s/r':>8} {'order r':>8}  factors")
    for row in last["rows"][:16]:
        print(f"{row['y']:>8} {row['bits']:<{max(9, n_count)}} {row['counts']:>6} {row['phase']:>12.5f} "
              f"{str(row['fraction']) if row['fraction'] is not None else '-':>8} {str(row['r']) if row['r'] else '-':>8}  "
              f"{row['factors'] if row['factors'] else '-'}")
    useful = sum(r["counts"] for r in last["rows"] if r["factors"]) / SHOTS
    print(f"\nFraction of shots that directly yield the factors: {useful:.3f}")
    if factors:
        p, q = factors
        print(f"\nRESULT: {N} = {p} x {q}   verification: {p} x {q} = {p * q} -> {'CORRECT' if p * q == N else 'WRONG'}")
    else:
        print(f"\nNo factors found in {len(attempts)} attempt(s); increase MAX_ATTEMPTS or SHOTS.")

    fig = plt.figure(figsize=(12, 6.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.8], hspace=0.45)
    ax = fig.add_subplot(grid[0])
    ys = np.array([r["y"] for r in last["rows"]])
    cs = np.array([r["counts"] for r in last["rows"]])
    colors = [BLUE if r["factors"] else INK_MUTED for r in last["rows"]]
    width = max(1.0, 2 ** n_count / 300)
    ax.bar(ys, cs / SHOTS, width=width, color=colors)
    r_last = classical_order(last["a"], N)
    for k in range(r_last):
        ax.axvline(k * 2 ** n_count / r_last, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_xlim(-2 * width, 2 ** n_count + 2 * width)
    ax.set_xlabel(f"measured counting-register value y (0 ... {2 ** n_count - 1}); thin lines: k * 2^t / r")
    ax.set_ylabel("probability")
    ax.set_title(f"Order finding for a = {last['a']} mod {N}: peaks at multiples of 2^t / r (blue = outcome yields the factors)",
                 loc="left")
    ax_m = fig.add_subplot(grid[1])
    monitor.plot(ax=ax_m, title="Memory during the execution cell")
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
if not simulate or not execution_ok:
    print("No benchmark report: " + (classical_shortcut or "the quantum simulation was skipped or failed."))
else:
    report = build_report("Shor's algorithm (integer factorization)", backend_cfg, total_qubits, SHOTS, phases, memory)
    report["Integer N / bits"] = f"{N} / {n_work}"
    report["Registers"] = f"{n_count} counting + {n_work} work qubits = {total_qubits}"
    report["Bases tried (a)"] = ", ".join(str(at["a"]) for at in attempts)
    report["Custom modular-multiplication gates"] = ", ".join(attempts[-1]["custom_ops"]) or "none"
    reference = trial_division(N) if N < 10 ** 12 else None
    report["Expected result"] = f"{N} = {reference[0]} x {reference[1]} (trial division)" if reference else "n/a"
    report["Actual result"] = f"{N} = {factors[0]} x {factors[1]}" if factors else "no factors found"
    report["Verification p x q == N"] = ("CORRECT" if factors and factors[0] * factors[1] == N else "FAILED")
    report["Order r found / classical check"] = f"{next((r['r'] for r in attempts[-1]['rows'] if r['factors']), '-')} / {classical_order(attempts[-1]['a'], N)}"
    report["Shots yielding the factors"] = f"{useful:.3f}"
    final_fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(ys, cs / SHOTS, width=width, color=colors)
    ax.set_xlabel("counting-register value y")
    ax.set_ylabel("probability")
    ax.set_title(f"Shor: {N} = {factors[0]} x {factors[1]} (a = {attempts[-1]['a']}, {total_qubits} qubits, {cudaq.get_target().name})"
                 if factors else f"Shor N={N}", loc="left")
    final_fig.tight_layout()
    show_report(report, save=SAVE_REPORT, out_dir=EXPORT_DIR, name="shor_benchmark_report", figure=final_fig)
    plt.show()
'''))

cells.append(md(r"""
---
### Things to try next
* `N_TO_FACTOR = 21`, `BASE_A = 2` (15 qubits): the order $r = 6$ does not divide $2^t$, so the peaks are spread and
  the continued-fraction step does real work.
* `BASE_A = 14` with `N_TO_FACTOR = 15`: $14 \equiv -1$, so $a^{r/2} \equiv -1$ - an unlucky base; the notebook retries.
* `N_TO_FACTOR = 143` (24 qubits) on a GPU, then compare the reports. Each extra bit of $N$ adds 3 qubits = 8x memory.
* Set `N_TO_FACTOR` to an RSA-sized number to see only the resource estimates (no simulation).
"""))

write_notebook(OUT, cells, "Shor's Algorithm - CUDA-Q benchmark")
print("wrote", OUT)
