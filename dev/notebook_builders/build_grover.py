from nbgen import md, code, common_imports, common_utils, write_notebook
from common_md import (title_md, INSTALL_MD, params_md, common_input_block, UTILS_MD,
                       backend_init_code, EXECUTION_MD)
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "Grover.ipynb"
ALLOWED = ["statevector", "mps", "tensornet", "density_matrix"]

cells = []
cells.append(md(title_md(
    4, "Grover's Algorithm - Quantum Search", "Finding one marked item among 2^N with NVIDIA CUDA-Q",
    "This notebook builds Grover's search algorithm in CUDA-Q, runs it on the backend you choose (CPU, "
    "NVIDIA GPU, multi-GPU or a quantum-hardware target), shows how the probability of finding the marked "
    "item grows and oscillates with the number of Grover iterations, and reports real timing and memory "
    "measurements.")))

cells.append(md(r"""
---
## Section 1 - Problem statement and mathematical explanation

### What problem does the algorithm solve?
**Unstructured search.** We are given $M = 2^N$ items labelled by $N$-bit strings $x$ and a black box
(the *oracle*) that recognises the one item we want, $x^\ast$:

$$f(x) = \begin{cases} 1 & \text{if } x = x^\ast \\ 0 & \text{otherwise.} \end{cases}$$

The task: find $x^\ast$ with as few oracle calls as possible.

### Why is this problem important?
* Grover's algorithm gives a provable **quadratic speed-up** for every problem whose solutions are easy to
  *check* but hard to *find* (satisfiability, constraint problems, database-style search).
* Its core idea, **amplitude amplification**, is a building block of many other quantum algorithms
  (quantum counting, minimum finding, collision finding).
* It sets key-length margins in cryptography: a $k$-bit key needs about $2^{k/2}$ Grover iterations
  instead of $2^{k}$ classical guesses.

### What is the classical approach?
Without structure the only option is to test the items one by one: on average $M/2$ oracle calls,
$M$ in the worst case - the cost is $O(M)$.

### How does the quantum algorithm solve it?
1. **Uniform superposition.** Hadamard gates on all qubits create
   $|s\rangle = H^{\otimes N}|0\rangle^{\otimes N} = \frac{1}{\sqrt{M}}\sum_{x=0}^{M-1}|x\rangle$.
2. **Oracle** $O = I - 2|x^\ast\rangle\langle x^\ast|$ flips the sign of the marked amplitude,
   $O|x\rangle = (-1)^{f(x)}|x\rangle$. Circuit: $X$ on every qubit where $x^\ast$ has a 0, a
   multi-controlled $Z$, then the same $X$ gates again.
3. **Diffusion** $D = 2|s\rangle\langle s| - I$ ("inversion about the mean") turns the sign flip into a larger
   amplitude. Circuit: $H^{\otimes N}X^{\otimes N}\,(\text{multi-controlled }Z)\,X^{\otimes N}H^{\otimes N}$,
   which equals $D$ up to an irrelevant global phase.
4. Repeat the **Grover iteration** $G = D\,O$ exactly $k$ times and measure all qubits.

**Geometric picture.** Define $\sin\theta = 1/\sqrt{M}$. The state always lies in the plane spanned by
$|x^\ast\rangle$ and the uniform superposition of the unmarked items, and every iteration rotates it by
$2\theta$ towards $|x^\ast\rangle$:

$$G^k|s\rangle = \sin\big((2k+1)\theta\big)\,|x^\ast\rangle + \cos\big((2k+1)\theta\big)\,|\text{rest}\rangle,
\qquad P_k(x^\ast) = \sin^2\big((2k+1)\theta\big).$$

The success probability is largest after

$$k_{\text{opt}} = \operatorname{round}\!\left(\frac{\pi}{4\theta} - \frac{1}{2}\right) \approx \frac{\pi}{4}\sqrt{M}$$

iterations: $O(\sqrt{M})$ oracle calls instead of $O(M)$. **More** iterations rotate *past* the target and
the probability falls again (over-rotation); Section 9 shows this oscillation.

### Measurement
Measuring every qubit in the computational ($Z$) basis returns the bitstring $x$ with probability
$|\langle x|\psi\rangle|^2$ (Born rule). Bitstrings are printed with **qubit 0 first** (leftmost character).

### What is the expected solution?
For the default $N = 15$: $M = 32\,768$, $\theta \approx 0.005524$, $k_{\text{opt}} = 142$ and
$P(x^\ast) \approx 0.99999$ - essentially every shot returns the marked bitstring, whereas a classical search
needs 16 384 oracle calls on average.
"""))

cells.append(md(INSTALL_MD))
cells.append(common_imports("pass"))

cells.append(md("""
---
## Section 3 - Input parameters

This is the **only cell you need to edit**. Every later cell reads these variables.
"""))
cells.append(code(r'''
# =====================================================================================
# Section 3 - INPUT PARAMETERS  (edit only this cell, then re-run the notebook)
# =====================================================================================

# ---- Problem definition ---------------------------------------------------------------
N_QUBITS = 15                    # search space has 2**N_QUBITS items (15 -> 32,768); try 20, 25, 30 on GPUs
MARKED_BITSTRING = "example"     # "example" = reproducible example of Section 4, or your own item as an
                                 # N-character string of 0/1 (qubit 0 first), or an integer 0 <= x < 2**N

# ---- Algorithm parameters -------------------------------------------------------------
GROVER_ITERATIONS = "optimal"    # "optimal" = round(pi/(4*theta) - 1/2), or any integer k >= 0
RUN_ITERATION_SWEEP = True       # also measure P(marked) after 0, 1, 2, ... iterations (Section 9)
SWEEP_MAX_ITERATIONS = "auto"    # "auto" = 2*k_opt + 1 (one full rise and fall), or an integer
SWEEP_MAX_POINTS = 60            # at most this many points on the sweep curve
''' + common_input_block(ALLOWED, "statevector", 2000, "exports/grover")))

cells.append(md(params_md(r"""
| `N_QUBITS` | Number of qubits $N$; the search space has $M = 2^N$ items. | Memory doubles with every qubit (state vector: 16 x 2^N bytes in fp64). The number of iterations grows like $\sqrt{M} = 2^{N/2}$, so state-vector run time grows roughly like $2^{1.5N}$: N=15 takes seconds, N=25 minutes on a GPU, N >= 30 needs large or multiple GPUs. |
| `MARKED_BITSTRING` | The item to find (qubit 0 = leftmost character). | Grover's performance does **not** depend on which item is marked; any valid item gives the same success probability. A wrong length or characters other than 0/1 stop the notebook with an error. |
| `GROVER_ITERATIONS` | Number $k$ of oracle + diffusion repetitions. | `"optimal"` maximises $P(x^\ast)$. Fewer iterations lower the probability; more iterations over-rotate and the probability falls again (it is periodic in $k$). |
| `RUN_ITERATION_SWEEP` | Measure $P(x^\ast)$ as a function of $k$. | Costs about `SWEEP_MAX_ITERATIONS` extra iterations; switch off for very large N. |
| `SWEEP_MAX_ITERATIONS` / `SWEEP_MAX_POINTS` | Range and resolution of the sweep. | More points give a smoother curve. With the state-vector method the cost depends only on the range, not on the number of points. |
""")))

cells.append(md(r"""
---
## Section 4 - Example inputs

**Default example (runs as is):** `N_QUBITS = 15`, `MARKED_BITSTRING = "example"`. The example item is a
pseudo-random 15-bit string generated from `SEED`, so it is reproducible but not "special" in any way. The
cell below prints it together with the derived quantities.

**How to change the example** - edit Section 3, for instance:

| Goal | Setting in Section 3 | What you will see |
|---|---|---|
| Your own item | `MARKED_BITSTRING = "110010101011100"` (exactly `N_QUBITS` characters) | that bitstring found with P ~ 1 |
| Integer input | `MARKED_BITSTRING = 12345` | binary of 12345 with N bits (qubit 0 = most significant bit) |
| Tiny instance | `N_QUBITS = 4`, `MARKED_BITSTRING = "1011"` | $k_{\text{opt}} = 3$, $P = 0.961$ - the whole circuit fits in the drawing |
| Over-rotation | `GROVER_ITERATIONS = 284` (= $2k_{\text{opt}}$ for N=15) | the marked item almost never appears |
| GPU benchmark | `N_QUBITS = 25`, `BACKEND = "gpu"` | $k_{\text{opt}} = 4\,549$ iterations on 33.5 million amplitudes |
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 4 - Build and validate the search problem from the Section 3 inputs
# ---------------------------------------------------------------------------------
if not isinstance(N_QUBITS, (int, np.integer)) or N_QUBITS < 2:
    raise ValueError("N_QUBITS must be an integer >= 2 (the multi-controlled Z needs at least 2 qubits).")
N_QUBITS = int(N_QUBITS)

if isinstance(MARKED_BITSTRING, str) and MARKED_BITSTRING.strip().lower() == "example":
    example_rng = np.random.default_rng(SEED)                       # reproducible example item
    marked_bitstring = "".join(str(b) for b in example_rng.integers(0, 2, N_QUBITS))
elif isinstance(MARKED_BITSTRING, (int, np.integer)) and not isinstance(MARKED_BITSTRING, bool):
    if not 0 <= int(MARKED_BITSTRING) < 2 ** N_QUBITS:
        raise ValueError(f"Integer MARKED_BITSTRING must satisfy 0 <= x < 2**{N_QUBITS} = {2 ** N_QUBITS}.")
    marked_bitstring = format(int(MARKED_BITSTRING), f"0{N_QUBITS}b")   # qubit 0 = most significant bit
else:
    marked_bitstring = str(MARKED_BITSTRING).strip()
    if len(marked_bitstring) != N_QUBITS or set(marked_bitstring) - {"0", "1"}:
        raise ValueError(f"MARKED_BITSTRING must contain exactly N_QUBITS={N_QUBITS} characters '0'/'1', "
                         f"got {marked_bitstring!r} (length {len(marked_bitstring)}).")
marked_bits = [int(c) for c in marked_bitstring]                    # list form passed to the CUDA-Q kernel

search_space = 2 ** N_QUBITS                                         # M = 2^N items
theta = math.asin(1.0 / math.sqrt(search_space))                     # rotation half-angle: sin(theta) = 1/sqrt(M)
k_optimal = max(0, round(math.pi / (4 * theta) - 0.5))               # iterations that maximise P(marked)
if isinstance(GROVER_ITERATIONS, str) and GROVER_ITERATIONS.strip().lower() == "optimal":
    iterations = k_optimal
elif isinstance(GROVER_ITERATIONS, (int, np.integer)) and int(GROVER_ITERATIONS) >= 0:
    iterations = int(GROVER_ITERATIONS)
else:
    raise ValueError("GROVER_ITERATIONS must be 'optimal' or a non-negative integer.")
if isinstance(SWEEP_MAX_ITERATIONS, str) and SWEEP_MAX_ITERATIONS.strip().lower() == "auto":
    sweep_max = 2 * k_optimal + 1                                    # one full rise and fall of P(marked)
else:
    sweep_max = int(SWEEP_MAX_ITERATIONS)
if not 0 < int(DRAW_QUBITS) <= 10:
    raise ValueError("DRAW_QUBITS must be between 1 and 10 (circuit drawings are limited to 10 qubits).")

print(f"Number of qubits N          : {N_QUBITS}")
print(f"Search space size M = 2^N   : {search_space:,}")
print(f"Marked bitstring x*         : {marked_bitstring}   (integer value {int(marked_bitstring, 2):,}, qubit 0 first)")
print(f"Rotation angle theta        : {theta:.6f} rad")
print(f"Optimal iterations k_opt    : {k_optimal:,}")
print(f"Iterations used             : {iterations:,}")
print(f"Classical expected queries  : {search_space / 2:,.0f} (average), {search_space:,} (worst case)")
'''))

cells.append(md(UTILS_MD))
cells.append(common_utils())
cells.append(code(backend_init_code(ALLOWED)))

cells.append(md(r"""
---
## Section 6 - Expected output (before execution)

**Expected graphs (Section 9)**
1. *Measured bitstrings* - one dominant bar at the marked bitstring holding almost all `SHOTS` counts.
2. *P(marked) versus the number of iterations* - a $\sin^2((2k+1)\theta)$ curve: starts at $1/M$, peaks near
   $k_{\text{opt}}$ and returns towards zero at about $2k_{\text{opt}}$. The simulated points must lie on the theory curve.
3. *Memory timeline* of the execution.

**Expected numbers** follow exactly from the formulas of Section 1 and are computed below: the success
probability, the expected number of marked counts and their binomial spread $\sqrt{S\,P(1-P)}$ for $S$ shots.

**Execution time and memory are not guessed.** The memory of the quantum state follows from an exact formula;
the execution time is *estimated* from a short calibration on your backend (time per gate) multiplied by the
number of gates. Treat the time as a rough estimate (factor ~3), because gate types, caching and Python
overhead differ from the calibration circuit.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6 - Expected results, memory requirement and execution-time estimate
# ---------------------------------------------------------------------------------
p_theory = math.sin((2 * iterations + 1) * theta) ** 2               # P(marked) after `iterations` iterations
expected_counts = SHOTS * p_theory                                   # expected number of shots on x*
count_spread = math.sqrt(SHOTS * p_theory * (1 - p_theory))          # binomial standard deviation
print("Expected result")
print(f"  P(marked) after {iterations} iterations : {p_theory:.8f}")
print(f"  Expected marked counts            : {expected_counts:.1f} +- {count_spread:.1f} of {SHOTS} shots")
print(f"  Expected most frequent bitstring  : {marked_bitstring}" if p_theory > 0.5 else
      "  P(marked) < 0.5: the marked item is NOT expected to dominate (check GROVER_ITERATIONS)")

# ---- Memory: the state vector (or MPS/density matrix) of N qubits ----------------------------
uses_state_sweep = RUN_ITERATION_SWEEP and backend_cfg["method"] == "statevector" and backend_cfg["device"] != "hardware"
state_bytes = estimate_memory_bytes(N_QUBITS, backend_cfg["method"], backend_cfg["precision"])
print("\nExpected memory")
check_memory(state_bytes, backend_cfg, MEMORY_SAFETY_FRACTION,
             extra_copies=1 if uses_state_sweep else 0)              # the sweep holds two states at once

# ---- Execution-time estimate: gates x measured time per gate --------------------------------
zeros = marked_bitstring.count("0")
gates_per_iteration = 4 * N_QUBITS + 2 * zeros + 2                    # H and X layers, oracle X gates, 2 multi-controlled Z
main_gates = N_QUBITS + iterations * gates_per_iteration              # initial H layer + iterations
sweep_gates = (sweep_max * gates_per_iteration) if RUN_ITERATION_SWEEP else 0
if RUN_ITERATION_SWEEP and not uses_state_sweep:                      # sampled sweep re-runs circuits from scratch
    sweep_gates = int(np.sum(np.linspace(0, sweep_max, num=min(SWEEP_MAX_POINTS, 12)))) * gates_per_iteration
seconds_per_gate, timing_note = measure_gate_time(backend_cfg, N_QUBITS)
print("\nExpected execution time (ESTIMATE)")
print(f"  Gates in the main circuit : {main_gates:,}   (per iteration: {gates_per_iteration})")
print(f"  Gates in the sweep        : {sweep_gates:,}")
print(f"  Calibration               : {timing_note}")
if seconds_per_gate is not None:
    print(f"  Main Grover run           : ~{format_duration(main_gates * seconds_per_gate)}")
    print(f"  Iteration sweep           : ~{format_duration(sweep_gates * seconds_per_gate)}")
'''))

cells.append(md(r"""
---
## Section 7 - Quantum circuit generation

The CUDA-Q kernels below implement Section 1 line by line. They are created inside a function
(`build_grover_kernels`) so that the execution cell can build **fresh** kernels and measure the circuit
construction and compilation time honestly (CUDA-Q caches compiled kernels).

* `grover_iteration` - oracle $O$ followed by diffusion $D$ on a register of qubits.
* `grover_search` - $H^{\otimes N}$, then $k$ iterations, then measurement of all qubits.
* `uniform_superposition` / `continue_iterations` - used by the iteration sweep: the state after $k$
  iterations is handed to the next kernel (`cudaq.qvector(state)`), so the whole sweep costs only
  `SWEEP_MAX_ITERATIONS` iterations instead of re-running every circuit from the start.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7a - CUDA-Q kernels for Grover's algorithm
# ---------------------------------------------------------------------------------
def build_grover_kernels():
    """Create and return fresh CUDA-Q kernels (called again in the execution cell)."""

    @cudaq.kernel
    def grover_iteration(qubits: cudaq.qview, marked: list[int]):
        """One Grover iteration G = D * O on `qubits`; `marked` lists the bits of x* (qubit 0 first)."""
        n = qubits.size()                            # number of qubits in the register
        # ---- Oracle O: multiply the amplitude of |x*> by -1 -------------------------------------
        for i in range(n):
            if marked[i] == 0:
                x(qubits[i])                         # X where x* has a 0: maps |x*> onto |11...1>
        z.ctrl(qubits[0:n - 1], qubits[n - 1])       # Z on the last qubit if all others are 1: flips the sign of |11...1>
        for i in range(n):
            if marked[i] == 0:
                x(qubits[i])                         # undo the X gates: the sign flip now sits on |x*>
        # ---- Diffusion D = 2|s><s| - I (inversion about the mean) ---------------------------------
        h(qubits)                                    # |s> -> |00...0>
        x(qubits)                                    # |00...0> -> |11...1>
        z.ctrl(qubits[0:n - 1], qubits[n - 1])       # sign flip of |11...1>, i.e. of |s> in the original basis
        x(qubits)                                    # undo X
        h(qubits)                                    # undo H (result: -D, the minus sign is a global phase)

    @cudaq.kernel
    def grover_search(n: int, marked: list[int], iterations: int):
        """Full search: uniform superposition, `iterations` Grover iterations, measurement."""
        qubits = cudaq.qvector(n)                    # n qubits, all starting in |0>
        h(qubits)                                    # uniform superposition |s> over all 2^n bitstrings
        for _ in range(iterations):                  # changing `iterations` changes P(marked) = sin^2((2k+1)theta)
            grover_iteration(qubits, marked)
        mz(qubits)                                   # measure every qubit in the Z basis

    @cudaq.kernel
    def uniform_superposition(n: int):
        """|s> without measurement: the k = 0 point of the iteration sweep."""
        qubits = cudaq.qvector(n)
        h(qubits)

    @cudaq.kernel
    def continue_iterations(state: cudaq.State, marked: list[int], steps: int):
        """Load a previously computed state and apply `steps` further Grover iterations."""
        qubits = cudaq.qvector(state)                # initialise the register from an existing state vector
        for _ in range(steps):
            grover_iteration(qubits, marked)

    return {"search": grover_search, "uniform": uniform_superposition, "continue": continue_iterations}


kernels_vis = build_grover_kernels()                 # kernel objects used for drawing and statistics only
print("Kernels defined:", ", ".join(kernels_vis))
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7b - Circuit visualization (at most 10 qubits) and full-circuit statistics
# ---------------------------------------------------------------------------------
n_draw = min(int(DRAW_QUBITS), 10, N_QUBITS)         # qubits shown in the drawing (never more than 10)
draw_marked = marked_bits[:n_draw]                    # same marked pattern, truncated to the drawn qubits
print(f"The drawing shows ONE Grover iteration on {n_draw} qubits (marked item {marked_bitstring[:n_draw]}).")
print(f"The executed circuit uses {N_QUBITS} qubits and repeats the iteration {iterations:,} times.\n")
show_circuit(kernels_vis["search"], (n_draw, draw_marked, 1), n_draw, N_QUBITS,
             title=f"Executed circuit: {N_QUBITS} qubits, {iterations} iterations.",
             export=EXPORT_CIRCUIT, formats=EXPORT_FORMATS, out_dir=EXPORT_DIR, basename="grover",
             full_kernel=kernels_vis["search"], full_args=(N_QUBITS, marked_bits, iterations))

# Statistics of the FULL circuit. estimate_resources counts gates without simulating. The circuit
# repeats one block, so counts for 0, 1 and 2 iterations give the exact totals for any k.
stats0 = circuit_statistics(kernels_vis["search"], N_QUBITS, marked_bits, 0)
stats1 = circuit_statistics(kernels_vis["search"], N_QUBITS, marked_bits, 1)
stats2 = circuit_statistics(kernels_vis["search"], N_QUBITS, marked_bits, 2)
if "error" in stats1:
    print(stats1["error"])
else:
    per_iteration = {g: stats1["gate counts by type"].get(g, 0) - stats0["gate counts by type"].get(g, 0)
                     for g in stats1["gate counts by type"]}
    totals = {g: stats0["gate counts by type"].get(g, 0) + iterations * c for g, c in per_iteration.items()}
    depth = stats1["depth"] + (iterations - 1) * (stats2["depth"] - stats1["depth"]) if iterations > 0 else stats0["depth"]
    print(f"\nFULL circuit statistics ({N_QUBITS} qubits, {iterations:,} iterations):")
    print(f"  gate counts by type : {totals}   ('c...cz' = multi-controlled Z with {N_QUBITS - 1} controls)")
    print(f"  total operations    : {sum(totals.values()):,} (including {totals.get('mz', 0)} measurements)")
    print(f"  circuit depth       : {depth:,}")
'''))

cells.append(md(r"""
---
## Section 8 - Execution with live memory monitoring

Running the next cell:
1. initialises the selected backend,
2. builds the quantum circuit (fresh kernels),
3. executes Grover's search with `SHOTS` shots and, optionally, the iteration sweep,
4. monitors host RAM and GPU memory during the whole cell (live line below the cell),
5. prints a short summary. Detailed results follow in Sections 9 and 10.
""" + EXECUTION_MD))
cells.append(code(r'''
# =====================================================================================
# >>> EXECUTION CELL - Grover's algorithm with live memory monitoring <<<
# =====================================================================================
execution_ok, execution_error = False, None
sweep_iterations, sweep_probabilities, sweep_mode, sweep_time = [], [], "not run", 0.0
monitor = MemoryMonitor(MEMORY_SAMPLE_INTERVAL_S, live=LIVE_MEMORY_DISPLAY)
monitor.start()                                                     # background memory sampling starts now
cell_start = time.perf_counter()
try:
    # Step 1 - initialise the backend (again, so that this cell is self-contained)
    initialize_backend(backend_cfg, seed=SEED)

    # Step 2 - build the circuit: brand-new kernel objects, compiled on their first launch
    kernels = build_grover_kernels()

    # Step 3 - execute the full Grover search; the first launch is traced to separate
    #          construction / compilation / execution time
    counts, phases = profile_first_launch(
        lambda: cudaq.sample(kernels["search"], N_QUBITS, marked_bits, iterations, shots_count=SHOTS))

    # Step 4 - P(marked) as a function of the number of iterations
    sweep_start = time.perf_counter()
    if RUN_ITERATION_SWEEP and uses_state_sweep:
        sweep_mode = "exact amplitudes (state vector evolved iteration by iteration)"
        stride = max(1, math.ceil(sweep_max / SWEEP_MAX_POINTS))    # iterations between recorded points
        state = cudaq.get_state(kernels["uniform"], N_QUBITS)       # k = 0: uniform superposition
        k_done = 0
        sweep_iterations.append(0)
        sweep_probabilities.append(abs(state.amplitude(marked_bits)) ** 2)   # |<x*|psi>|^2
        while k_done + stride <= sweep_max:
            state = cudaq.get_state(kernels["continue"], state, marked_bits, stride)
            k_done += stride
            sweep_iterations.append(k_done)
            sweep_probabilities.append(abs(state.amplitude(marked_bits)) ** 2)
        del state                                                   # free the state vector memory
    elif RUN_ITERATION_SWEEP:
        sweep_mode = f"sampled with {SHOTS} shots per point (this method/backend has no state access)"
        for k in np.unique(np.linspace(0, sweep_max, num=min(SWEEP_MAX_POINTS, 12)).round().astype(int)):
            result_k = cudaq.sample(kernels["search"], N_QUBITS, marked_bits, int(k), shots_count=SHOTS)
            sweep_iterations.append(int(k))
            sweep_probabilities.append(result_k.count(marked_bitstring) / SHOTS)
    sweep_time = time.perf_counter() - sweep_start
    execution_ok = True
except Exception as exc:                                            # report errors instead of crashing later cells
    execution_error = exc
    print(f"\nEXECUTION FAILED: {type(exc).__name__}: {exc}")
    print("Hints: reduce N_QUBITS, choose another BACKEND / SIMULATION_METHOD, or check the GPU driver (README).")
finally:
    memory = monitor.stop()                                         # always stop the sampler
    total_wall = time.perf_counter() - cell_start

if execution_ok:
    phases["total_wall_s"] = total_wall
    measured_marked = counts.count(marked_bitstring)                # shots that returned x*
    p_measured = measured_marked / SHOTS
    most_frequent = counts.most_probable()
    print(f"\nMost frequent bitstring : {most_frequent}  ({'= marked item' if most_frequent == marked_bitstring else 'NOT the marked item'})")
    print(f"Marked-state counts     : {measured_marked} of {SHOTS}  ->  P(marked) = {p_measured:.4f} (theory {p_theory:.4f})")
    print(f"Distinct outcomes       : {len(counts)}")
    print(f"Timing (first launch)   : construction {fmt_seconds(phases['construction_s'])}, "
          f"compilation {fmt_seconds(phases['compilation_s'])}, execution {fmt_seconds(phases['execution_s'])}")
    print(f"Iteration sweep         : {len(sweep_iterations)} points in {sweep_time:.2f} s ({sweep_mode})")
    print(f"Total cell wall time    : {total_wall:.2f} s\n")
print_memory_summary(memory)
'''))

cells.append(md(r"""
---
## Section 9 - Results and visualization

* **Left:** the most frequent measured bitstrings (the marked item highlighted). This is the measured
  probability distribution: counts / `SHOTS`.
* **Right:** probability of the marked item versus the number of Grover iterations - simulated points on top of
  the exact theory curve $\sin^2((2k+1)\theta)$.
* **Bottom:** memory used during the execution cell.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 9 - Measurement table and plots
# ---------------------------------------------------------------------------------
if not execution_ok:
    print("Skipped: the execution cell failed (see the error message above).")
else:
    outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)   # (bitstring, count), most frequent first
    print(f"{'rank':>4}  {'bitstring':<{max(9, N_QUBITS)}}  {'counts':>7}  {'probability':>11}  marked?")
    for rank, (bits, c) in enumerate(outcomes[:10], start=1):
        print(f"{rank:>4}  {bits:<{max(9, N_QUBITS)}}  {c:>7}  {c / SHOTS:>11.4f}  {'<-- x*' if bits == marked_bitstring else ''}")
    others = SHOTS - measured_marked
    print(f"\nAll other bitstrings together: {others} counts ({others / SHOTS:.4f}); "
          f"theory: {(1 - p_theory):.6f} spread over {search_space - 1:,} bitstrings")
    n_stat = 1.0 / math.sqrt(SHOTS)
    print(f"Statistical resolution of the measured probabilities: about +-{n_stat:.3f} (1/sqrt(SHOTS))")

    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.45, wspace=0.35)

    # (a) measured distribution: horizontal bars, marked bitstring in blue, everything else muted
    ax_a = fig.add_subplot(grid[0, 0])
    shown = outcomes[:10]                                            # the 10 most frequent outcomes
    n_slots = 10                                                     # fixed number of rows keeps bars thin
    positions = [n_slots - 1 - r for r in range(len(shown))]         # rank 1 at the top
    labels = [b for b, _ in shown]
    values = [c for _, c in shown]
    colors = [BLUE if b == marked_bitstring else INK_MUTED for b in labels]
    ax_a.barh(positions, values, height=0.6, color=colors, edgecolor=SURFACE, linewidth=1.5)
    ax_a.set_yticks(positions, labels, family="DejaVu Sans Mono", fontsize=8)
    ax_a.set_ylim(-0.6, n_slots - 0.4)
    ax_a.set_xlabel(f"counts (of {SHOTS} shots)")
    ax_a.set_title(f"Measured bitstrings: top {len(shown)} of {len(counts)} distinct outcomes", loc="left")
    for pos, v in zip(positions, values):
        ax_a.text(v, pos, f" {v}", va="center", fontsize=8, color=INK_SECONDARY)
    if len(shown) == 1:
        ax_a.text(0.02, 0.35, "every shot returned the same bitstring\n(blue = marked item x*)",
                  transform=ax_a.transAxes, fontsize=9, color=INK_SECONDARY)
    ax_a.set_xlim(0, max(values) * 1.18)
    ax_a.grid(axis="y", visible=False)

    # (b) P(marked) versus iterations: theory curve + simulated points
    ax_b = fig.add_subplot(grid[0, 1])
    k_max_plot = max(sweep_iterations + [iterations, 1])
    k_fine = np.linspace(0, k_max_plot, 600)
    ax_b.plot(k_fine, np.sin((2 * k_fine + 1) * theta) ** 2, color=INK_SECONDARY, linewidth=1.2,
              label=r"theory $\sin^2((2k+1)\theta)$")
    if sweep_iterations:
        ax_b.plot(sweep_iterations, sweep_probabilities, "o", color=BLUE, markeredgecolor=SURFACE,
                  markeredgewidth=1.0, markersize=5, label="CUDA-Q simulation")
    ax_b.plot([iterations], [p_measured], "D", color=ORANGE, markeredgecolor=SURFACE, markersize=8,
              label=f"main run ({SHOTS} shots)")
    ax_b.axvline(k_optimal, color=AXIS_COLOR, linewidth=1)
    ax_b.text(k_optimal, 0.02, f" k_opt = {k_optimal}", ha="left", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax_b.set_ylim(-0.02, 1.1)
    ax_b.set_xlabel("number of Grover iterations k")
    ax_b.set_ylabel("P(marked bitstring)")
    ax_b.set_title("Amplitude amplification and over-rotation", loc="left")
    ax_b.legend(loc="lower center", fontsize=8)

    # (c) memory timeline
    ax_c = fig.add_subplot(grid[1, :])
    monitor.plot(ax=ax_c, title="Memory during the execution cell")
    plt.show()

    if sweep_iterations:
        theory_at_points = np.sin((2 * np.array(sweep_iterations) + 1) * theta) ** 2
        deviation = np.max(np.abs(np.array(sweep_probabilities) - theory_at_points))
        print(f"Sweep ({sweep_mode}): max |simulation - theory| = {deviation:.2e}")
'''))

cells.append(md(r"""
---
## Section 10 - Final benchmark report

The report collects the configuration, the timing breakdown, the measured memory peaks and the comparison
between expected and actual results. It is also saved as `benchmark_report.json` (+ figure) in `EXPORT_DIR`
when `SAVE_REPORT = True`, so runs on different machines and backends can be compared.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 10 - Final benchmark report
# ---------------------------------------------------------------------------------
if not execution_ok:
    print(f"No report: execution failed with {type(execution_error).__name__}: {execution_error}")
else:
    report = build_report("Grover's algorithm (quantum search)", backend_cfg, N_QUBITS, SHOTS, phases, memory,
                          extra_times={"Iteration sweep time (execution)": sweep_time})
    report["Grover iterations (k)"] = f"{iterations:,} (optimal: {k_optimal:,})"
    report["Marked bitstring"] = marked_bitstring
    report["Expected result"] = f"P(marked) = {p_theory:.6f} -> about {expected_counts:.0f} of {SHOTS} shots"
    report["Actual result"] = f"P(marked) = {p_measured:.4f} ({measured_marked}/{SHOTS}); most frequent = {most_frequent}"
    report["Error |P_measured - P_theory|"] = f"{abs(p_measured - p_theory):.4f} (1-sigma shot noise: {count_spread / SHOTS:.4f})"
    if sweep_iterations:
        report["Sweep max deviation from theory"] = f"{deviation:.2e} ({sweep_mode})"
    report["Search succeeded"] = "YES" if most_frequent == marked_bitstring else "NO"

    final_fig, (fa, fb) = plt.subplots(1, 2, figsize=(11, 3.6))
    fa.barh(positions, values, height=0.6, color=colors, edgecolor=SURFACE, linewidth=1.5)
    fa.set_yticks(positions, labels, family="DejaVu Sans Mono", fontsize=8)
    fa.set_ylim(-0.6, n_slots - 0.4)
    fa.set_xlabel("counts")
    fa.set_title(f"Grover N={N_QUBITS}: measured bitstrings", loc="left")
    fa.grid(axis="y", visible=False)
    fb.plot(k_fine, np.sin((2 * k_fine + 1) * theta) ** 2, color=INK_SECONDARY, linewidth=1.2, label="theory")
    if sweep_iterations:
        fb.plot(sweep_iterations, sweep_probabilities, "o", color=BLUE, markeredgecolor=SURFACE, markersize=5, label="simulation")
    fb.set_xlabel("Grover iterations k")
    fb.set_ylabel("P(marked)")
    fb.set_title(f"P(marked) vs k  ({cudaq.get_target().name})", loc="left")
    fb.legend(fontsize=8)
    final_fig.tight_layout()
    show_report(report, save=SAVE_REPORT, out_dir=EXPORT_DIR, name="grover_benchmark_report", figure=final_fig)
    plt.show()
'''))

cells.append(md(r"""
---
### Things to try next
* Increase `N_QUBITS` to 20 / 25 / 30 with `BACKEND = "gpu"` and compare the execution times and memory peaks
  in the saved reports - the state vector doubles with every qubit and $k_{\text{opt}}$ grows by $\sqrt{2}$.
* Set `GPU_PRECISION = "fp32"` to halve the GPU memory and compare the speed.
* Set `GROVER_ITERATIONS` to $k_{\text{opt}}/2$ or $2k_{\text{opt}}$ and check the measured probability against
  the sweep curve.
* `BACKEND = "hardware"` with `HARDWARE_EMULATE = True` compiles the circuit for a real QPU's native gate set and
  runs it in the provider's local emulator (keep N small: multi-controlled gates decompose into many native gates).
"""))

write_notebook(OUT, cells, "Grover's Algorithm - CUDA-Q benchmark")
print("wrote", OUT)
