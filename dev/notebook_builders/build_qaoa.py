from nbgen import md, code, common_imports, common_utils, write_notebook
from common_md import (title_md, INSTALL_MD, params_md, common_input_block, UTILS_MD,
                       backend_init_code, EXECUTION_MD)
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "QAOA.ipynb"
ALLOWED = ["statevector", "mps", "tensornet", "density_matrix"]

cells = []
cells.append(md(title_md(
    2, "QAOA - Quantum Approximate Optimization Algorithm", "Solving Max-Cut on a graph with NVIDIA CUDA-Q",
    "This notebook encodes the Max-Cut problem of a graph in a cost Hamiltonian, optimises a depth-p QAOA "
    "circuit with a classical optimizer, samples candidate solutions, and compares them with the exact "
    "optimum found by brute force. Time and memory are measured on the backend you select.")))

cells.append(md(r"""
---
## Section 1 - Problem statement and mathematical explanation

### What problem does the algorithm solve?
**Max-Cut.** Given a graph with nodes $0,\dots,N-1$ and edges $(i,j)$ with weights $w_{ij}$, split the nodes into
two groups so that the total weight of edges *between* the groups (the **cut**) is as large as possible.
A split is a bitstring $z = z_0 z_1 \dots z_{N-1}$ ($z_i$ = group of node $i$) and

$$C(z) = \sum_{(i,j)\in E} w_{ij}\,\big[z_i \neq z_j\big].$$

### Why is this problem important?
Max-Cut is NP-hard and is the standard test bed for quantum optimisation. Many practical problems (portfolio
optimisation, scheduling, circuit layout, clustering) map onto the same "quadratic unconstrained binary
optimisation" (QUBO) form, so the QAOA workflow shown here carries over directly.

### What is the classical approach?
Exact: try all $2^N$ splits ($O(2^N)$ - used below as the reference for small graphs). Practical: heuristics such as
greedy local search, simulated annealing, or the Goemans-Williamson SDP algorithm (guaranteed ratio 0.878).

### How does the quantum algorithm solve it?
Each node becomes a qubit ($Z$-eigenvalue $+1$ for group 0, $-1$ for group 1). Since $[z_i\neq z_j] = (1 - Z_iZ_j)/2$,

$$C = \frac{W}{2} - H_C,\qquad H_C = \sum_{(i,j)\in E}\frac{w_{ij}}{2}\,Z_iZ_j,\qquad W = \sum_{(i,j)} w_{ij},$$

so **maximising the cut = minimising the energy of $H_C$**. QAOA prepares the trial state

$$|\boldsymbol\gamma,\boldsymbol\beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l H_M}\,e^{-i\gamma_l H_C}\;|+\rangle^{\otimes N},
\qquad H_M = \sum_i X_i ,$$

* cost layer $e^{-i\gamma H_C}$ = for every edge `CNOT(i,j)`, `RZ(gamma*w_ij)` on $j$, `CNOT(i,j)`,
* mixer layer $e^{-i\beta H_M}$ = `RX(2*beta)` on every qubit,

and a classical optimizer adjusts the $2p$ angles to maximise the expected cut
$F(\boldsymbol\gamma,\boldsymbol\beta) = \langle C\rangle = W/2 - \langle H_C\rangle$, computed with `cudaq.observe`.
Finally the optimised circuit is **sampled**: every measured bitstring is a candidate split whose cut value
$C(z)$ is computed classically, and the best one is returned.

### Measurement
Sampling in the $Z$ basis returns bitstrings $z$ (qubit 0 first = node 0 first). Their frequencies estimate
$|\langle z|\boldsymbol\gamma,\boldsymbol\beta\rangle|^2$; good parameters concentrate probability on large cuts.

### What is the expected solution?
The **approximation ratio** $r = \langle C\rangle / C_{\max}$ measures the quality. A random split gives
$\langle C\rangle = W/2$. For $p = 1$ on 3-regular graphs QAOA is proven to reach $r \ge 0.6924$; for other graphs
there is no fixed guarantee, so the notebook computes the exact optimum $C_{\max}$ by brute force (for
$N \le 22$) and reports $r$, the best sampled cut and the probability of sampling an optimal split.
"""))

cells.append(md(INSTALL_MD))
cells.append(common_imports("from scipy.optimize import minimize               # classical optimizer for the QAOA angles"))

cells.append(md("""
---
## Section 3 - Input parameters

This is the **only cell you need to edit**. Every later cell reads these variables.
"""))
cells.append(code(r'''
# =====================================================================================
# Section 3 - INPUT PARAMETERS  (edit only this cell, then re-run the notebook)
# =====================================================================================

# ---- Problem definition: the graph -----------------------------------------------------
N_QUBITS = 15                    # number of graph nodes = number of qubits (try 20 / 25 / 30 on GPUs)
GRAPH = "example"                # "example" (random graph, Section 4) | "ring" | "complete" | "custom"
EDGE_PROBABILITY = 0.3           # "example": probability that two nodes are connected
CUSTOM_EDGES = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]   # "custom": list of (node_i, node_j, weight)

# ---- QAOA and optimizer ------------------------------------------------------------------
QAOA_LAYERS = 2                  # depth p (number of cost + mixer layers); 2p angles are optimised
OPTIMIZER = "COBYLA"             # "COBYLA" | "Nelder-Mead" | "Powell" | "L-BFGS-B"
MAX_EVALUATIONS = 80             # budget of objective-function evaluations (= quantum circuit runs)
INITIAL_PARAMETERS = "ramp"      # "ramp" (gamma rising, beta falling: annealing-like) or a list of 2p numbers
OBJECTIVE_SHOTS = None           # None: exact <H_C> (simulators); an integer: estimate <H_C> from that many shots
BRUTE_FORCE_MAX_NODES = 22       # exact classical optimum only up to this many nodes (2^N cuts are evaluated)
TOP_BITSTRINGS = 12              # rows in the table of measured bitstrings
''' + common_input_block(ALLOWED, "statevector", 4000, "exports/qaoa")))

cells.append(md(params_md(r"""
| `N_QUBITS` | Number of graph nodes = qubits. | State memory doubles per node (16 x 2^N bytes, fp64); the number of edges (and gates) of the example graph grows like $N^2$. The brute-force reference is skipped above `BRUTE_FORCE_MAX_NODES` and replaced by a heuristic. |
| `GRAPH` / `EDGE_PROBABILITY` / `CUSTOM_EDGES` | The Max-Cut instance. | More edges = more two-qubit gates per layer and a harder landscape. Custom edges must use node numbers `0 ... N_QUBITS-1`; weights may be any real numbers. |
| `QAOA_LAYERS` | Depth $p$. | Larger $p$ can reach higher approximation ratios (exact as $p\to\infty$) but has more angles to optimise and $p$ times more gates. |
| `OPTIMIZER` | SciPy method that adjusts the angles. | COBYLA / Nelder-Mead / Powell are gradient-free; L-BFGS-B uses finite-difference gradients ($2p+1$ circuit runs per gradient). |
| `MAX_EVALUATIONS` | Maximum number of objective evaluations. | Run time is roughly proportional; too few evaluations stop before convergence (the convergence plot shows it). |
| `INITIAL_PARAMETERS` | Starting angles. | QAOA landscapes have local optima; the annealing-like ramp is a robust default. |
| `OBJECTIVE_SHOTS` | Exact or shot-based objective. | Shot-based objectives are noisy (as on real hardware) and make optimisation harder; on hardware targets shots are always used. |
| `BRUTE_FORCE_MAX_NODES` | Limit for the exact classical reference. | Brute force costs $2^N$ cut evaluations (N=22: 4 million). |
| `TOP_BITSTRINGS` | Rows in the bitstring table. | Display only. |
""")))

cells.append(md(r"""
---
## Section 4 - Example inputs

**Default example (runs as is):** a random graph on `N_QUBITS = 15` nodes where each pair of nodes is connected
with probability `EDGE_PROBABILITY = 0.3` (unit weights), generated reproducibly from `SEED`.

| Goal | Setting in Section 3 | Notes |
|---|---|---|
| Default random graph | `GRAPH = "example"` | about 31 edges for N = 15, SEED = 42 |
| Ring (cycle) graph | `GRAPH = "ring"` | max cut = N for even N, N-1 for odd N |
| Complete graph | `GRAPH = "complete"` | max cut = floor(N/2) * ceil(N/2); hard for low depth |
| Your own graph | `GRAPH = "custom"`, `CUSTOM_EDGES = [(0, 1, 1.0), (1, 2, 2.5), ...]`, set `N_QUBITS` to the node count | weighted edges allowed |
| Bigger instance | `N_QUBITS = 24`, `BACKEND = "gpu"` | about 83 edges |

The cell below builds the graph, validates it, computes the exact optimum classically (brute force) and draws it.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 4 - Build and validate the Max-Cut instance, compute the classical reference
# ---------------------------------------------------------------------------------
if not isinstance(N_QUBITS, (int, np.integer)) or N_QUBITS < 2:
    raise ValueError("N_QUBITS (number of nodes) must be an integer >= 2.")
N_QUBITS = int(N_QUBITS)
graph_kind = str(GRAPH).strip().lower()
if graph_kind == "example":
    graph_rng = np.random.default_rng(SEED)                        # reproducible random graph
    edges = [(i, j, 1.0) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)
             if graph_rng.random() < EDGE_PROBABILITY]
elif graph_kind == "ring":
    edges = [(i, (i + 1) % N_QUBITS, 1.0) for i in range(N_QUBITS)] if N_QUBITS > 2 else [(0, 1, 1.0)]
elif graph_kind == "complete":
    edges = [(i, j, 1.0) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]
elif graph_kind == "custom":
    edges = [(int(e[0]), int(e[1]), float(e[2]) if len(e) > 2 else 1.0) for e in CUSTOM_EDGES]
else:
    raise ValueError("GRAPH must be 'example', 'ring', 'complete' or 'custom'.")
if not edges:
    raise ValueError("The graph has no edges - increase EDGE_PROBABILITY or provide CUSTOM_EDGES.")
for i, j, w in edges:
    if not (0 <= i < N_QUBITS and 0 <= j < N_QUBITS) or i == j:
        raise ValueError(f"Invalid edge ({i}, {j}): nodes must be distinct and in 0..{N_QUBITS - 1}.")
edge_i = [e[0] for e in edges]                                     # kernel arguments: first node of each edge
edge_j = [e[1] for e in edges]                                     # second node of each edge
edge_w = [e[2] for e in edges]                                     # weight of each edge
total_weight = float(sum(edge_w))                                  # W: random split has <C> = W/2

def cut_value(bitstring):
    """C(z) for a bitstring z (character i = group of node i)."""
    return sum(w for i, j, w in edges if bitstring[i] != bitstring[j])

def brute_force_maxcut():
    """Exact optimum by evaluating all 2^N splits with vectorised NumPy (node 0 = most significant bit)."""
    index = np.arange(2 ** N_QUBITS, dtype=np.int64)
    values = np.zeros(2 ** N_QUBITS)
    for i, j, w in edges:
        values += w * (((index >> (N_QUBITS - 1 - i)) ^ (index >> (N_QUBITS - 1 - j))) & 1)
    best = values.max()
    optimal = [format(int(z), f"0{N_QUBITS}b") for z in np.flatnonzero(np.isclose(values, best))]
    return best, optimal

def local_search_maxcut(restarts=200, seed=SEED):
    """Heuristic for large graphs: greedy single-node flips from random starts (NOT guaranteed optimal)."""
    rng = np.random.default_rng(seed)
    best_value, best_z = -np.inf, None
    for _ in range(restarts):
        z = list(rng.integers(0, 2, N_QUBITS).astype(str))
        improved = True
        while improved:
            improved = False
            for node in range(N_QUBITS):
                before = cut_value("".join(z))
                z[node] = "1" if z[node] == "0" else "0"
                if cut_value("".join(z)) > before:
                    improved = True
                else:
                    z[node] = "1" if z[node] == "0" else "0"     # undo the flip
        value = cut_value("".join(z))
        if value > best_value:
            best_value, best_z = value, "".join(z)
    return best_value, [best_z]

t0 = time.perf_counter()
if N_QUBITS <= BRUTE_FORCE_MAX_NODES:
    max_cut, optimal_bitstrings = brute_force_maxcut()
    reference_kind = "exact (brute force over all 2^N splits)"
else:
    max_cut, optimal_bitstrings = local_search_maxcut()
    reference_kind = "heuristic local search (best known, NOT guaranteed optimal)"
classical_time = time.perf_counter() - t0

print(f"Nodes (qubits)        : {N_QUBITS}")
print(f"Edges                 : {len(edges)}  (total weight W = {total_weight:g})")
print(f"Random-split baseline : <C> = W/2 = {total_weight / 2:g}")
print(f"Classical reference   : max cut = {max_cut:g}  [{reference_kind}, {classical_time:.2f} s]")
print(f"Optimal splits        : {len(optimal_bitstrings)} (e.g. {optimal_bitstrings[0]})")

def draw_graph(ax, split=None, title="Graph"):
    """Nodes on a circle; with a split: blue/orange groups, cut edges dark, uncut edges light."""
    angle = 2 * np.pi * np.arange(N_QUBITS) / N_QUBITS
    pos = np.c_[np.cos(angle), np.sin(angle)]
    for i, j, w in edges:
        is_cut = split is not None and split[i] != split[j]
        ax.plot(*pos[[i, j]].T, color=INK_SECONDARY if is_cut or split is None else GRID_COLOR,
                linewidth=1.6 if is_cut else 1.0, zorder=1)
    node_colors = [BLUE if split is None or split[k] == "0" else ORANGE for k in range(N_QUBITS)]
    ax.scatter(pos[:, 0], pos[:, 1], s=260, c=node_colors, edgecolors=SURFACE, linewidths=2, zorder=2)
    for k in range(N_QUBITS):
        ax.text(*pos[k], str(k), ha="center", va="center", fontsize=8, color="white", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left")
    if split is not None:                                          # legend: node groups and cut edges
        handles = [matplotlib.lines.Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=8, label="group 0"),
                   matplotlib.lines.Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=8, label="group 1"),
                   matplotlib.lines.Line2D([], [], color=INK_SECONDARY, linewidth=1.6, label="cut edge"),
                   matplotlib.lines.Line2D([], [], color=GRID_COLOR, linewidth=1.0, label="uncut edge")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
draw_graph(ax, title=f"Max-Cut instance: {N_QUBITS} nodes, {len(edges)} edges")
plt.show()
'''))

cells.append(md(UTILS_MD))
cells.append(common_utils())
cells.append(code(backend_init_code(ALLOWED) + r'''
# Hardware targets estimate expectation values from shots; simulators can compute them exactly.
objective_shots = OBJECTIVE_SHOTS if OBJECTIVE_SHOTS is not None else (SHOTS if backend_cfg["device"] == "hardware" else None)
print(f"Objective evaluation  : {'exact expectation value' if objective_shots is None else f'{objective_shots} shots per evaluation'}")
'''))

cells.append(md(r"""
---
## Section 6 - Expected output (before execution)

**Expected graphs (Section 9)**
1. *Objective convergence* - expected cut $\langle C\rangle$ versus optimizer evaluation, rising from the
   random-split value $W/2$ towards (but below) the maximum cut. A flat end means the optimizer converged.
2. *Bitstring distribution* - the most frequently measured splits with their cut values; optimal splits highlighted.
3. *Best split drawn on the graph* (cut edges dark) and the *memory timeline*.

**Expected numbers:** the brute-force optimum $C_{\max}$ (Section 4) is the target. Typical depth-2 QAOA results
on random graphs reach $r \approx 0.75-0.9$ - a typical range, not a guarantee. The best *sampled* split is often
optimal when the optimal splits receive enough probability (expected optimal hits = $P_{\text{opt}}\times$ `SHOTS`).

**Execution time and memory** - exact state memory and a calibrated (rough) time estimate are computed below.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6 - Memory requirement and execution-time estimate
# ---------------------------------------------------------------------------------
state_bytes = estimate_memory_bytes(N_QUBITS, backend_cfg["method"], backend_cfg["precision"])
print("Expected memory")
check_memory(state_bytes, backend_cfg, MEMORY_SAFETY_FRACTION)

gates_per_circuit = N_QUBITS + QAOA_LAYERS * (3 * len(edges) + N_QUBITS)   # H layer + p * (3 gates/edge + RX layer)
seconds_per_gate, timing_note = measure_gate_time(backend_cfg, N_QUBITS)
evaluations = MAX_EVALUATIONS + 1                                   # optimizer budget + final sampling run
print("\nExpected execution time (ESTIMATE)")
print(f"  Gates per QAOA circuit : {gates_per_circuit:,} ({len(edges)} edges x 3 gates x {QAOA_LAYERS} layers + rotations)")
print(f"  Circuit evaluations    : up to {evaluations}")
print(f"  Calibration            : {timing_note}")
if seconds_per_gate is not None:
    print(f"  Estimated total        : ~{format_duration(evaluations * gates_per_circuit * seconds_per_gate)} "
          "(expectation values add a few % on simulators)")
print(f"\nTarget: maximum cut {max_cut:g}; random-split baseline {total_weight / 2:g}")
'''))

cells.append(md(r"""
---
## Section 7 - Quantum circuit generation

* `qaoa_state` - prepares $|\boldsymbol\gamma,\boldsymbol\beta\rangle$ (used by `cudaq.observe` during optimisation).
* `qaoa_sample` - the same circuit followed by measurement (used for the final bitstring distribution).
* `cost_hamiltonian` - $H_C = \sum_{(i,j)} \tfrac{w_{ij}}{2} Z_i Z_j$ as a CUDA-Q spin operator.

The angle vector is `params = [gamma_1 ... gamma_p, beta_1 ... beta_p]`.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7a - CUDA-Q kernels and cost Hamiltonian for QAOA
# ---------------------------------------------------------------------------------
def build_qaoa_kernels():
    """Create and return fresh CUDA-Q kernels (called again in the execution cell)."""

    @cudaq.kernel
    def qaoa_state(n: int, edge_i: list[int], edge_j: list[int], weights: list[float],
                   layers: int, params: list[float]):
        """|gamma, beta> = prod_l exp(-i beta_l H_M) exp(-i gamma_l H_C) |+>^n."""
        qubits = cudaq.qvector(n)
        h(qubits)                                            # |+>^n: every split equally likely
        for layer in range(layers):
            gamma = params[layer]                            # cost angle of this layer
            beta = params[layers + layer]                    # mixer angle of this layer
            for e in range(len(edge_i)):                     # exp(-i gamma w/2 Z_i Z_j) for every edge
                x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
                rz(gamma * weights[e], qubits[edge_j[e]])    # RZ(t) = exp(-i t Z/2) -> phase depends on z_i XOR z_j
                x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
            for q in range(n):
                rx(2.0 * beta, qubits[q])                    # mixer exp(-i beta X) moves probability between splits

    @cudaq.kernel
    def qaoa_sample(n: int, edge_i: list[int], edge_j: list[int], weights: list[float],
                    layers: int, params: list[float]):
        """Same circuit, then measure every qubit (bit i = group of node i)."""
        qubits = cudaq.qvector(n)
        h(qubits)
        for layer in range(layers):
            gamma = params[layer]
            beta = params[layers + layer]
            for e in range(len(edge_i)):
                x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
                rz(gamma * weights[e], qubits[edge_j[e]])
                x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
            for q in range(n):
                rx(2.0 * beta, qubits[q])
        mz(qubits)

    return {"state": qaoa_state, "sample": qaoa_sample}


def cost_hamiltonian():
    """H_C = sum_(i,j) w_ij/2 Z_i Z_j  (maximising the cut <=> minimising <H_C>)."""
    hamiltonian = None
    for i, j, w in edges:
        term = 0.5 * w * cudaq.spin.z(i) * cudaq.spin.z(j)       # one Z_i Z_j term per edge
        hamiltonian = term if hamiltonian is None else hamiltonian + term
    return hamiltonian


if isinstance(INITIAL_PARAMETERS, str) and INITIAL_PARAMETERS.strip().lower() == "ramp":
    initial_params = np.concatenate([np.linspace(0.2, 0.6, QAOA_LAYERS),        # gamma_1 .. gamma_p rising
                                     np.linspace(0.6, 0.2, QAOA_LAYERS)])       # beta_1 .. beta_p falling
else:
    initial_params = np.asarray(INITIAL_PARAMETERS, dtype=float)
    if initial_params.size != 2 * QAOA_LAYERS:
        raise ValueError(f"INITIAL_PARAMETERS needs 2 * QAOA_LAYERS = {2 * QAOA_LAYERS} numbers.")
H_C = cost_hamiltonian()
kernels_vis = build_qaoa_kernels()
print(f"Cost Hamiltonian: {H_C.term_count} Z_iZ_j terms on {N_QUBITS} qubits")
print(f"Initial angles  : gamma = {np.round(initial_params[:QAOA_LAYERS], 3).tolist()}, "
      f"beta = {np.round(initial_params[QAOA_LAYERS:], 3).tolist()}")
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7b - Circuit visualization (at most 10 qubits) and full-circuit statistics
# ---------------------------------------------------------------------------------
n_draw = min(int(DRAW_QUBITS), 10, N_QUBITS)
if not 0 < n_draw <= 10:
    raise ValueError("DRAW_QUBITS must be between 1 and 10.")
draw_edges = [(i, j, w) for i, j, w in edges if i < n_draw and j < n_draw]   # sub-graph on the first n_draw nodes
if not draw_edges:
    draw_edges = [(0, 1, 1.0)]
print(f"The drawing uses the sub-graph of the first {n_draw} nodes ({len(draw_edges)} edges) and ONE QAOA layer.")
show_circuit(kernels_vis["sample"],
             (n_draw, [e[0] for e in draw_edges], [e[1] for e in draw_edges], [e[2] for e in draw_edges],
              1, [float(initial_params[0]), float(initial_params[QAOA_LAYERS])]),
             n_draw, N_QUBITS, title="One QAOA layer.", export=EXPORT_CIRCUIT, formats=EXPORT_FORMATS,
             out_dir=EXPORT_DIR, basename="qaoa", full_kernel=kernels_vis["sample"],
             full_args=(N_QUBITS, edge_i, edge_j, edge_w, QAOA_LAYERS, initial_params.tolist()))
stats = circuit_statistics(kernels_vis["sample"], N_QUBITS, edge_i, edge_j, edge_w, QAOA_LAYERS, initial_params.tolist())
print(f"\nFULL circuit statistics ({N_QUBITS} qubits, p = {QAOA_LAYERS}):")
for key, value in stats.items():
    print(f"  {key:<32}: {value}")
'''))

cells.append(md(r"""
---
## Section 8 - Execution with live memory monitoring

Running the next cell:
1. initialises the selected backend,
2. builds the QAOA circuit (fresh kernels),
3. optimises the $2p$ angles (each evaluation = one `cudaq.observe` of $H_C$) and samples the optimised circuit
   with `SHOTS` shots,
4. monitors host RAM and GPU memory during the whole cell,
5. prints a short summary.
""" + EXECUTION_MD))
cells.append(code(r'''
# =====================================================================================
# >>> EXECUTION CELL - QAOA optimisation + sampling with live memory monitoring <<<
# =====================================================================================
execution_ok, execution_error = False, None
history_cut, history_params = [], []                               # convergence record
phases, later_launch_time = None, 0.0
monitor = MemoryMonitor(MEMORY_SAMPLE_INTERVAL_S, live=LIVE_MEMORY_DISPLAY)
monitor.start()
cell_start = time.perf_counter()
try:
    # Step 1 - initialise the backend
    initialize_backend(backend_cfg, seed=SEED)
    # Step 2 - build the circuit (fresh kernels)
    kernels = build_qaoa_kernels()
    observe_kwargs = {} if objective_shots is None else {"shots_count": int(objective_shots)}

    def objective(params):
        """Minimise <H_C>; record the expected cut <C> = W/2 - <H_C> after every evaluation."""
        global phases, later_launch_time
        launch = lambda: cudaq.observe(kernels["state"], H_C, N_QUBITS, edge_i, edge_j, edge_w,
                                       QAOA_LAYERS, list(params), **observe_kwargs)
        if phases is None:                                           # first launch: traced timing breakdown
            result, phases = profile_first_launch(launch)
        else:
            t0 = time.perf_counter()
            result = launch()
            later_launch_time += time.perf_counter() - t0
        energy = result.expectation()
        history_cut.append(total_weight / 2 - energy)
        history_params.append(np.array(params, dtype=float))
        return energy

    # Step 3 - classical optimisation of the angles
    method = OPTIMIZER.strip()
    options = {"COBYLA": {"maxiter": MAX_EVALUATIONS, "rhobeg": 0.2}, "NELDER-MEAD": {"maxfev": MAX_EVALUATIONS},
               "POWELL": {"maxfev": MAX_EVALUATIONS}, "L-BFGS-B": {"maxfun": MAX_EVALUATIONS}}.get(method.upper())
    if options is None:
        raise ValueError("OPTIMIZER must be 'COBYLA', 'Nelder-Mead', 'Powell' or 'L-BFGS-B'.")
    optimisation_start = time.perf_counter()
    opt = minimize(objective, initial_params, method=method, options=options)
    optimisation_time = time.perf_counter() - optimisation_start
    best_index = int(np.argmax(history_cut))                        # best evaluated angles (not only the last)
    best_params = history_params[best_index]

    # Step 4 - sample the optimised circuit
    t0 = time.perf_counter()
    counts = cudaq.sample(kernels["sample"], N_QUBITS, edge_i, edge_j, edge_w, QAOA_LAYERS,
                          best_params.tolist(), shots_count=SHOTS)
    sampling_time = time.perf_counter() - t0
    execution_ok = True
except Exception as exc:
    execution_error = exc
    print(f"\nEXECUTION FAILED: {type(exc).__name__}: {exc}")
    print("Hints: reduce N_QUBITS or MAX_EVALUATIONS, choose another BACKEND / SIMULATION_METHOD (README).")
finally:
    memory = monitor.stop()
    total_wall = time.perf_counter() - cell_start

if execution_ok:
    phases["total_wall_s"] = total_wall
    expected_cut = history_cut[best_index]
    ratio = expected_cut / max_cut
    samples = sorted(((b, c, cut_value(b)) for b, c in counts.items()), key=lambda t: (-t[1], -t[2]))
    best_sample = max(samples, key=lambda t: (t[2], t[1]))
    optimal_set = set(optimal_bitstrings)
    optimal_hits = sum(c for b, c, v in samples if np.isclose(v, max_cut))
    print(f"\nOptimizer           : {method}, {len(history_cut)} evaluations, message: {opt.message}")
    print(f"Best expected cut   : <C> = {expected_cut:.4f}  ->  approximation ratio r = {ratio:.4f}")
    print(f"Best sampled split  : {best_sample[0]}  cut = {best_sample[2]:g}  (max cut {max_cut:g})")
    print(f"P(optimal split)    : {optimal_hits / SHOTS:.4f} ({optimal_hits} of {SHOTS} shots)")
    print(f"Timing              : first launch construction {fmt_seconds(phases['construction_s'])}, compilation "
          f"{fmt_seconds(phases['compilation_s'])}; optimisation {optimisation_time:.2f} s; sampling {sampling_time:.2f} s")
    print(f"Total cell wall time: {total_wall:.2f} s\n")
print_memory_summary(memory)
'''))

cells.append(md(r"""
---
## Section 9 - Results and visualization

The table lists the most frequently measured bitstrings (qubit/node 0 first) with their **objective value**
(cut size). Optimal splits are marked. A split and its complement (all bits flipped) describe the same cut.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 9 - Measured bitstrings, objective values and plots
# ---------------------------------------------------------------------------------
if not execution_ok:
    print("Skipped: the execution cell failed (see the error message above).")
else:
    print(f"{'rank':>4}  {'bitstring':<{max(9, N_QUBITS)}}  {'counts':>6}  {'prob.':>6}  {'cut C(z)':>8}  {'C/Cmax':>6}")
    for rank, (b, c, v) in enumerate(samples[:TOP_BITSTRINGS], start=1):
        flag = "  <-- optimal" if np.isclose(v, max_cut) else ""
        print(f"{rank:>4}  {b:<{max(9, N_QUBITS)}}  {c:>6}  {c / SHOTS:>6.3f}  {v:>8g}  {v / max_cut:>6.3f}{flag}")
    cut_of_shots = np.repeat([v for _, _, v in samples], [c for _, c, _ in samples])
    print(f"\nMean cut over all {SHOTS} shots : {cut_of_shots.mean():.3f}   (expected <C> = {expected_cut:.3f}, random = {total_weight / 2:g})")

    fig = plt.figure(figsize=(12, 8.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.5, wspace=0.42)
    # (a) objective convergence
    ax_c = fig.add_subplot(grid[0, 0:2])
    ax_c.plot(np.arange(1, len(history_cut) + 1), history_cut, color=BLUE, label=r"expected cut $\langle C\rangle$")
    ax_c.axhline(max_cut, color=INK_SECONDARY, linewidth=1)
    ax_c.text(len(history_cut), max_cut, f"max cut = {max_cut:g} ({'exact' if N_QUBITS <= BRUTE_FORCE_MAX_NODES else 'heuristic'}) ",
              ha="right", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax_c.axhline(total_weight / 2, color=AXIS_COLOR, linewidth=1)
    ax_c.text(len(history_cut), total_weight / 2, f"random split = {total_weight / 2:g} ", ha="right", va="bottom",
              fontsize=8, color=INK_SECONDARY)
    ax_c.set_xlabel("optimizer evaluation")
    ax_c.set_ylabel("objective (cut value)")
    ax_c.set_title(f"Objective convergence ({OPTIMIZER}, p = {QAOA_LAYERS})", loc="left")
    ax_c.set_ylim(min(history_cut + [total_weight / 2]) * 0.95, max_cut * 1.08)
    # (b) best split on the graph
    ax_g = fig.add_subplot(grid[0, 2])
    draw_graph(ax_g, split=best_sample[0], title=f"Best sampled split: cut {best_sample[2]:g}")
    # (c) bitstring distribution with objective values
    ax_b = fig.add_subplot(grid[1, 0:2])
    shown = samples[:TOP_BITSTRINGS]
    xpos = np.arange(len(shown))
    colors = [BLUE if np.isclose(v, max_cut) else INK_MUTED for _, _, v in shown]
    ax_b.bar(xpos, [c / SHOTS for _, c, _ in shown], width=0.45, color=colors, edgecolor=SURFACE, linewidth=1.5)
    ax_b.set_xticks(xpos, [f"{b}\nC={v:g}" for b, _, v in shown], rotation=90, fontsize=7, family="DejaVu Sans Mono")
    ax_b.set_ylabel("probability")
    ax_b.set_title("Most frequent bitstrings (blue = optimal cut, grey = sub-optimal)", loc="left")
    ax_b.grid(axis="x", visible=False)
    # (d) distribution of cut values over all shots
    ax_h = fig.add_subplot(grid[1, 2])
    values, freq = np.unique(cut_of_shots, return_counts=True)
    ax_h.bar(values, freq / SHOTS, width=0.6 * (np.min(np.diff(values)) if len(values) > 1 else 1.0),
             color=[BLUE if np.isclose(v, max_cut) else INK_MUTED for v in values], edgecolor=SURFACE)
    ax_h.set_xlabel("cut value C(z) of measured splits")
    ax_h.set_ylabel("fraction of shots")
    ax_h.set_title("Quality of all sampled splits", loc="left")
    plt.show()

    fig_m, ax_m = plt.subplots(figsize=(12, 2.8))
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
if not execution_ok:
    print(f"No report: execution failed with {type(execution_error).__name__}: {execution_error}")
else:
    phases_report = dict(phases)
    phases_report["execution_s"] = phases["execution_s"] + later_launch_time if phases["execution_s"] is not None else None
    report = build_report("QAOA (Max-Cut)", backend_cfg, N_QUBITS, SHOTS, phases_report, memory,
                          extra_times={"Final sampling run": sampling_time,
                                       "Classical optimizer + Python overhead": optimisation_time - later_launch_time
                                       - (phases["wall_s"] if phases["wall_s"] else 0)})
    report["Graph"] = f"{graph_kind}: {N_QUBITS} nodes, {len(edges)} edges, total weight {total_weight:g}"
    report["QAOA depth p / optimizer"] = f"{QAOA_LAYERS} / {OPTIMIZER} ({len(history_cut)} evaluations)"
    report["Expected result"] = f"max cut = {max_cut:g} ({reference_kind})"
    report["Actual result"] = f"best sampled cut = {best_sample[2]:g} ({best_sample[0]}); <C> = {expected_cut:.4f}"
    report["Approximation ratio <C>/Cmax"] = f"{ratio:.4f}"
    report["Best sampled cut / Cmax"] = f"{best_sample[2] / max_cut:.4f}"
    report["P(sampling an optimal split)"] = f"{optimal_hits / SHOTS:.4f}"
    final_fig, (fa, fb) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fa.plot(np.arange(1, len(history_cut) + 1), history_cut, color=BLUE)
    fa.axhline(max_cut, color=INK_SECONDARY, linewidth=1)
    fa.set_xlabel("optimizer evaluation")
    fa.set_ylabel(r"expected cut $\langle C\rangle$")
    fa.set_title(f"QAOA p={QAOA_LAYERS}, N={N_QUBITS} on {cudaq.get_target().name}", loc="left")
    draw_graph(fb, split=best_sample[0], title=f"best split, cut {best_sample[2]:g} / {max_cut:g}")
    final_fig.tight_layout()
    show_report(report, save=SAVE_REPORT, out_dir=EXPORT_DIR, name="qaoa_benchmark_report", figure=final_fig)
    plt.show()
'''))

cells.append(md(r"""
---
### Things to try next
* Increase `QAOA_LAYERS` to 3-4 (and `MAX_EVALUATIONS` to ~200): the approximation ratio usually improves.
* Set `OBJECTIVE_SHOTS = 1000` to optimise with shot noise, as on real hardware, and compare the convergence curve.
* `N_QUBITS = 24-30` with `BACKEND = "gpu"`: every optimizer evaluation is a full state-vector simulation, so the
  saved reports show how GPU execution time scales with the number of qubits and edges.
"""))

write_notebook(OUT, cells, "QAOA - CUDA-Q benchmark")
print("wrote", OUT)
