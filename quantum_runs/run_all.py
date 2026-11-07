"""Run the 8 textbook notebooks at their maximum qubit counts.

    QR_KEYS_XLSX=/path/to/API_KEYS.xlsx python quantum_runs/run_all.py [run label ...]

Without labels every run below starts, all in parallel. Each run writes
quantum_runs/results/<label>/<Notebook>.ipynb (+ exports) and results/<label>/run.log.

IBM runs (cudaq.sample / cudaq.observe sent to a 156-qubit IBM Heron chip):
  every notebook is used unchanged; only Section 3 parameters are overridden.
Simulator runs: Shor, HHL and Noise_Simulation refuse BACKEND="hardware" by design
  (exact custom unitaries / noise models), so they run on this machine's CPU simulator
  at the largest size that fits its memory.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = os.environ.get("QR_PYTHON", sys.executable)

IBM = ['BACKEND="hardware"', 'HARDWARE_PROVIDER="ibm"', 'HARDWARE_EMULATE=False',
       'LIVE_MEMORY_DISPLAY=False', 'EXPORT_DIR="exports"']
CPU = ['BACKEND="cpu"', 'LIVE_MEMORY_DISPLAY=False', 'EXPORT_DIR="exports"']

# label: (notebook, IBM?, account group, parameter overrides)
RUNS = {
    # 156 qubits = every qubit of the IBM Heron chip
    "qft_ibm_156q": ("QFT.ipynb", True, "1-6", ["N_QUBITS=156"] + IBM),
    # one Grover iteration: the optimal count (~pi/4 * 2^78) cannot be built at this size
    "grover_ibm_156q": ("Grover.ipynb", True, "7-12",
                        ["N_QUBITS=156", "GROVER_ITERATIONS=1", "RUN_ITERATION_SWEEP=False"] + IBM),
    # Powell respects the evaluation budget (COBYLA first evaluates once per parameter:
    # ~470 hardware jobs for 468 parameters); every evaluation is one IBM job
    "vqe_ibm_156q": ("VQE.ipynb", True, "13-18",
                     ["N_QUBITS=156", 'OPTIMIZER="Powell"', "MAX_EVALUATIONS=5", "OBJECTIVE_SHOTS=4000"] + IBM),
    # ring graph: 156 edges (the random "example" graph would have ~3,600 at this size)
    "qaoa_ibm_156q": ("QAOA.ipynb", True, "19-24",
                      ["N_QUBITS=156", 'GRAPH="ring"', "MAX_EVALUATIONS=6", "OBJECTIVE_SHOTS=4000"] + IBM),
    # 12 x 13 image = 156 pixels = 156 qubits
    "qml_ibm_156q": ("QML.ipynb", True, "25-33",
                     ["IMAGE_ROWS=12", "IMAGE_COLS=13", "TRAIN_IMAGES=2", "TEST_IMAGES=2",
                      'OPTIMIZER="Powell"', "MAX_ITERATIONS=3", 'TARGET="ibm"']),
    # Simulator sizes are the largest that finish in about an hour on this 4-core machine
    # (each extra qubit doubles the time; the notebooks' own estimates: Shor 24 qubits ~7.9 h,
    # 27 qubits ~5.4 days; HHL 23 qubits ~4.6 days).
    # 119 = 7 x 17 (7 bits) -> 3 x 7 = 21 qubits (estimate ~38 min)
    "shor_sim_21q": ("Shor.ipynb", False, "", ["N_TO_FACTOR=119"] + CPU),
    # 221 = 13 x 17 (8 bits) -> 24 qubits; the 21-qubit run took 4 min (~6x under its estimate)
    "shor_sim_24q": ("Shor.ipynb", False, "", ["N_TO_FACTOR=221"] + CPU),
    # random 256 x 256 system (8 qubits) + 10 clock qubits + 1 ancilla = 19 qubits
    "hhl_sim_19q": ("HHL.ipynb", False, "", ['MATRIX_A="random"', "RANDOM_SIZE=256",
                                              "NUM_CLOCK_QUBITS=10"] + CPU),
    # GHZ + Pauli noise runs on the stabilizer simulator, which scales to large circuits
    "noise_sim_1000q": ("Noise_Simulation.ipynb", False, "", ["N_QUBITS=1000"] + CPU),
}


def launch(label):
    nb, ibm, group, overrides = RUNS[label]
    run_dir = HERE / "results" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QR_ACCOUNTS=group, QR_MODE=os.environ.get("QR_MODE", "ibm"))
    cmd = [PY, str(HERE / "run_notebook.py"), nb, label] + (["--ibm"] if ibm else []) + overrides
    log = open(run_dir / "run.log", "a")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(cmd[2:])}\n")
    log.flush()
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE.parent)


def main(labels):
    labels = labels or list(RUNS)
    procs = {label: launch(label) for label in labels}
    print("started:", ", ".join(labels), flush=True)
    while procs:
        for label, p in list(procs.items()):
            if p.poll() is not None:
                print(f"{time.strftime('%H:%M:%S')} finished {label} (exit {p.returncode})", flush=True)
                del procs[label]
        time.sleep(15)


if __name__ == "__main__":
    main(sys.argv[1:])
