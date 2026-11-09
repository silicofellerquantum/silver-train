"""Build quantum_runs/results/summary.md from the executed notebooks and the job logs."""
import csv
import glob
import json
import re
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ORDER = ["qft_ibm_156q", "qft_ibm_63q", "grover_ibm_156q", "vqe_ibm_156q", "qaoa_ibm_156q", "qml_ibm_156q",
         "shor_sim_21q", "shor_sim_24q", "hhl_sim_19q", "noise_sim_1000q"]
KEYS = ("Algorithm", "Number of qubits", "CUDA-Q target", "Number of shots", "Expected result",
        "Actual result", "Error", "Total wall time", "Gates in", "VQE energy", "Test accuracy",
        "Best cut", "Factors", "QCNN", "Training finished", "test accuracy")


def report_lines(nb_path):
    nb = nbformat.read(nb_path, 4)
    lines, failed = [], []
    for c in nb.cells:
        for o in c.get("outputs", []):
            text = o.get("text", "") if o.get("output_type") == "stream" else ""
            for line in text.splitlines():
                s = line.strip()
                if "EXECUTION FAILED" in s:
                    failed.append(s)
                if any(s.startswith(k) or k in s[:40] for k in KEYS) and len(s) < 220:
                    lines.append(s)
            if o.get("output_type") == "error":
                failed.append(f"{o.get('ename')}: {o.get('evalue', '')[:200]}")
    seen, out = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out, failed


def jobs_for(label):
    path = RESULTS / "jobs" / f"{label}.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path)))
    usage = json.loads((RESULTS / "qpu_seconds.json").read_text()) if (RESULTS / "qpu_seconds.json").exists() else {}
    for r in rows:
        if not r.get("quantum_seconds") and r["job_id"] in usage:
            r["quantum_seconds"] = str(usage[r["job_id"]])
    # A job logged before the pause and reused after it appears twice: keep the later row.
    latest = {}
    for r in rows:
        if r["status"] != "COMPLETED" or not r["job_id"]:
            latest[id(r)] = r
        else:
            latest[r["job_id"]] = r
    return list(latest.values())


# Headline result of each run, read from its notebook report (kept short for the table).
RESULT = {
    "qft_ibm_156q": "4000 shots -> 4000 distinct outcomes (ideal: 8 peaks); notebook analysis stops above "
                    "63 qubits (OverflowError, 64-bit integers) - see qft_ibm_63q",
    "qft_ibm_63q": "0% of shots on the 8 expected peaks (ideal 100%)",
    "grover_ibm_156q": "1 iteration: P(marked) 0/2000, theory ~0",
    "vqe_ibm_156q": "E = -43.65 vs exact E0 = -165.04 (relative error 74%)",
    "qaoa_ibm_156q": "<C> = 71.98 on a 156-node ring; exact max cut 156 (notebook's local search: 140) "
                     "-> ratio 0.46, below random (0.5)",
    "qml_ibm_156q": "test accuracy 50% (= random guessing), training 100% on 2 images",
    "shor_sim_21q": "119 = 7 x 17; 94% of shots yield the factors",
    "shor_sim_24q": "see below",
    "hhl_sim_19q": "solution fidelity 0.9952",
    "noise_sim_1000q": "GHZ, depolarizing p = 0.01: Hellinger fidelity 0.0003",
}
# Runs whose hardware result is complete although a later notebook section stopped.
ANALYSIS_LIMIT = {"qft_ibm_156q"}


def main():
    md = ["# quantum_runs - results", "",
          "All 8 textbook notebooks run at their maximum qubit count. The notebooks in the repository",
          "root were not modified: each run executes a copy with Section 3 parameter overrides only.",
          "At the full 156-qubit width every hardware result is dominated by hardware noise - the",
          "expected outcome of a scaling run on current devices.",
          "", "| Run | Qubits | Where | Status | IBM jobs | QPU s | Result |", "|---|---|---|---|---|---|---|"]
    details = []
    for label in ORDER + sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and p.name not in ORDER
                                and p.name not in ("jobs", "accounts_state", "ibm_raw")):
        nbs = glob.glob(str(RESULTS / label / "*.ipynb"))
        if not nbs:
            continue
        lines, failed = report_lines(nbs[0])
        jobs = jobs_for(label)
        done = [j for j in jobs if j["status"] == "COMPLETED"]
        qsec = sum(float(j["quantum_seconds"]) for j in done if j.get("quantum_seconds"))
        where = "IBM " + ", ".join(sorted({j["backend"] for j in done})) if jobs else "CPU simulator"
        status = ("hardware result complete; analysis limit" if label in ANALYSIS_LIMIT
                  else "failed" if failed else "completed")
        qubits = re.search(r"_(\d+)q$", label).group(1) if re.search(r"_(\d+)q$", label) else ""
        md.append(f"| `{label}` | {qubits} | {where} | {status} | {len(done)} | {qsec:.0f} | "
                  f"{RESULT.get(label, '')} |")
        details += ["", f"## {label}", "", f"Notebook: `{Path(nbs[0]).relative_to(HERE)}`", ""]
        details += [f"    {line}" for line in lines] or ["    (no report lines)"]
        if failed:
            details += ["", "Errors:", ""] + [f"    {f}" for f in failed]
        if jobs:
            details += ["", "| time (UTC) | account S.No | backend | job id | status | QPU s | chip qubits | depth | 2q gates |",
                        "|---|---|---|---|---|---|---|---|---|"]
            details += [f"| {j['time_utc']} | {j['account_sno']} | {j['backend']} | `{j['job_id']}` | {j['status']} | "
                        f"{j['quantum_seconds']} | {j['chip_qubits_used']} | {j['depth']} | {j['two_qubit_gates']} |"
                        for j in jobs]
    (RESULTS / "summary.md").write_text("\n".join(md + details) + "\n")
    print((RESULTS / "summary.md").read_text()[:3000])


if __name__ == "__main__":
    main()
