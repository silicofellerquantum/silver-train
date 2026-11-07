"""Execute one textbook notebook with overridden Section 3 parameters, optionally on IBM.

    python quantum_runs/run_notebook.py NOTEBOOK LABEL [--ibm] [NAME=value ...]

* The notebook in the repository root is only READ; the executed copy is written to
  quantum_runs/results/<LABEL>/<NOTEBOOK>.ipynb and its exports go to the same folder.
* Parameter overrides work like dev/notebook_builders/run_variant.py.
* --ibm adds one first cell that installs ibm_shim (routes cudaq.sample/observe to IBM).
"""
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    nb_name, label = argv[0], argv[1]
    use_ibm = "--ibm" in argv[2:]
    overrides = [a for a in argv[2:] if a != "--ibm"]

    nb = nbformat.read(REPO / nb_name, 4)
    for c in nb.cells:                                  # the repo notebooks are saved with outputs:
        if c.cell_type == "code":                       # clear them so a cell that never ran here
            c.outputs, c.execution_count = [], None     # cannot show a stale result
    cell = next(c for c in nb.cells if c.cell_type == "code" and "Section 3 - INPUT PARAMETERS" in c.source)
    for ov in overrides:
        name, value = ov.split("=", 1)
        pattern = re.compile(rf"^{re.escape(name)}\s*=.*?(?=\s+#|$)", re.M)
        if not pattern.search(cell.source):
            raise SystemExit(f"parameter {name} not found in {nb_name}")
        cell.source = pattern.sub(lambda _m: f"{name} = {value}", cell.source, count=1)

    results_dir = Path(os.environ.get("QR_RESULTS_DIR", HERE / "results"))
    run_dir = results_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QR_RUN_LABEL"] = label
    os.environ["QR_RESULTS_DIR"] = str(results_dir)

    header = nbformat.v4.new_markdown_cell(
        f"**quantum_runs** - executed copy of `{nb_name}` (run `{label}`).\n\n"
        f"Parameter overrides: `{' '.join(overrides) or 'none'}`"
        + ("\n\nThe next cell routes `cudaq.sample` / `cudaq.observe` to IBM Quantum hardware "
           "(see `quantum_runs/ibm_shim.py`). The notebook's own code is unchanged." if use_ibm else ""))
    new_cells = [header]
    if use_ibm:
        new_cells.append(nbformat.v4.new_code_cell(
            f"import sys; sys.path.insert(0, {str(HERE)!r})\nimport ibm_shim; ibm_shim.install()"))
    nb.cells = new_cells + nb.cells

    client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": str(run_dir)}})
    t0 = time.time()
    aborted = None
    try:
        client.execute()
        print(f"executed in {time.time() - t0:.1f}s")
    except Exception as exc:                           # keep partial outputs for debugging
        aborted = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"EXECUTION ABORTED after {time.time() - t0:.1f}s: {aborted}")
    out = run_dir / nb_name
    nbformat.write(nb, out)
    errs = [(i, o.ename, o.evalue[:300]) for i, c in enumerate(nb.cells) if c.cell_type == "code"
            for o in c.get("outputs", []) if o.output_type == "error"]
    if aborted:
        errs.append((-1, "ExecutionAborted", aborted))
    # The notebooks catch execution errors themselves and print "EXECUTION FAILED: ..."
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            text = o.get("text", "") if o.output_type == "stream" else ""
            for line in text.splitlines():
                if "EXECUTION FAILED" in line or line.startswith("No report: execution failed"):
                    errs.append((i, "NotebookReported", line.strip()[:300]))
    print(f"saved {out}")
    print("errors:", errs if errs else "none")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
