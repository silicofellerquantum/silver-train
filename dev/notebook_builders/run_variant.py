"""Execute a notebook with overridden Section 3 parameters: run_variant.py NB OUT 'NAME=value' ..."""
import sys, re, time, nbformat
from nbclient import NotebookClient
nb_path, out_path, overrides = sys.argv[1], sys.argv[2], sys.argv[3:]
nb = nbformat.read(nb_path, 4)
cell = next(c for c in nb.cells if c.cell_type == "code" and "Section 3 - INPUT PARAMETERS" in c.source)
for ov in overrides:
    name, value = ov.split("=", 1)
    pattern = re.compile(rf"^{re.escape(name)}\s*=.*?(?=\s+#|$)", re.M)
    if not pattern.search(cell.source):
        raise SystemExit(f"parameter {name} not found")
    cell.source = pattern.sub(f"{name} = {value}", cell.source, count=1)
client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=True,
                        resources={"metadata": {"path": __import__("os").path.dirname(__import__("os").path.abspath(nb_path))}})
t0 = time.time()
try:
    client.execute()
    print(f"executed in {time.time() - t0:.1f}s")
except Exception as exc:                      # e.g. DeadKernelError: keep the partial outputs for debugging
    print(f"EXECUTION ABORTED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
nbformat.write(nb, out_path)
errs = [(i, o.ename, o.evalue[:300]) for i, c in enumerate(nb.cells) if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
print("errors:", errs if errs else "none")
