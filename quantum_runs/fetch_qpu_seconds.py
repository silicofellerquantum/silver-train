"""Fill results/qpu_seconds.json with the QPU time IBM reports for every logged job.

    QR_KEYS_XLSX=/path/to/API_KEYS.xlsx python quantum_runs/fetch_qpu_seconds.py
"""
import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ibm_shim  # noqa: E402

out_path = HERE / "results" / "qpu_seconds.json"
known = json.loads(out_path.read_text()) if out_path.exists() else {}
accounts = {a.sno: a for a in ibm_shim._accounts().accounts}
for path in glob.glob(str(HERE / "results" / "jobs" / "*.csv")):
    for row in csv.DictReader(open(path)):
        job_id = row["job_id"]
        if row["status"] != "COMPLETED" or job_id in known:
            continue
        acct = accounts[int(row["account_sno"])]
        if acct.client is None:
            from cudaq_ibm import IBMQuantumClient
            acct.client = IBMQuantumClient(api_key=acct._key, crn=acct._crn, timeout=60)
        usage = acct.client._get(f"/jobs/{job_id}/metrics").get("usage") or {}
        known[job_id] = usage.get("quantum_seconds", usage.get("seconds"))
out_path.write_text(json.dumps(known, indent=1))
print(f"{len(known)} jobs, {sum(v or 0 for v in known.values())} QPU seconds in total")
