"""Route the textbook notebooks' CUDA-Q calls to IBM Quantum hardware.

The notebooks in the repository root are NOT modified. `run_notebook.py` executes a copy
in memory with one extra first cell that calls `ibm_shim.install()`. After that:

* `cudaq.has_target("ibm")` is True, so the notebooks accept BACKEND="hardware",
  HARDWARE_PROVIDER="ibm" (QML: TARGET="ibm") without code changes.
* `cudaq.set_target("ibm", ...)` keeps a local CPU target active for compiling kernels.
* `cudaq.sample(...)` and `cudaq.observe(...)` are sent to IBM through the
  cudaq-ibm-backend bridge (github.com/brunopacheco1/cudaq-ibm-backend):
  kernel -> OpenQASM 2 -> Qiskit -> ISA circuit for the chip -> IBM Sampler job.

API keys are read in memory from the spreadsheet named by QR_KEYS_XLSX (columns
S.No, Email, API Key, CRN). They are never printed, logged or written to disk.
Accounts are used in S.No order; when one runs out of QPU time the next one is used.

Every job is appended to results/jobs.csv and every result is cached under
results/ibm_raw/<hash>.json, so a re-run never submits the same circuit twice.

Modes (QR_MODE):
  ibm        real hardware (default)
  transpile  no IBM calls: transpile for a 156-qubit fake Heron chip, record circuit
             sizes in results/dryrun.csv and return placeholder counts (dry runs only)
  fake       no IBM calls: noisy local simulation of the chip (small circuits only)
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cudaq

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("QR_RESULTS_DIR", HERE / "results"))
MODE = os.environ.get("QR_MODE", "ibm").strip().lower()
RUN_LABEL = os.environ.get("QR_RUN_LABEL", "unlabelled")
# Notebooks run in parallel; each starts with its own group of accounts (e.g. "1-6"),
# then borrows the others only when its group is used up.
ACCOUNT_GROUP = os.environ.get("QR_ACCOUNTS", "")

RAW_DIR = RESULTS / "ibm_raw"
JOBS_CSV = RESULTS / "jobs" / f"{RUN_LABEL}.csv"          # one log per run (parallel-safe)
DRYRUN_CSV = RESULTS / "dryrun.csv"
STATE_JSON = RESULTS / "accounts_state" / f"{RUN_LABEL}.json"
DEFAULT_SHOTS = int(os.environ.get("QR_DEFAULT_SHOTS", "1000"))
JOB_TIMEOUT_S = float(os.environ.get("QR_JOB_TIMEOUT_S", str(12 * 3600)))
POLL_S = float(os.environ.get("QR_POLL_S", "10"))
FIXED_BACKEND = os.environ.get("QR_IBM_BACKEND") or None
MIN_QUBITS = int(os.environ.get("QR_MIN_BACKEND_QUBITS", "156"))

_QUOTA_WORDS = re.compile(r"usage|limit|quota|exceed|insufficient|allocation|plan", re.I)

_real = {}            # original cudaq functions
_ibm_active = False   # True after cudaq.set_target("ibm")
_machine = None       # HARDWARE_MACHINE passed to set_target("ibm", machine=...)


# =====================================================================================
# Accounts (keys stay in memory only)
# =====================================================================================
class _Account:
    __slots__ = ("sno", "_key", "_crn", "client")

    def __init__(self, sno, key, crn):
        self.sno, self._key, self._crn, self.client = sno, key, crn, None

    def __repr__(self):                                   # never show the key or CRN
        return f"<IBM account S.No {self.sno}>"


class AccountPool:
    def __init__(self):
        path = os.environ.get("QR_KEYS_XLSX")
        if not path or not Path(path).is_file():
            raise RuntimeError("Set QR_KEYS_XLSX to the API-keys spreadsheet (S.No, Email, API Key, CRN).")
        import openpyxl
        ws = openpyxl.load_workbook(path, data_only=True, read_only=True).worksheets[0]
        rows = [r for r in ws.iter_rows(values_only=True) if any(c not in (None, "") for c in r)]
        header = [str(h).strip().lower() for h in rows[0]]
        i_sno, i_key, i_crn = header.index("s.no"), header.index("api key"), header.index("crn")
        self.accounts, seen_crn = [], {}
        for r in rows[1:]:
            acct = _Account(int(r[i_sno]), str(r[i_key]).strip(), str(r[i_crn]).strip())
            self.accounts.append(acct)
            seen_crn.setdefault(acct._crn, []).append(acct.sno)
        # Rows that share a CRN share one QPU allowance: when one is exhausted, all are.
        self.same_instance = {s: group for group in seen_crn.values() for s in group}
        group = _parse_group(ACCOUNT_GROUP)
        self.accounts.sort(key=lambda a: (0 if (not group or a.sno in group) else 1, a.sno))
        self.state = _load_json(STATE_JSON, {"exhausted": [], "bad": []})

    def _save(self):
        _write_json(STATE_JSON, self.state)

    def usable(self):
        skip = set(self.state["exhausted"]) | set(self.state["bad"]) | _shared_exhausted()
        return [a for a in self.accounts if a.sno not in skip]

    def current(self):
        for acct in self.usable():
            if acct.client is None:
                from cudaq_ibm import IBMQuantumClient
                acct.client = IBMQuantumClient(api_key=acct._key, crn=acct._crn, timeout=120.0)
            return acct
        raise RuntimeError("ALL IBM ACCOUNTS ARE EXHAUSTED OR INVALID - see results/accounts_state.json")

    def mark_exhausted(self, acct, why):
        for sno in self.same_instance.get(acct.sno, [acct.sno]):
            if sno not in self.state["exhausted"]:
                self.state["exhausted"].append(sno)
        self.state.setdefault("log", []).append({"time": _now(), "sno": acct.sno, "event": "exhausted",
                                                 "detail": _scrub(why)[:300]})
        self._save()
        print(f"[ibm] account S.No {acct.sno} has no QPU time left -> switching to the next account")

    def mark_bad(self, acct, why):
        if acct.sno not in self.state["bad"]:
            self.state["bad"].append(acct.sno)
        self.state.setdefault("log", []).append({"time": _now(), "sno": acct.sno, "event": "invalid",
                                                 "detail": _scrub(why)[:300]})
        self._save()
        print(f"[ibm] account S.No {acct.sno} could not sign in -> skipped")


def _parse_group(text):
    out = set()
    for part in filter(None, (p.strip() for p in str(text).split(","))):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _shared_exhausted():
    """Accounts that any parallel run has already found exhausted or invalid."""
    seen = set()
    for f in (RESULTS / "accounts_state").glob("*.json"):
        st = _load_json(f, {})
        seen.update(st.get("exhausted", []))
        seen.update(st.get("bad", []))
    return seen


_pool = None


def _accounts():
    global _pool
    if _pool is None:
        _pool = AccountPool()
    return _pool


# =====================================================================================
# Small helpers
# =====================================================================================
def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _scrub(text):
    """Remove anything that looks like a key, token or CRN from an error message."""
    text = str(text)
    text = re.sub(r"crn:v1:[^\s'\"]+", "crn:<hidden>", text)
    text = re.sub(r"[A-Za-z0-9_\-]{40,}", "<hidden>", text)
    return text


def _append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _http_detail(exc):
    resp = getattr(exc, "response", None)
    if resp is None:
        return None, str(exc)
    try:
        body = resp.text
    except Exception:
        body = ""
    return resp.status_code, body


# =====================================================================================
# Circuit preparation
# =====================================================================================
# The bridge's own exporter (cudaq_ibm.kernel_to_circuit) cannot handle the textbook
# kernels in CUDA-Q 0.16 (list arguments; large loops). kernel_trace.py records the kernel
# exactly at any size; for small circuits it is cross-checked against CUDA-Q's compiler
# route (cudaq_export.py) and the run stops if the two ever disagree.
VERIFY_MAX_QUBITS = int(os.environ.get("QR_VERIFY_MAX_QUBITS", "10"))
_verified = set()


def _cross_check(kernel, args, circ):
    if circ.num_qubits > VERIFY_MAX_QUBITS:
        return
    key = (getattr(kernel, "name", id(kernel)), circ.num_qubits, repr(args)[:2000])
    if key in _verified:
        return
    try:
        from cudaq_export import kernel_to_circuit as compiled
        ref = compiled(kernel, *args, measure_all_if_none=False, decompose=False)
    except Exception:
        return                                  # compiler route cannot express it: nothing to compare
    import numpy as np
    from qiskit.quantum_info import Statevector
    a, b = circ.copy(), ref.copy()
    a.remove_final_measurements()
    b.remove_final_measurements()
    if a.num_qubits != b.num_qubits:
        return
    overlap = abs(np.vdot(Statevector(a).data, Statevector(b).data))
    if abs(overlap - 1) > 1e-6:
        raise RuntimeError(f"circuit cross-check failed for {key[0]} (overlap {overlap:.6f})")
    _verified.add(key)


def _circuit_unmeasured(kernel, args):
    from kernel_trace import kernel_to_circuit
    circ = kernel_to_circuit(kernel, *args, measure_all_if_none=False)
    _cross_check(kernel, args, circ)
    if circ.num_clbits:
        circ.remove_final_measurements(inplace=True)
    return circ


def _circuit_measured(kernel, args):
    from kernel_trace import kernel_to_circuit
    circ = kernel_to_circuit(kernel, *args)
    _cross_check(kernel, args, circ)
    return circ


_targets = {}


def _fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def _target_for(acct, backend):
    from cudaq_ibm import target_from_configuration
    if backend not in _targets:
        if acct is None:
            cfg = _fake_backend().configuration().to_dict()
        else:
            cfg = acct.client.backend_configuration(backend)
        _targets[backend] = target_from_configuration(cfg)
    return _targets[backend]


def _isa(circuits, target):
    from cudaq_ibm import to_isa
    return [to_isa(c, target) for c in circuits]


def _stats(isa):
    ops = isa.count_ops()
    twoq = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz"))
    return {"qubits_logical": None, "qubits_used": len({isa.find_bit(q).index for inst in isa.data
                                                        for q in inst.qubits}),
            "depth": isa.depth(), "two_qubit_gates": twoq, "total_gates": sum(ops.values())}


# =====================================================================================
# Backend choice
# =====================================================================================
_chosen_backend = {}


def _pick_backend(acct):
    if FIXED_BACKEND or _machine:
        return FIXED_BACKEND or _machine
    if acct.sno in _chosen_backend:
        return _chosen_backend[acct.sno]
    names = []
    try:
        listing = acct.client._get("/backends")
        items = listing.get("devices", listing.get("backends", listing)) if isinstance(listing, dict) else listing
        for it in items:
            names.append(it if isinstance(it, str) else it.get("name"))
    except Exception as exc:
        print(f"[ibm] could not list backends ({_scrub(exc)[:120]}); using ibm_fez")
        names = ["ibm_fez"]
    best, best_q = None, None
    for name in filter(None, names):
        try:
            cfg = acct.client.backend_configuration(name)
            if int(cfg.get("n_qubits", 0)) < MIN_QUBITS:
                continue
            st = acct.client._get(f"/backends/{name}/status")
            if not st.get("state", True) or str(st.get("status", "active")).lower() not in ("active", "online"):
                continue
            q = int(st.get("length_queue", 0))
            if best is None or q < best_q:
                best, best_q = name, q
        except Exception:
            continue
    best = best or "ibm_fez"
    _chosen_backend[acct.sno] = best
    print(f"[ibm] account S.No {acct.sno}: using backend {best} (queue {best_q})")
    return best


# =====================================================================================
# Job submission with key rotation, caching and the job log
# =====================================================================================
def _job_key(qasm3_list, shots, backend_hint):
    h = hashlib.sha256()
    for q in qasm3_list:
        h.update(q.encode())
    h.update(f"|{shots}|{RUN_LABEL}|{backend_hint}".encode())
    return h.hexdigest()[:24]


def _usage_left(acct):
    try:
        u = acct.client._get("/instances/usage")
        if u.get("usage_limit_reached"):
            return 0.0
        rem = u.get("usage_remaining_seconds")
        return None if rem is None else float(rem)
    except Exception:
        return None


def _run_pubs(circuits, shots, kind):
    """Run logical circuits (measured) on IBM. Returns (list of raw pub data, list of isa circuits)."""
    from qiskit import qasm3

    if MODE == "transpile":
        target = _target_for(None, "fake_fez")
        isas = _isa(circuits, target)
        for c, i in zip(circuits, isas):
            st = _stats(i)
            st["qubits_logical"] = c.num_qubits
            qbytes = len(qasm3.dumps(i).encode())
            _append_csv(DRYRUN_CSV, {"time": _now(), "run": RUN_LABEL, "kind": kind, "shots": shots,
                                     **st, "qasm3_bytes": qbytes})
        return None, isas

    if MODE in ("fake", "ideal"):
        from qiskit_ibm_runtime import SamplerV2
        if MODE == "fake":                                   # noisy copy of the 156-qubit chip
            backend = _fake_backend()
            isas = _isa(circuits, _target_for(None, "fake_fez"))
        else:                                                # noiseless check of the exported circuit
            from qiskit import transpile
            from qiskit_aer import AerSimulator
            backend = AerSimulator(seed_simulator=7)
            isas = [transpile(c, backend, optimization_level=0) for c in circuits]
        res = SamplerV2(mode=backend).run(isas, shots=shots).result()
        data = []
        for pub, isa in zip(res, isas):
            data.append({creg.name: {"samples": [hex(int(b, 2)) for b in
                                                 getattr(pub.data, creg.name).get_bitstrings()]}
                         for creg in isa.cregs})
        return data, isas

    pool = _accounts()
    while True:
        acct = pool.current()
        try:
            backend = _pick_backend(acct)
            target = _target_for(acct, backend)
        except Exception as exc:
            code, body = _http_detail(exc)
            if code in (400, 401, 403) and "iam" in str(getattr(getattr(exc, "response", None), "url", "")):
                pool.mark_bad(acct, f"{code} {body}")
                continue
            if code in (401, 403):
                pool.mark_bad(acct, f"{code} {body}")
                continue
            raise
        isas = _isa(circuits, target)
        qasms = [qasm3.dumps(i) for i in isas]
        key = _job_key(qasms, shots, backend)
        cached = _load_json(RAW_DIR / f"{key}.json", None)
        if cached and cached.get("status") == "COMPLETED":
            return cached["data"], isas

        left = _usage_left(acct)
        if left is not None and left <= 1.0:
            pool.mark_exhausted(acct, f"usage endpoint reports {left} s left")
            continue

        pending = cached.get("job_id") if cached and cached.get("status") == "SUBMITTED" else None
        try:
            if pending and cached.get("sno") == acct.sno:
                job_id = pending
                print(f"[ibm] resuming job {job_id}")
            else:
                job_id = acct.client.submit_sampler(backend, qasms, shots)
                _write_json(RAW_DIR / f"{key}.json", {"status": "SUBMITTED", "job_id": job_id, "sno": acct.sno,
                                                      "backend": backend, "run": RUN_LABEL, "kind": kind})
                print(f"[ibm] submitted job {job_id} ({len(qasms)} circuit(s), {shots} shots, "
                      f"{isas[0].num_qubits}-qubit chip {backend}, account S.No {acct.sno})")
            t0 = time.time()
            acct.client.wait(job_id, poll_seconds=POLL_S, timeout=JOB_TIMEOUT_S)
            result = acct.client.results(job_id)
            job = acct.client._get(f"/jobs/{job_id}")
        except Exception as exc:
            code, body = _http_detail(exc)
            text = f"{code} {body} {exc}"
            job_info = {}
            try:
                if "job_id" in locals():
                    job_info = acct.client._get(f"/jobs/{job_id}")
                    text += " " + json.dumps(job_info.get("state", {}))
            except Exception:
                pass
            _append_csv(JOBS_CSV, _job_row(kind, isas, shots, acct, backend,
                                           locals().get("job_id"), "FAILED", None, _scrub(text)[:200]))
            if code in (401, 403) and not _QUOTA_WORDS.search(text):
                pool.mark_bad(acct, text)
                _drop_cache(key)
                continue
            if _QUOTA_WORDS.search(text) and (code in (400, 402, 403, 429) or "FAILED" in text or code is None):
                pool.mark_exhausted(acct, text)
                _drop_cache(key)
                continue
            raise RuntimeError(f"IBM job failed: {_scrub(text)[:500]}") from None

        data = [pub["data"] for pub in result["results"]]
        qsec = (job.get("usage") or {}).get("quantum_seconds") or (job.get("usage") or {}).get("seconds")
        _write_json(RAW_DIR / f"{key}.json", {"status": "COMPLETED", "job_id": job_id, "sno": acct.sno,
                                              "backend": backend, "run": RUN_LABEL, "kind": kind,
                                              "quantum_seconds": qsec, "wall_s": round(time.time() - t0, 1),
                                              "data": data})
        _append_csv(JOBS_CSV, _job_row(kind, isas, shots, acct, backend, job_id, "COMPLETED", qsec, ""))
        return data, isas


def _drop_cache(key):
    try:
        (RAW_DIR / f"{key}.json").unlink()
    except FileNotFoundError:
        pass


def _job_row(kind, isas, shots, acct, backend, job_id, status, qsec, note):
    st = _stats(isas[0]) if isas else {}
    return {"time_utc": _now(), "run": RUN_LABEL, "kind": kind, "circuits": len(isas), "shots": shots,
            "account_sno": acct.sno, "backend": backend, "job_id": job_id or "", "status": status,
            "quantum_seconds": qsec if qsec is not None else "", "chip_qubits_used": st.get("qubits_used", ""),
            "depth": st.get("depth", ""), "two_qubit_gates": st.get("two_qubit_gates", ""), "note": note}


# =====================================================================================
# Result objects that behave like CUDA-Q's
# =====================================================================================
class IBMSampleResult(dict):
    """dict bitstring -> count, plus the cudaq.SampleResult methods the notebooks use."""

    def count(self, bits):
        return self.get(bits, 0)

    def most_probable(self):
        return max(self, key=self.get)

    def probability(self, bits):
        total = sum(self.values())
        return self.get(bits, 0) / total if total else 0.0

    def get_total_shots(self):
        return sum(self.values())

    def dump(self):
        print(dict(self))

    def __str__(self):
        return "{ " + " ".join(f"{k}:{v}" for k, v in sorted(self.items(), key=lambda kv: -kv[1])) + " }"


class IBMObserveResult:
    def __init__(self, value, counts_by_group):
        self._value, self._counts = value, counts_by_group

    def expectation(self, *_):
        return self._value

    def counts(self, *_):
        return self._counts


def _placeholder_counts(isa, shots):
    width = sum(c.size for c in isa.cregs)
    return {"0" * width: shots}


# =====================================================================================
# Replacements for cudaq.sample / cudaq.observe
# =====================================================================================
def _ibm_sample(kernel, *args, shots_count=1000, **kwargs):
    from cudaq_ibm import samples_to_counts
    circ = _circuit_measured(kernel, args)
    data, isas = _run_pubs([circ], shots_count, "sample")
    if data is None:                                         # transpile-only dry run
        return IBMSampleResult(_placeholder_counts(circ, shots_count))
    regs = [(c.name, c.size) for c in isas[0].cregs]
    return IBMSampleResult(samples_to_counts(data[0], regs))


def _pauli_terms(spin_op, n_qubits):
    """[(coefficient, pauli string of length n_qubits)] for a CUDA-Q spin operator."""
    terms = []
    if not isinstance(spin_op, cudaq.SpinOperator):        # a single term, e.g. spin.z(q)
        spin_op = cudaq.SpinOperator(spin_op)
    for term in spin_op:
        coef = term.evaluate_coefficient() if hasattr(term, "evaluate_coefficient") else term.get_coefficient()
        word = term.get_pauli_word(n_qubits)
        terms.append((complex(coef).real, word))
    return terms


def _group_terms(terms):
    """Greedy qubit-wise-commuting groups: each group is measured with one circuit."""
    groups = []                                              # [(basis list, [term index])]
    for idx, (_, word) in enumerate(terms):
        if set(word) <= {"I"}:
            continue
        for basis, members in groups:
            if all(b == "I" or w == "I" or b == w for b, w in zip(basis, word)):
                for q, w in enumerate(word):
                    if w != "I":
                        basis[q] = w
                members.append(idx)
                break
        else:
            groups.append((list(word), [idx]))
    return groups


def _ibm_observe(kernel, spin_op, *args, shots_count=-1, **kwargs):
    from qiskit import ClassicalRegister
    from cudaq_ibm import samples_to_counts
    shots = shots_count if shots_count and shots_count > 0 else DEFAULT_SHOTS
    base = _circuit_unmeasured(kernel, args)
    n = base.num_qubits
    terms = _pauli_terms(spin_op, n)
    groups = _group_terms(terms)
    circuits = []
    for basis, _ in groups:
        c = base.copy()
        for q, b in enumerate(basis):
            if b == "X":
                c.h(q)
            elif b == "Y":
                c.sdg(q)
                c.h(q)
        creg = ClassicalRegister(n, "meas")
        c.add_register(creg)
        c.measure(range(n), creg)
        circuits.append(c)
    value = sum(coef for coef, w in terms if set(w) <= {"I"})
    if not circuits:
        return IBMObserveResult(value, [])
    data, isas = _run_pubs(circuits, shots, "observe")
    if data is None:                                         # transpile-only dry run
        return IBMObserveResult(value, [])
    all_counts = []
    for (basis, members), pub, isa in zip(groups, data, isas):
        counts = samples_to_counts(pub, [(c.name, c.size) for c in isa.cregs])
        all_counts.append(counts)
        total = sum(counts.values())
        for idx in members:
            coef, word = terms[idx]
            support = [q for q, w in enumerate(word) if w != "I"]
            acc = 0
            for bits, cnt in counts.items():              # bits: qubit 0 first (CUDA-Q order)
                parity = sum(bits[q] == "1" for q in support) % 2
                acc += cnt * (-1 if parity else 1)
            value += coef * acc / total
    return IBMObserveResult(value, all_counts)


# =====================================================================================
# Install the hooks
# =====================================================================================
class _TargetView:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, item):
        return getattr(self._real, item)

    @property
    def name(self):
        return "ibm"


def install():
    if _real:
        return
    for fn in ("has_target", "set_target", "get_target", "sample", "observe", "get_state"):
        _real[fn] = getattr(cudaq, fn)

    def has_target(name, *a, **k):
        return str(name).strip().lower() == "ibm" or _real["has_target"](name, *a, **k)

    def set_target(name, *a, **k):
        global _ibm_active, _machine
        if str(getattr(name, "name", name)).strip().lower() == "ibm":
            _ibm_active, _machine = True, k.get("machine")
            return _real["set_target"]("qpp-cpu")            # local target only compiles the kernels
        _ibm_active = False
        return _real["set_target"](name, *a, **k)

    def get_target(*a, **k):
        t = _real["get_target"](*a, **k)
        return _TargetView(t) if _ibm_active and not a else t

    def sample(kernel, *args, **kwargs):
        if _ibm_active:
            return _ibm_sample(kernel, *args, **kwargs)
        return _real["sample"](kernel, *args, **kwargs)

    def observe(kernel, spin_op, *args, **kwargs):
        if _ibm_active:
            return _ibm_observe(kernel, spin_op, *args, **kwargs)
        return _real["observe"](kernel, spin_op, *args, **kwargs)

    def get_state(*args, **kwargs):
        if _ibm_active:
            raise RuntimeError("get_state() is a simulator feature and is not available on IBM hardware.")
        return _real["get_state"](*args, **kwargs)

    cudaq.has_target, cudaq.set_target, cudaq.get_target = has_target, set_target, get_target
    cudaq.sample, cudaq.observe, cudaq.get_state = sample, observe, get_state
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[ibm] IBM routing installed (mode={MODE}, run={RUN_LABEL})")
