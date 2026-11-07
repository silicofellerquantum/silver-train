"""Record a CUDA-Q kernel call as a Qiskit circuit by running the kernel's Python body.

Why: CUDA-Q 0.16 can only print OpenQASM for kernels it can fully unroll, and its loop
unroller has a fixed size limit, so the textbook kernels fail at large qubit counts
(e.g. QFT on 156 qubits). The textbook kernels are plain Python (loops, `range`, list
indexing, gate calls), so we execute that Python with recording stand-ins for the gate
functions. Every angle is the exact Python float the kernel computes - nothing is
rounded or approximated - and there is no size limit.

Supported: qvector/qview (indexing, slices, size/front/back), broadcast single-qubit
gates, h x y z s t (+ .adj), rx ry rz r1 u3 (+ .adj), swap, .ctrl with any number of
controls, mz, and calls to other @cudaq.kernel functions (sub-kernels).
Anything else raises NotImplementedError, so a circuit is never silently incomplete.

`cudaq_export.py` (CUDA-Q's own compiler route) is used to cross-check small circuits.
"""

from __future__ import annotations

import math
import types

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (HGate, PhaseGate, RXGate, RYGate, RZGate, SdgGate, SGate,
                                    SwapGate, TdgGate, TGate, UGate, XGate, YGate, ZGate)


class _Qubit:
    __slots__ = ("idx",)

    def __init__(self, idx):
        self.idx = idx


class _QVec:
    def __init__(self, qubits):
        self._q = list(qubits)

    def size(self):
        return len(self._q)

    def __len__(self):
        return len(self._q)

    def __iter__(self):
        return iter(self._q)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _QVec(self._q[key])
        return self._q[int(key)]

    def front(self, n=None):
        return self._q[0] if n is None else _QVec(self._q[:n])

    def back(self, n=None):
        return self._q[-1] if n is None else _QVec(self._q[-n:])

    def slice(self, start, count):
        return _QVec(self._q[start:start + count])


def _flatten(items):
    out = []
    for it in items:
        if isinstance(it, _Qubit):
            out.append(it.idx)
        elif isinstance(it, (_QVec, list, tuple)):
            out.extend(_flatten(list(it)))
        else:
            raise NotImplementedError(f"unsupported qubit argument {type(it).__name__}")
    return out


class _Recorder:
    def __init__(self):
        self.n = 0
        self.ops = []            # (gate, [qubits])
        self.measured = []       # qubit indices in measurement order

    def alloc(self, n):
        if not isinstance(n, int) or isinstance(n, bool):
            try:
                n = int(n)
            except Exception:
                raise NotImplementedError("qvector initialised from a state is not supported") from None
        qs = [_Qubit(self.n + i) for i in range(n)]
        self.n += n
        return _QVec(qs)

    def build(self):
        qr = QuantumRegister(self.n, "q")
        circ = QuantumCircuit(qr)
        for gate, qubits in self.ops:
            circ.append(gate, [qr[i] for i in qubits])
        if self.measured:
            cr = ClassicalRegister(len(self.measured), "meas")
            circ.add_register(cr)
            for k, q in enumerate(self.measured):
                circ.measure(qr[q], cr[k])
        return circ


_ADJ = {"s": SdgGate, "t": TdgGate, "sdg": SGate, "tdg": TGate}


class _Gate:
    """One gate family, e.g. `x`, with `.ctrl` and `.adj` variants."""

    def __init__(self, rec, name, n_params, adjoint=False):
        self.rec, self.name, self.n_params, self.adjoint = rec, name, n_params, adjoint

    def _base(self, params):
        name, adj = self.name, self.adjoint
        if name in ("s", "t"):
            return (_ADJ[name] if adj else {"s": SGate, "t": TGate}[name])()
        if self.n_params:
            params = [float(p) for p in params]
            if adj:
                if name == "u3":
                    theta, phi, lam = params
                    params = [-theta, -lam, -phi]
                else:
                    params = [-p for p in params]
        return {"h": HGate, "x": XGate, "y": YGate, "z": ZGate, "swap": SwapGate,
                "rx": RXGate, "ry": RYGate, "rz": RZGate, "r1": PhaseGate, "u3": UGate}[name](*params)

    @property
    def adj(self):
        return _Gate(self.rec, self.name, self.n_params, not self.adjoint)

    def __call__(self, *args):
        params, qargs = args[:self.n_params], args[self.n_params:]
        gate = self._base(params)
        if self.name == "swap":
            self.rec.ops.append((gate, _flatten(qargs)))
            return
        for q in _flatten(qargs):                       # broadcast over registers
            self.rec.ops.append((gate, [q]))

    def ctrl(self, *args):
        params, qargs = args[:self.n_params], args[self.n_params:]
        gate = self._base(params)
        n_targets = 2 if self.name == "swap" else 1
        qubits = _flatten(qargs)
        controls, targets = qubits[:-n_targets], qubits[-n_targets:]
        if not controls:
            self.rec.ops.append((gate, targets))
        else:
            self.rec.ops.append((gate.control(len(controls)), controls + targets))


class _FakeCudaq(types.SimpleNamespace):
    pass


def _gate_table(rec):
    table = {name: _Gate(rec, name, 0) for name in ("h", "x", "y", "z", "s", "t", "swap")}
    table.update({name: _Gate(rec, name, 1) for name in ("rx", "ry", "rz", "r1")})
    table["u3"] = _Gate(rec, "u3", 3)
    table["sdg"] = _Gate(rec, "s", 0, adjoint=True)
    table["tdg"] = _Gate(rec, "t", 0, adjoint=True)

    def cx(c, t):
        table["x"].ctrl(c, t)

    def cz(c, t):
        table["z"].ctrl(c, t)

    def mz(*qargs, **_):
        rec.measured.extend(_flatten(qargs))

    def unsupported(name):
        def f(*a, **k):
            raise NotImplementedError(f"`{name}` is not supported by the kernel recorder")
        return f

    table.update({"cx": cx, "cz": cz, "mz": mz, "mx": unsupported("mx"), "my": unsupported("my"),
                  "exp_pauli": unsupported("exp_pauli"), "reset": unsupported("reset")})
    return table


def _is_kernel(obj):
    return type(obj).__name__ == "PyKernelDecorator" and hasattr(obj, "kernelFunction")


def _rebind(fn, rec, table, fake_cudaq, cache):
    """Copy of `fn` whose gate names, `cudaq` and sub-kernels point at the recorder."""
    if id(fn) in cache:
        return cache[id(fn)]
    glb = dict(fn.__globals__)
    glb.update(table)
    glb["cudaq"] = fake_cudaq
    for name in fn.__code__.co_names:
        if _is_kernel(glb.get(name)):
            glb[name] = _rebind(glb[name].kernelFunction, rec, table, fake_cudaq, cache)
    closure = None
    if fn.__closure__:
        cells = []
        for var, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                value = cell.cell_contents
            except ValueError:
                cells.append(cell)
                continue
            if _is_kernel(value):
                cells.append(types.CellType(_rebind(value.kernelFunction, rec, table, fake_cudaq, cache)))
            elif var in table or var == "cudaq":
                cells.append(types.CellType(table.get(var, fake_cudaq)))
            else:
                cells.append(cell)
        closure = tuple(cells)
    new = types.FunctionType(fn.__code__, glb, fn.__name__, fn.__defaults__, closure)
    cache[id(fn)] = new
    return new


def kernel_to_circuit(kernel, *args, measure_all_if_none=True):
    """Qiskit circuit for kernel(*args), recorded by running the kernel's Python body."""
    if not _is_kernel(kernel):
        raise NotImplementedError("only @cudaq.kernel functions can be recorded")
    rec = _Recorder()
    table = _gate_table(rec)
    fake_cudaq = _FakeCudaq(qvector=rec.alloc, qview=_QVec, qubit=lambda: rec.alloc(1)[0],
                            State=object, pi=math.pi)
    fn = _rebind(kernel.kernelFunction, rec, table, fake_cudaq, {})
    fn(*args)
    if measure_all_if_none and not rec.measured:
        rec.measured = list(range(rec.n))                # cudaq.sample measures everything
    return rec.build()
