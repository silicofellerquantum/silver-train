"""Export a CUDA-Q kernel *with arguments* to OpenQASM 2 (and a Qiskit circuit).

Why this exists
---------------
The cudaq-ibm-backend bridge converts kernels with
``cudaq.translate(cudaq.synthesize(kernel, *args), format="openqasm2")``. In CUDA-Q 0.16
that fails for kernels that take **list** arguments (every textbook notebook does):
``synthesize`` bakes the list into a constant array but leaves the element *loads*
(``cc.load``) and the ``if`` blocks that depend on them, and the OpenQASM 2 emitter
cannot translate those ("'cc.load' op unable to translate op to OpenQASM 2.0").

What this module does (nothing approximate - all values are exact)
------------------------------------------------------------------
1. ``cudaq.synthesize(kernel, *args)``: CUDA-Q bakes every argument into the IR.
2. Every ``cc.load`` that reads element k of a baked constant array is replaced by an
   ``arith.constant`` holding that element (full double precision).
3. CUDA-Q's own MLIR passes then remove the now-constant ``if`` blocks, unroll loops,
   inline sub-kernels and decompose multi-controlled gates into 1- and 2-qubit gates.
4. CUDA-Q's OpenQASM 2 emitter prints the straight-line circuit, which Qiskit parses.
"""

from __future__ import annotations

import re

import cudaq
from cudaq.kernel.kernel_decorator import PyKernelDecorator
from cudaq.mlir.dialects import arith
from cudaq.mlir.ir import (ArrayAttr, F32Type, F64Type, FloatAttr, InsertionPoint, IntegerAttr,
                           Module, StringAttr)
from cudaq.mlir.passmanager import PassManager

# Passes run after the constant folding. Each entry is tried in order; names that do not
# exist in this CUDA-Q build are skipped, so the module also works on nearby versions.
_PIPELINE = [
    "func.func(canonicalize,cse)",
    "lambda-lifting",                            # captured sub-kernels -> named functions
    "indirect-to-direct-calls",                  # cc.call_callable -> func.call
    "apply-op-specialization",                   # sub-kernel .ctrl / .adj variants
    "inline",                                    # inline sub-kernels into the entry kernel
    "func.func(canonicalize,cse)",
    "func.func(cc-loop-unroll,canonicalize,cse)",
    "func.func(sccp,canonicalize,cse)",
    "func.func(multicontrol-decomposition)",
    "func.func(canonicalize,cse)",
    "symbol-dce",
]
# Ops that mean the circuit is not yet straight-line OpenQASM.
_UNRESOLVED = ("cc.loop", "cc.if", "cc.call_callable", "func.call", "cc.load", "cc.create_lambda")
_DYNAMIC_INDEX = -2147483648                 # marker CUDA-Q uses for a non-constant index


def _attr_values(attr):
    out = []
    for a in ArrayAttr(attr):
        try:
            out.append(IntegerAttr(a).value)
        except Exception:
            out.append(FloatAttr(a).value)
    return out


def _indices(op):
    raw = str(op.attributes["rawConstantIndices"])         # e.g. "array<i32: 1>"
    nums = [int(x) for x in re.findall(r"-?\d+", raw.split(":", 1)[1])] if ":" in raw else []
    return nums


def _fold_constant_loads(module):
    """Replace loads of constant-array elements by constants. Returns how many were folded."""
    known = {}                                 # Value -> (values list, offset)
    folded = 0

    def visit(op):
        nonlocal folded
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    o = child.operation
                    name = o.name
                    if name == "cc.const_array":
                        known[o.results[0]] = (_attr_values(o.attributes["constantValues"]), 0)
                    elif name in ("cc.reify_span", "cc.sequence_data", "cc.cast") and len(o.operands) == 1:
                        src = known.get(o.operands[0])
                        if src is not None:
                            known[o.results[0]] = src
                    elif name == "cc.compute_ptr" and len(o.operands) >= 1:
                        src = known.get(o.operands[0])
                        idx = _indices(o)
                        if src is not None and len(idx) == 1:
                            if idx[0] != _DYNAMIC_INDEX and len(o.operands) == 1:
                                known[o.results[0]] = (src[0], src[1] + idx[0])
                            elif idx[0] == _DYNAMIC_INDEX and len(o.operands) == 2:
                                k = _const_int(o.operands[1])        # index given as an SSA constant
                                if k is not None:
                                    known[o.results[0]] = (src[0], src[1] + k)
                    elif name in ("cc.sequence_size", "cc.stdvec_size") and len(o.operands) == 1:
                        src = known.get(o.operands[0])
                        if src is not None:
                            rtype = o.results[0].type
                            with InsertionPoint(o), o.location:
                                const = arith.ConstantOp(rtype, IntegerAttr.get(rtype, len(src[0]) - src[1]))
                            o.results[0].replace_all_uses_with(const.result)
                            o.erase()
                            folded += 1
                            continue
                    elif name == "cc.load" and len(o.operands) == 1:
                        src = known.get(o.operands[0])
                        if src is not None and 0 <= src[1] < len(src[0]):
                            value = src[0][src[1]]
                            rtype = o.results[0].type
                            with InsertionPoint(o), o.location:
                                if F64Type.isinstance(rtype) or F32Type.isinstance(rtype):
                                    attr = FloatAttr.get(rtype, float(value))
                                else:
                                    attr = IntegerAttr.get(rtype, int(value))
                                const = arith.ConstantOp(rtype, attr)
                            o.results[0].replace_all_uses_with(const.result)
                            o.erase()
                            folded += 1
                            continue
                    visit(o)

    visit(module.operation)
    return folded


def _const_int(value):
    owner = value.owner
    try:
        op = owner.operation if hasattr(owner, "operation") else owner
        if op.name == "arith.constant":
            return IntegerAttr(op.attributes["value"]).value
    except Exception:
        pass
    return None


def _const_bool(value):
    owner = value.owner
    try:
        op = owner.operation if hasattr(owner, "operation") else owner
        if op.name == "arith.constant":
            return str(op.attributes["value"]).startswith("true")
    except Exception:
        pass
    return None


def _fold_constant_ifs(module):
    """Inline `cc.if(true) {A} else {B}` as A (or B for false). Returns how many were folded."""
    folded = 0

    def visit(op):
        nonlocal folded
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    o = child.operation
                    if o.name == "cc.if" and len(o.operands) == 1 and len(o.results) == 0:
                        cond = _const_bool(o.operands[0])
                        if cond is not None:
                            taken = o.regions[0] if cond else (o.regions[1] if len(o.regions) > 1 else None)
                            if taken is not None:
                                for blk in taken.blocks:
                                    for inner in list(blk.operations):
                                        if inner.operation.name in ("cc.continue", "cc.yield"):
                                            continue
                                        inner.operation.move_before(o)
                            o.erase()
                            folded += 1
                            continue
                    visit(o)

    visit(module.operation)
    return folded


def _decorators_from(objs, seen):
    """Yield every PyKernelDecorator reachable from processed args / captured variables."""
    for obj in objs:
        dec = obj if isinstance(obj, PyKernelDecorator) else getattr(obj, "decorator", None)
        if isinstance(dec, PyKernelDecorator) and id(dec) not in seen:
            seen.add(id(dec))
            yield dec
            captured = getattr(dec, "captured_variables", None)
            try:
                captured = captured() if callable(captured) else captured
            except Exception:
                captured = None
            captured = captured or {}
            values = captured.values() if isinstance(captured, dict) else list(captured)
            yield from _decorators_from(list(values), seen)
            fn = getattr(dec, "kernelFunction", None)
            glb = getattr(fn, "__globals__", {}) if fn else {}
            names = getattr(getattr(fn, "__code__", None), "co_names", ())
            yield from _decorators_from([glb[n] for n in names if n in glb], seen)
            closure = getattr(fn, "__closure__", None) or ()
            yield from _decorators_from([c.cell_contents for c in closure
                                         if _safe_cell(c) is not None], seen)


def _safe_cell(cell):
    try:
        return cell.cell_contents
    except ValueError:
        return None


def _declarations(module):
    """Names of functions that are declared (no body) in the module."""
    out = []
    for child in module.body.operations:
        o = child.operation
        if o.name == "func.func" and len(o.regions[0].blocks) == 0:
            out.append(str(o.attributes["sym_name"]).strip('"'))
    return out


def _link_subkernels(module, roots):
    """Copy the bodies of called sub-kernels into `module` (synthesize only declares them)."""
    ctx = module.context
    decorators = list(_decorators_from(roots, set()))
    for _ in range(10):
        missing = [n for n in _declarations(module) if n.startswith("__nvqpp__mlirgen__")]
        if not missing:
            return
        progress = False
        for dec in decorators:
            sub = Module.parse(str(dec.qkeModule), ctx)
            for child in list(sub.body.operations):
                o = child.operation
                if o.name != "func.func" or len(o.regions[0].blocks) == 0:
                    continue
                name = str(o.attributes["sym_name"]).strip('"')
                if name in missing:
                    for decl in list(module.body.operations):
                        d = decl.operation
                        if d.name == "func.func" and str(d.attributes["sym_name"]).strip('"') == name:
                            d.erase()
                    with ctx:                        # private: removed by symbol-dce once inlined
                        o.attributes["sym_visibility"] = StringAttr.get("private")
                    module.body.append(o)            # move the definition into the module
                    missing.remove(name)
                    progress = True
        if not progress:
            break
    still = [n for n in _declarations(module) if n.startswith("__nvqpp__mlirgen__")]
    if still:
        raise RuntimeError(f"sub-kernel bodies not found for: {still}")


def _run_pipeline(module, decompose=True):
    ctx = module.context
    for step in _PIPELINE:
        if not decompose and "multicontrol-decomposition" in step:
            continue                             # keeps qubit numbering identical to the kernel
        try:
            PassManager.parse(f"builtin.module({step})", ctx).run(module.operation)
        except Exception:
            pass                                 # pass not available in this build: skip it


def kernel_to_qasm2(kernel, *args, decompose=True):
    """OpenQASM 2 text for kernel(*args). Raises RuntimeError if CUDA-Q cannot emit it.

    decompose=True lets CUDA-Q split gates with 3+ controls into 1-/2-qubit gates; this can
    add helper (ancilla) qubits. Measured results are unaffected (only the kernel's own
    measurements are read). Use decompose=False when qubit numbering must match the kernel
    exactly (expectation values).
    """
    # prepare_call adds the hidden arguments CUDA-Q uses for captured sub-kernels
    # (e.g. QFT's `qft(qubits)` helper); synthesize needs those too.
    processed, _ = kernel.prepare_call(*args)
    synth = cudaq.synthesize(kernel, *processed) if processed else kernel
    source = synth.qkeModule
    module = Module.parse(str(source), source.context)
    _link_subkernels(module, [kernel, *processed])
    for _ in range(8):                           # folding, inlining and unrolling feed each other
        _fold_constant_loads(module)
        _run_pipeline(module, decompose)
        if _fold_constant_ifs(module):
            _run_pipeline(module, decompose)
        text = str(module)
        if not any(op in text for op in _UNRESOLVED):
            break
    else:
        left = sorted({op for op in _UNRESOLVED if op in str(module)})
        raise RuntimeError(f"kernel could not be reduced to a straight-line circuit (left: {left})")
    specialised = PyKernelDecorator(None, module=module, kernelName=kernel.name, decorator=kernel)
    qasm = cudaq.translate(specialised, format="openqasm2")
    if "translation failed" in qasm:
        raise RuntimeError("CUDA-Q could not translate the kernel to OpenQASM 2.0")
    return qasm


def kernel_to_circuit(kernel, *args, measure_all_if_none=True, decompose=True):
    """Qiskit QuantumCircuit for kernel(*args)."""
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit.from_qasm_str(kernel_to_qasm2(kernel, *args, decompose=decompose))
    if measure_all_if_none and circuit.num_clbits == 0:
        circuit.measure_all()
    return circuit
