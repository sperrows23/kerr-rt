"""CUDA smoke test — confirms Taichi can JIT and run a kernel on the GPU.

Per CLAUDE.md the backend is LOCKED to ``ti.init(arch=ti.cuda)`` (never
``ti.gpu``, which can silently fall back to CPU/Metal). On the RTX 5060
(sm_120 / Blackwell) this exercises the CUDA JIT path for Taichi 1.7.4.

Run standalone:  python tests/cuda_smoke_test.py
"""

from __future__ import annotations

import sys

import numpy as np
import taichi as ti


def main() -> int:
    # LOCKED backend — ti.cuda, never ti.gpu (see CLAUDE.md).
    ti.init(arch=ti.cuda)

    n = 1 << 20
    field = ti.field(dtype=ti.f32, shape=n)

    @ti.kernel
    def fill():
        for i in field:
            field[i] = ti.sqrt(ti.cast(i, ti.f32))

    fill()
    host = field.to_numpy()

    # Spot-check a few entries against the CPU reference.
    idx = np.array([0, 1, 2, n // 2, n - 1])
    expected = np.sqrt(idx.astype(np.float32))
    got = host[idx]
    max_err = float(np.max(np.abs(got - expected)))

    print(f"Taichi version : {ti.__version__}")
    print(f"Backend        : {ti.cfg.arch}")
    print(f"Kernel elements: {n}")
    print(f"Sample idx     : {idx.tolist()}")
    print(f"Sample got     : {got.tolist()}")
    print(f"Sample expected: {expected.tolist()}")
    print(f"Max abs error  : {max_err:.3e}")

    if str(ti.cfg.arch) != "Arch.cuda":
        print("FAIL: backend is not CUDA — refusing to proceed (see CLAUDE.md).")
        return 1
    if max_err > 1e-3:
        print("FAIL: kernel result diverges from CPU reference.")
        return 1

    print("PASS: CUDA backend active and kernel results correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
