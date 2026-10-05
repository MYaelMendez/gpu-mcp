"""Bounded local CuPy verification; no network, arbitrary code or shell execution.

Run with the GPU Python interpreter, not an unrelated pip/Python installation.
This receipt is a self-reported compute check, not a signature or enterprise proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def verify(size: int = 256, repeats: int = 5) -> dict:
    if type(size) is not int or type(repeats) is not int or size not in (64, 128, 256, 512) or repeats not in (1, 3, 5):
        raise ValueError('Unsupported workload: use bounded CLI size/repeats choices')
    import cupy as cp
    import numpy as np

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('No CUDA device: no CPU substitution allowed')
    with cp.cuda.Device(0):
        props = cp.cuda.runtime.getDeviceProperties(0)
        # Exercise the NVRTC path that failed in the Unicode-path environment.
        vector = cp.arange(1024, dtype=cp.float64)
        actual_sum = float(cp.sum(vector * vector).get())
        expected_sum = float(np.sum(np.arange(1024, dtype=np.float64) ** 2))
        np.testing.assert_equal(actual_sum, expected_sum)

        rng = np.random.default_rng(42)
        left = rng.random((size, size), dtype=np.float32)
        right = rng.random((size, size), dtype=np.float32)
        expected = left @ right
        gpu_left, gpu_right = cp.asarray(left), cp.asarray(right)
        result = gpu_left @ gpu_right  # Warm-up excluded from timings.
        cp.cuda.get_current_stream().synchronize()
        gpu_ms, host_ms = [], []
        for _ in range(repeats):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            before = time.perf_counter()
            start.record()
            result = gpu_left @ gpu_right
            end.record()
            end.synchronize()
            host_ms.append((time.perf_counter() - before) * 1000)
            gpu_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
        actual = cp.asnumpy(result)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
        device_name = props['name']
        if isinstance(device_name, bytes):
            device_name = device_name.decode('utf-8')
        receipt = {
            'schema': 'gpu-mcp.python-verification.v1',
            'ok': True,
            'evidence_scope': 'local-compute-self-report-not-signed',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'python_executable': sys.executable,
            'python_version': platform.python_version(),
            'cupy_version': cp.__version__,
            'numpy_version': np.__version__,
            'device': device_name,
            'compute_capability': f"{props['major']}.{props['minor']}",
            'device_memory_bytes': int(props['totalGlobalMem']),
            'cuda_runtime': int(cp.cuda.runtime.runtimeGetVersion()),
            'cuda_driver': int(cp.cuda.runtime.driverGetVersion()),
            'nvrtc_sum_of_squares': {'actual': actual_sum, 'expected': expected_sum, 'verified': True},
            'matmul': {
                'size': size, 'dtype': 'float32', 'seed': 42, 'repeats': repeats,
                'verified_against': 'NumPy CPU matmul', 'rtol': 1e-4, 'atol': 1e-3,
                'max_absolute_error': float(np.max(np.abs(actual - expected))),
                'cuda_event_ms': gpu_ms, 'cuda_event_median_ms': statistics.median(gpu_ms),
                'host_enqueue_and_sync_ms': host_ms,
                'host_enqueue_and_sync_median_ms': statistics.median(host_ms),
                'timing_excludes': ['imports', 'warm-up', 'host/device transfers', 'CPU validation'],
            },
        }
    canonical = json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    receipt['integrity_sha256'] = hashlib.sha256(canonical).hexdigest()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(64, 128, 256, 512), default=256)
    parser.add_argument('--repeats', type=int, choices=(1, 3, 5), default=5)
    parser.add_argument('--output', type=Path, help='New receipt file; existing files are never overwritten')
    args = parser.parse_args()
    try:
        receipt = verify(args.size, args.repeats)
        text = json.dumps(receipt, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as destination:
                destination.write(text + '\n')
        print(text)
        return 0
    except Exception as error:
        print(json.dumps({'ok': False, 'error_type': type(error).__name__, 'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
