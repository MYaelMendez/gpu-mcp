"""CPU-only tests of validation/CLI boundaries; not GPU execution proof."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'python_gpu_check.py'
spec = importlib.util.spec_from_file_location('python_gpu_check', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PythonGPUCheckTests(unittest.TestCase):
    def test_invalid_workloads_fail_before_optional_imports(self):
        for size, repeats in [(65, 5), (512, 6), (-1, 1), (64.0, 1), (64, True)]:
            with self.subTest(size=size, repeats=repeats):
                with self.assertRaisesRegex(ValueError, 'Unsupported workload'):
                    probe.verify(size, repeats)

    def test_help_needs_no_cuda_or_cupy(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('{64,128,256,512}', result.stdout)
        self.assertIn('existing files are never overwritten', result.stdout)

    def test_cli_rejects_unbounded_size(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--size', '65536'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)


if __name__ == '__main__':
    unittest.main()
