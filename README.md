# gpu-mcp

**Sovereign local compute over the Model Context Protocol.**

`gpu-mcp` is a zero-dependency (stdlib-only), air-gappable **MCP server** that
exposes your local **NVIDIA GPU (CUDA)** and **Rust → WASM** toolchain to *any*
MCP client — Claude Desktop, Cursor, gemini-cli, or Hermes Agent — as
protocol-native tools. Nothing leaves the machine: the brain is a local model
socket, the hands are local processes.

Canonical scheme (Hermes): `ae://glocal-agent` (alias `+ae://cc`, `home://`).

## Tools

| tool | what it does |
|------|--------------|
| `probe_gpu` | Live `nvidia-smi` telemetry on the local machine |
| `compile_kernel` | Compile a CUDA matmul kernel with `nvcc` (via MSVC `vcvars64`) |
| `run_kernel` | Execute the compiled CUDA kernel on the local GPU, host-side timed |
| `rust_build_wasm` | Compile a Rust crate to `wasm32-unknown-unknown` (local) |
| `rust_run_wasm` | Run a `.wasm` via `wasmtime` if present; else report capability |

## Install

```bash
pip install gpu-mcp
# or, from source
git clone https://github.com/MYaelMendez/gpu-mcp && cd gpu-mcp
pip install -e .
```

## Run the server

```bash
python -m gpu_mcp                 # stdio MCP server (register this with your client)
python -m gpu_mcp --self-test    # MCP handshake self-check (no GPU required)
```

## Register with an MCP client

Point any MCP client at:

```json
{
  "mcpServers": {
    "gpu-mcp": {
      "command": "python",
      "args": ["-m", "gpu_mcp"]
    }
  }
}
```

Works with Claude Desktop, Cursor, gemini-cli, and the Hermes Agent
`ae://glocal-agent` surface.

## Hermes Agent integration

`gpu-mcp` is a first-party primitive of the [Hermes Agent](https://github.com/NousResearch/hermes-agent)
sovereign stack. In `hermes-fork`, the VS Code extension bundles it and the
conductor resolves `ae://glocal-agent` → `mcp://gpu-mcp`. This repo is the
canonical, standalone, pip-installable source of truth.

## Tests

```bash
pytest                              # MCP handshake + offline hands
python -m gpu_mcp --self-test     # quick in-process handshake check
```

GPU/wasm tool tests exercise real `nvidia-smi`, `nvcc`, and `cargo` when
present; they fail loud (not silent) when the local toolchain is missing.

## Python on the RTX — verified optional compute environment

The MCP transport remains stdlib-only. The optional CuPy environment runs actual Python array kernels and cuBLAS locally, without uploading code or introducing an arbitrary remote `run_python` endpoint.

On Windows, create the GPU virtual environment in an **ASCII-only physical path** (the Victus reference uses `C:/gpu`). Preserve `æ://` and Unicode in the language, source and UI; NVRTC's filesystem/include handling is a separate concern. Use `uv pip --python <GPU interpreter>` rather than a `pip` belonging to another Python.

```bash
uv venv --python 3.12 C:/gpu
uv pip install --python C:/gpu/Scripts/python.exe -r requirements-gpu.txt
C:/gpu/Scripts/python.exe tools/python_gpu_check.py --size 256 --repeats 5 --output receipt.json
```

Do not recreate an existing environment blindly. Install only **one** CuPy wheel family: `cupy-cuda13x` for the CUDA 13 reference toolchain. The older CUDA 12 wheel could perform NVRTC arithmetic after moving to an ASCII path, but its cuBLAS import failed because the local toolkit provides CUDA 13 libraries. Changing paths and selecting the matching wheel solved different problems.

The bounded verifier checks NVRTC sum-of-squares and GPU float32 matrix multiplication against NumPy CPU references. It reports CUDA-event and host enqueue/synchronize timings separately, excludes initialization/transfers/validation from those timings, and fails instead of substituting CPU execution. Existing receipt files are never overwritten. Its SHA-256 protects record integrity; it is **not** a signature, independent witness, settlement proof or performance SLA.

Reference run: Python 3.12.13, CuPy 14.2.0, NumPy 2.5.3; RTX 3050 Laptop, compute capability 8.6; 256×256 float32 matmul, five warm repeats, CUDA-event median 0.075584 ms, maximum absolute error 0.0000686646. This is a small validated workload, not a whole-system throughput claim. Only the Windows reference was exercised.

## Why

Cloud agents can't give you your own GPU. `gpu-mcp` is the command-&-control
surface for a **bounded, offline local agent** — your silicon, your weights,
your rules. #opensourceware #hermiphicationisinevitable

---

MIT — © Yael Mendez · [æ.store](https://ae.store)
