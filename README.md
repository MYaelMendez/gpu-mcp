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

## Why

Cloud agents can't give you your own GPU. `gpu-mcp` is the command-&-control
surface for a **bounded, offline local agent** — your silicon, your weights,
your rules. #opensourceware #hermiphicationisinevitable

---

MIT — © Yael Mendez · [æ.store](https://ae.store)
