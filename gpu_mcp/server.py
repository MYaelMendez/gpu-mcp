"""server.py — GPU-MCP: expose the local GPU + Rust/WASM hands over MCP.

Canonical scheme (Hermes): ae://glocal-agent  (alias: +ae://cc, home://)
  ae://         = the sovereign agent protocol
  glocal-agent  = the bounded, offline local agent primitive (+ae^glocal):
                  a local brain (ollama) + local CUDA/Rust-WASM hands, exposed
                  as a protocol-native command & control surface for any
                  x86_64-pc-windows + NVIDIA node.

A zero-dependency (stdlib-only) Model Context Protocol server over stdio.
Exposes the hands as MCP tools so any MCP client (Claude Desktop,
Cursor, gemini-cli, Hermes Agent, ...) can drive the local GPU / wasm toolchain:

  CUDA   : probe_gpu, compile_kernel, run_kernel
  Rust   : rust_build_wasm, rust_run_wasm

Run:        python -m gpu_mcp
Self-test:  python -m gpu_mcp --self-test     (MCP handshake over a pipe)

Register (Claude Desktop / Cursor): point an MCP server at
    python -m gpu_mcp
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from .gpu_agent import GPUAgent
from .rust_wasm_tools import build_wasm, run_wasm

_GPU = GPUAgent()

SERVER_INFO = {"name": "gpu-mcp", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"


def _tool(name, description, props, required=None):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": props,
            "required": required or [],
        },
    }


TOOLS = [
    _tool("probe_gpu", "Read live NVIDIA GPU telemetry (nvidia-smi) on the local machine.", {}),
    _tool("compile_kernel", "Compile a CUDA matmul kernel with nvcc (via MSVC vcvars).",
          {"name": {"type": "string", "description": "kernel name", "default": "matmul"}}, ["name"]),
    _tool("run_kernel", "Run a previously compiled CUDA kernel on the local GPU.",
          {"name": {"type": "string", "description": "kernel name", "default": "matmul"}}, ["name"]),
    _tool("rust_build_wasm", "Compile a Rust crate to wasm32-unknown-unknown (local).",
          {"crate_dir": {"type": "string", "description": "path to cargo crate"}}, ["crate_dir"]),
    _tool("rust_run_wasm", "Run a .wasm via wasmtime if present; else report capability.",
          {"wasm_path": {"type": "string", "description": "path to .wasm"}}, ["wasm_path"]),
]


def _call(name, args):
    args = args or {}
    if name == "probe_gpu":
        return _GPU.probe_gpu()
    if name == "compile_kernel":
        return _GPU.compile_kernel(args.get("name", "matmul"))
    if name == "run_kernel":
        return _GPU.run_kernel(args.get("name", "matmul"))
    if name == "rust_build_wasm":
        return build_wasm(args["crate_dir"])
    if name == "rust_run_wasm":
        return run_wasm(args["wasm_path"])
    return {"error": f"unknown tool {name}"}


def _respond(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _handle(msg):
    """Return a response dict (or None for a notification)."""
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        return _respond(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "tools/list":
        return _respond(req_id, {"tools": TOOLS})
    if method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments", {})
        out = _call(name, args)
        return _respond(req_id, {
            "content": [{"type": "text", "text": json.dumps(out, default=str)}],
            "isError": bool(out.get("error") or out.get("ok") is False),
        })
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(msg)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


def self_test():
    """Exercise the MCP handshake over an in-process pipe (no GPU needed for list)."""
    import io
    from contextlib import redirect_stdout

    script = io.StringIO()
    script.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n")
    script.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
    script.seek(0)
    prev_stdin, sys.stdin = sys.stdin, script
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            main()
    finally:
        sys.stdin = prev_stdin
    lines = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
    assert lines[0]["result"]["serverInfo"]["name"] == "gpu-mcp", lines
    names = [t["name"] for t in lines[1]["result"]["tools"]]
    assert names == ["probe_gpu", "compile_kernel", "run_kernel",
                     "rust_build_wasm", "rust_run_wasm"], names
    print("[gpu-mcp self-test] OK — initialize + tools/list returned 5 tools")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
