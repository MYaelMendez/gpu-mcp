"""gpu-mcp — sovereign local compute over the Model Context Protocol.

Expose the local NVIDIA GPU (CUDA) + Rust/WASM hands to any MCP client
(Claude Desktop, Cursor, gemini-cli, Hermes Agent, ...) as stdlib-only
JSON-RPC tools. Fully air-gappable: no network, no third-party deps.

    python -m gpu_mcp            # run the stdio MCP server
    python -m gpu_mcp --self-test   # MCP handshake self-check

Canonical scheme (Hermes): ae://glocal-agent  (alias +ae://cc, home://)
"""
from __future__ import annotations

from .server import main, TOOLS, _call
from .gpu_agent import GPUAgent
from .host_loop import LocalHostLoop
from .rust_wasm_tools import build_wasm, run_wasm, toolset

__version__ = "0.1.0"
__all__ = ["main", "TOOLS", "_call", "GPUAgent", "LocalHostLoop",
           "build_wasm", "run_wasm", "toolset"]
