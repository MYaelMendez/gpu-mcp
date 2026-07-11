"""__main__ — `python -m gpu_mcp` launches the MCP stdio server (or --self-test)."""
import sys

from .server import main, self_test

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
