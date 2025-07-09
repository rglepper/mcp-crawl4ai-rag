#!/usr/bin/env python3
"""
Crawl4AI RAG MCP Server Launcher
This script ensures the correct Python environment and runs the MCP server.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the project directory
    os.chdir(script_dir)
    
    # Path to the virtual environment Python
    venv_python = script_dir / ".venv" / "bin" / "python"
    
    # Path to the MCP server script
    mcp_server = script_dir / "src" / "crawl4ai_mcp.py"
    
    # Check if virtual environment exists
    if not venv_python.exists():
        print(f"Error: Virtual environment not found at {venv_python}")
        print("Please run: uv venv && uv pip install -e .")
        sys.exit(1)
    
    # Check if MCP server script exists
    if not mcp_server.exists():
        print(f"Error: MCP server script not found at {mcp_server}")
        sys.exit(1)
    
    # Run the MCP server with the virtual environment Python
    try:
        subprocess.run([str(venv_python), str(mcp_server)] + sys.argv[1:], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running MCP server: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nMCP server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
