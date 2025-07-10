"""
Main entry point for the Crawl4AI MCP server.

This module provides a single entry point that imports and runs the MCP server
from the src.mcp_server module.
"""
import asyncio
from src.mcp_server import main

if __name__ == "__main__":
    asyncio.run(main())
