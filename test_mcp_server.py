"""
Minimal test MCP server to verify Augment connectivity.
"""
import asyncio
import os
from mcp.server.fastmcp import FastMCP

# Create a minimal MCP server
mcp = FastMCP(
    "test-crawl4ai-rag",
    description="Test MCP server for Crawl4AI RAG - minimal version"
)

@mcp.tool()
def test_tool(message: str = "Hello from MCP!") -> str:
    """A simple test tool to verify MCP connectivity."""
    return f"Test successful: {message}"

@mcp.tool()
def ping() -> str:
    """Simple ping tool to test server connectivity."""
    return "pong"

async def main():
    """Main entry point for the test MCP server."""
    print("🚀 Starting minimal test MCP server...")
    await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())
