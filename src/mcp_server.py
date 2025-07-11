"""
MCP server setup and configuration for Crawl4AI RAG.

This module sets up the FastMCP server with all tools and manages the application lifecycle.
"""
import os
import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler, BrowserConfig
from sentence_transformers import CrossEncoder
from neo4j import AsyncGraphDatabase

from src.config import get_settings
from src.models import Crawl4AIContext

# Import all tools
from src.tools.crawling_tools import crawl_single_page, smart_crawl_url, ingest_local_directory
from src.tools.source_management_tools import (
    cleanup_source, analyze_crawl_types, get_knowledge_base_guide, get_available_sources
)
from src.tools.search_tools import perform_rag_query, search_code_examples
from src.tools.knowledge_graph_tools import (
    check_ai_script_hallucinations, query_knowledge_graph, parse_github_repository
)
from src.tools.temporary_analysis_tools import (
    analyze_repository_temporarily, search_temporary_analysis,
    list_temporary_analyses, cleanup_temporary_analysis
)


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle and application dependencies.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing all application dependencies
    """
    settings = get_settings()

    # Create browser configuration
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False
    )

    # Create crawler configuration (stored in lifespan context for tools to use)
    # crawler_config = CrawlerRunConfig(
    #     cache_mode=CacheMode.BYPASS,
    #     verbose=False
    # )

    # Initialize Supabase client
    from supabase import create_client
    supabase_client = create_client(settings.supabase_url, settings.supabase_service_key)
    print("✓ Supabase client initialized")

    # Initialize reranking model if enabled
    reranking_model = None
    if settings.use_reranking:
        try:
            reranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✓ Reranking model loaded")
        except Exception as e:
            print(f"Warning: Could not load reranking model: {e}")

    # Initialize Neo4j driver if knowledge graph is enabled
    neo4j_driver = None
    if settings.enable_knowledge_graph and settings.neo4j_password:
        try:
            neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            # Test connection
            async with neo4j_driver.session() as session:
                await session.run("RETURN 1")
            print("✓ Neo4j driver initialized")
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            neo4j_driver = None

    # Initialize AsyncWebCrawler
    async with AsyncWebCrawler(
        config=browser_config,
        verbose=False
    ) as crawler:
        print("✓ Crawl4AI AsyncWebCrawler initialized")

        # Create context
        context = Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model,
            neo4j_driver=neo4j_driver,
            settings=settings
        )

        try:
            yield context
        finally:
            # Cleanup
            print("🧹 Cleaning up resources...")

            if neo4j_driver:
                try:
                    await neo4j_driver.close()
                    print("✓ Neo4j driver closed")
                except Exception as e:
                    print(f"Error closing Neo4j driver: {e}")


# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051"))
)

# Register crawling tools
mcp.tool()(crawl_single_page)
mcp.tool()(smart_crawl_url)
mcp.tool()(ingest_local_directory)

# Register source management tools
mcp.tool()(cleanup_source)
mcp.tool()(analyze_crawl_types)
mcp.tool()(get_knowledge_base_guide)
mcp.tool()(get_available_sources)

# Register search tools
mcp.tool()(perform_rag_query)
mcp.tool()(search_code_examples)

# Register knowledge graph tools
mcp.tool()(check_ai_script_hallucinations)
mcp.tool()(query_knowledge_graph)
mcp.tool()(parse_github_repository)

# Register temporary analysis tools
mcp.tool()(analyze_repository_temporarily)
mcp.tool()(search_temporary_analysis)
mcp.tool()(list_temporary_analyses)
mcp.tool()(cleanup_temporary_analysis)


async def main():
    """Main entry point for the MCP server."""
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
