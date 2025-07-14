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
    try:
        print("🚀 Starting MCP server initialization...")
        settings = get_settings()
        print("✓ Settings loaded")

        # Create browser configuration
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False
        )
        print("✓ Browser config created")
    except Exception as e:
        print(f"❌ Error during initial setup: {e}")
        raise

    # Create crawler configuration (stored in lifespan context for tools to use)
    # crawler_config = CrawlerRunConfig(
    #     cache_mode=CacheMode.BYPASS,
    #     verbose=False
    # )

    # Initialize Supabase client
    try:
        from supabase import create_client
        supabase_client = create_client(settings.supabase_url, settings.supabase_service_key)
        print("✓ Supabase client initialized")
    except Exception as e:
        print(f"❌ Error initializing Supabase: {e}")
        raise

    # Initialize core services
    print("Initializing core services...")

    # Initialize database service
    from src.services.database import DatabaseService
    database_service = DatabaseService(settings)
    print("✓ Database service initialized")

    # Initialize embedding service
    from src.services.embedding import EmbeddingService
    embedding_service = EmbeddingService(settings)
    print("✓ Embedding service initialized")

    # Initialize reranking model if enabled
    reranking_model = None
    if settings.use_reranking:
        try:
            print("Loading reranking model...")
            reranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✓ Reranking model loaded")
        except Exception as e:
            print(f"Warning: Could not load reranking model: {e}")

    # Initialize search service (depends on database and embedding services)
    from src.services.rag_search import SearchService
    search_service = SearchService(
        settings=settings,
        database_service=database_service,
        embedding_service=embedding_service,
        reranking_model=reranking_model
    )
    print("✓ Search service initialized")

    # Initialize directory ingestion service
    from src.services.directory_ingestion import DirectoryIngestionService
    directory_ingestion_service = DirectoryIngestionService(supabase_client, settings)
    print("✓ Directory ingestion service initialized")

    # Initialize source management service
    from src.services.source_management import SourceManagementService
    source_management_service = SourceManagementService(supabase_client, settings)
    print("✓ Source management service initialized")

    # Initialize temporary analysis service
    from src.services.temporary_analysis import TemporaryAnalysisService
    temporary_analysis_service = TemporaryAnalysisService(settings)
    print("✓ Temporary analysis service initialized")

    # Initialize Neo4j driver and services if knowledge graph is enabled
    neo4j_driver = None
    hallucination_detector_service = None
    knowledge_graph_service = None
    neo4j_parser_service = None

    # Check if knowledge graph functionality is enabled (matching original behavior)
    knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"

    if knowledge_graph_enabled:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                print("Connecting to Neo4j...")
                neo4j_driver = AsyncGraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password)
                )
                # Test connection
                async with neo4j_driver.session() as session:
                    await session.run("RETURN 1")
                print("✓ Neo4j driver initialized")

                # Initialize Neo4j services
                print("Initializing Neo4j services...")

                from src.services.hallucination_detector import HallucinationDetectorService
                from src.services.knowledge_graph import KnowledgeGraphService
                from src.services.neo4j_parser import Neo4jParserService

                # Create and initialize services
                hallucination_detector_service = HallucinationDetectorService(neo4j_driver, settings)
                await hallucination_detector_service.initialize()
                print("✓ Hallucination detector service initialized")

                knowledge_graph_service = KnowledgeGraphService(neo4j_driver, settings)
                await knowledge_graph_service.initialize()
                print("✓ Knowledge graph service initialized")

                neo4j_parser_service = Neo4jParserService(neo4j_driver, settings)
                await neo4j_parser_service.initialize()
                print("✓ Neo4j parser service initialized")

            except Exception as e:
                print(f"Warning: Could not connect to Neo4j or initialize services: {e}")
                print("Knowledge graph tools will be unavailable")
                neo4j_driver = None
                hallucination_detector_service = None
                knowledge_graph_service = None
                neo4j_parser_service = None
        else:
            print("Neo4j credentials not configured - knowledge graph tools will be unavailable")
    else:
        print("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")

    # Initialize AsyncWebCrawler
    async with AsyncWebCrawler(
        config=browser_config,
        verbose=False
    ) as crawler:
        print("✓ Crawl4AI AsyncWebCrawler initialized")

        # Initialize web crawling service (depends on crawler)
        from src.services.web_crawling import WebCrawlingService
        web_crawling_service = WebCrawlingService(crawler, settings)
        print("✓ Web crawling service initialized")

        # Create context
        context = Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model,
            neo4j_driver=neo4j_driver,
            settings=settings,
            database_service=database_service,
            embedding_service=embedding_service,
            search_service=search_service,
            web_crawling_service=web_crawling_service,
            directory_ingestion_service=directory_ingestion_service,
            source_management_service=source_management_service,
            temporary_analysis_service=temporary_analysis_service,
            hallucination_detector_service=hallucination_detector_service,
            knowledge_graph_service=knowledge_graph_service,
            neo4j_parser_service=neo4j_parser_service
        )

        try:
            yield context
        finally:
            # Cleanup
            print("🧹 Cleaning up resources...")

            # Close Neo4j services
            if hallucination_detector_service:
                try:
                    await hallucination_detector_service.close()
                    print("✓ Hallucination detector service closed")
                except Exception as e:
                    print(f"Error closing hallucination detector service: {e}")

            if knowledge_graph_service:
                try:
                    await knowledge_graph_service.close()
                    print("✓ Knowledge graph service closed")
                except Exception as e:
                    print(f"Error closing knowledge graph service: {e}")

            if neo4j_parser_service:
                try:
                    await neo4j_parser_service.close()
                    print("✓ Neo4j parser service closed")
                except Exception as e:
                    print(f"Error closing neo4j parser service: {e}")

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
    port=os.getenv("PORT", "8051")
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
