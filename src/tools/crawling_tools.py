"""
Crawling tools for the Crawl4AI MCP server.

This module provides MCP tool wrappers for web crawling operations including
single page crawling, smart URL crawling, and local directory ingestion.
"""
import json

from mcp.server.fastmcp import Context

from src.models import CrawlRequest
from src.services.web_crawling import WebCrawlingService


async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        JSON string with crawl results
    """
    try:
        # Get web crawling service from context
        service = ctx.request_context.lifespan_context.web_crawling_service

        # Create request
        request = CrawlRequest(
            url=url,
            max_depth=1,
            max_concurrent=1,
            chunk_size=5000
        )

        # Process request
        result = await service.process_crawl_request(request)

        # Convert to dictionary for JSON serialization
        result_dict = {
            "success": result.success,
            "url": result.url,
            "crawl_type": result.crawl_type,
            "pages_crawled": result.pages_crawled,
            "chunks_stored": result.chunks_stored,
            "code_examples_stored": result.code_examples_stored
        }

        if not result.success:
            result_dict["error"] = result.error

        return json.dumps(result_dict, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)


async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.

    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth

    All crawled content is chunked and stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL to crawl
        max_depth: Maximum recursion depth for internal links (default: 3)
        max_concurrent: Maximum concurrent browser sessions (default: 10)
        chunk_size: Maximum chunk size in characters (default: 5000)

    Returns:
        JSON string with crawl results
    """
    try:
        # Get web crawling service from context
        service = ctx.request_context.lifespan_context.web_crawling_service

        # Create request
        request = CrawlRequest(
            url=url,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            chunk_size=chunk_size
        )

        # Process request
        result = await service.process_crawl_request(request)

        # Convert to dictionary for JSON serialization
        result_dict = {
            "success": result.success,
            "url": result.url,
            "crawl_type": result.crawl_type,
            "pages_crawled": result.pages_crawled,
            "chunks_stored": result.chunks_stored,
            "code_examples_stored": result.code_examples_stored
        }

        if not result.success:
            result_dict["error"] = result.error

        return json.dumps(result_dict, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)


async def ingest_local_directory(ctx: Context, directory_path: str, source_name: str = None, file_extensions: str = ".md,.txt,.markdown", recursive: bool = True, chunk_size: int = 5000) -> str:
    """
    Ingest markdown and text files from a local filesystem directory into the knowledge base.

    This tool reads local markdown/text files, processes them with the same pipeline as web crawling
    (chunking, embedding, code extraction), and stores them in Supabase for semantic search.

    Args:
        ctx: The MCP server provided context
        directory_path: Path to the directory containing markdown/text files
        source_name: Optional custom source name (default: directory name)
        file_extensions: Comma-separated list of file extensions to process (default: .md,.txt,.markdown)
        recursive: Whether to recursively process subdirectories (default: True)
        chunk_size: Maximum chunk size in characters (default: 5000)

    Returns:
        JSON string with ingestion results
    """
    try:
        # Get directory ingestion service from context
        service = ctx.request_context.lifespan_context.directory_ingestion_service

        # Process directory
        extensions = file_extensions.split(',')
        result = await service.ingest_directory(
            directory_path=directory_path,
            source_name=source_name,
            file_extensions=extensions,
            recursive=recursive,
            chunk_size=chunk_size
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "directory_path": directory_path,
            "error": str(e)
        }, indent=2)
