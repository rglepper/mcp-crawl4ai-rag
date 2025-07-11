"""
Search tools for the Crawl4AI MCP server.

This module provides MCP tool wrappers for search operations including
RAG queries and code example searches.
"""
import json

from mcp.server.fastmcp import Context

from src.models import SearchRequest
from src.services.rag_search import SearchService


async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with search results
    """
    try:
        # Get dependencies from context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        embedding_service = ctx.request_context.lifespan_context.embedding_service
        
        # Create service
        service = SearchService(
            supabase_client,
            embedding_service,
            ctx.request_context.lifespan_context.settings
        )
        
        # Create request
        request = SearchRequest(
            query=query,
            source=source,
            match_count=match_count
        )
        
        # Process request
        result = await service.perform_rag_query(request)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": f"RAG query failed: {str(e)}"
        }, indent=2)


async def search_code_examples(ctx: Context, query: str, source_id: str = None, match_count: int = 5) -> str:
    """
    Search for code examples relevant to the query.

    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with code example search results
    """
    try:
        # Get dependencies from context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        embedding_service = ctx.request_context.lifespan_context.embedding_service
        
        # Create service
        service = SearchService(
            supabase_client,
            embedding_service,
            ctx.request_context.lifespan_context.settings
        )
        
        # Create request
        request = SearchRequest(
            query=query,
            source=source_id,
            match_count=match_count
        )
        
        # Process request
        result = await service.search_code_examples(request)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": f"Code search failed: {str(e)}"
        }, indent=2)
