"""
Source management tools for the Crawl4AI MCP server.

This module provides MCP tool wrappers for source management operations including
source cleanup, crawl type analysis, knowledge base guide generation, and source listing.
"""
import json
from typing import Any

from mcp.server.fastmcp import Context

from src.models import SourceCleanupRequest
from src.services.source_management import SourceManagementService


async def cleanup_source(ctx: Context, source_id: str, confirm: bool = False) -> str:
    """
    Remove a source and all its associated data from the knowledge base.

    This tool removes all crawled pages, code examples, and source metadata
    for a specified source_id. Use with caution as this action is irreversible.

    Args:
        ctx: The MCP server provided context
        source_id: The source ID to remove (e.g., 'example.com')
        confirm: Must be set to True to actually perform the deletion

    Returns:
        JSON string with cleanup results
    """
    try:
        # Check confirmation
        if not confirm:
            return json.dumps({
                "success": False,
                "source_id": source_id,
                "error": "Confirmation required. Set confirm=True to proceed with deletion."
            }, indent=2)
        
        # Get Supabase client from context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Create service
        service = SourceManagementService(supabase_client, ctx.request_context.lifespan_context.settings)
        
        # Create request
        request = SourceCleanupRequest(source_id=source_id, confirm=confirm)
        
        # Process request
        result = await service.cleanup_source(request)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "source_id": source_id,
            "error": f"Cleanup failed: {str(e)}"
        }, indent=2)


async def analyze_crawl_types(ctx: Context) -> str:
    """
    Analyze the crawl types used for each source to understand crawling scope.

    This tool shows whether each source was crawled as:
    - 'sitemap': Full site crawl from sitemap (many pages)
    - 'webpage': Recursive crawl from a starting page (multiple pages)
    - 'text_file': Single file crawl (one page)

    Returns:
        JSON string with crawl type analysis for each source
    """
    try:
        # Get Supabase client from context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Create service
        service = SourceManagementService(supabase_client, ctx.request_context.lifespan_context.settings)
        
        # Process request
        result = await service.analyze_crawl_types()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }, indent=2)


async def get_knowledge_base_guide(ctx: Context) -> str:
    """
    Get a concise guide to available knowledge base resources with usage recommendations.

    This tool provides LLMs with a simple list of available resources including:
    - Documentation sources in Supabase
    - Code repositories in the knowledge graph
    - Brief description and when to use each resource

    Returns:
        JSON string with concise resource guide and usage instructions
    """
    try:
        # Get Supabase client and Neo4j driver from context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        neo4j_driver = ctx.request_context.lifespan_context.neo4j_driver
        
        # Create service
        service = SourceManagementService(supabase_client, ctx.request_context.lifespan_context.settings)
        
        # Process request
        result = await service.get_knowledge_base_guide(neo4j_driver)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to generate knowledge base guide: {str(e)}"
        }, indent=2)


async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with available sources
    """
    try:
        # Get Supabase client from context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Create service
        service = SourceManagementService(supabase_client, ctx.request_context.lifespan_context.settings)
        
        # Process request
        result = await service.get_available_sources()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get available sources: {str(e)}"
        }, indent=2)
