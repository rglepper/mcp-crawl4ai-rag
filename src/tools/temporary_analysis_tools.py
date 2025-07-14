"""
Temporary analysis tools for the Crawl4AI MCP server.

This module provides MCP tool wrappers for temporary repository analysis operations
including repository analysis, search, listing, and cleanup.
"""
import json

from mcp.server.fastmcp import Context

from src.models import TemporaryAnalysisRequest, TemporaryAnalysisSearchRequest
from src.services.temporary_analysis import TemporaryAnalysisService


async def analyze_repository_temporarily(ctx: Context, repo_url: str, focus_areas: str = None) -> str:
    """
    Temporarily analyze a GitHub repository for research and example purposes without storing in knowledge graph.

    This tool clones a repository, analyzes its Python code structure, and provides detailed information
    about classes, methods, functions, and patterns. Perfect for research when you want to understand
    how a codebase implements certain features or patterns to inform your own development.

    The repository is cloned to a temporary location and deleted after analysis - nothing is stored
    permanently in the knowledge graph.

    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
        focus_areas: Optional comma-separated list of areas to focus on (e.g., 'auth,database,api')

    Returns:
        JSON string with analysis results and analysis_id for future searches
    """
    try:
        # Get temporary analysis service from context
        service = ctx.request_context.lifespan_context.temporary_analysis_service

        # Create request
        request = TemporaryAnalysisRequest(
            repo_url=repo_url,
            focus_areas=focus_areas.split(',') if focus_areas else None
        )

        # Process request
        result = await service.analyze_repository_temporarily(request)

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"Temporary analysis failed: {str(e)}"
        }, indent=2)


async def search_temporary_analysis(ctx: Context, analysis_id: str, search_query: str, search_type: str = "all") -> str:
    """
    Search through temporary repository analysis data for specific patterns, classes, methods, or functions.

    This tool helps you find specific information within saved temporary analysis results.
    Perfect for research when you want to find examples of specific patterns or implementations.

    Args:
        ctx: The MCP server provided context
        analysis_id: Analysis ID from analyze_repository_temporarily tool (e.g., 'pydantic_20250109_143022')
        search_query: What to search for (e.g., 'auth', 'database', 'api', class names, method names)
        search_type: Type of search - 'all', 'classes', 'methods', 'functions', 'modules' (default: 'all')

    Returns:
        JSON string with search results
    """
    try:
        # Get temporary analysis service from context
        service = ctx.request_context.lifespan_context.temporary_analysis_service

        # Create request
        request = TemporaryAnalysisSearchRequest(
            analysis_id=analysis_id,
            search_query=search_query,
            search_type=search_type
        )

        # Process request
        result = await service.search_temporary_analysis(request)

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "analysis_id": analysis_id,
            "search_query": search_query,
            "error": f"Search failed: {str(e)}"
        }, indent=2)


async def list_temporary_analyses(ctx: Context) -> str:
    """
    List all available temporary repository analyses saved in the project.

    This tool shows all previously analyzed repositories that are available for searching.
    Useful to see what analyses are available before using search_temporary_analysis.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with list of available analyses
    """
    try:
        # Get temporary analysis service from context
        service = ctx.request_context.lifespan_context.temporary_analysis_service

        # Process request
        result = await service.list_temporary_analyses()

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list analyses: {str(e)}"
        }, indent=2)


async def cleanup_temporary_analysis(ctx: Context, analysis_id: str = None, all_analyses: bool = False) -> str:
    """
    Remove temporary repository analysis files from the project.

    This tool helps clean up old analysis files to save disk space.

    Args:
        ctx: The MCP server provided context
        analysis_id: Specific analysis ID to remove (e.g., 'pydantic_20250109_143022')
        all_analyses: If True, removes all temporary analyses (default: False)

    Returns:
        JSON string with cleanup results
    """
    try:
        # Validate parameters
        if not analysis_id and not all_analyses:
            return json.dumps({
                "success": False,
                "error": "Must specify either analysis_id or set all_analyses=True"
            }, indent=2)

        # Get temporary analysis service from context
        service = ctx.request_context.lifespan_context.temporary_analysis_service

        # Process request
        if all_analyses:
            result = await service.cleanup_all_temporary_analyses()
        else:
            result = await service.cleanup_temporary_analysis(analysis_id)

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "analysis_id": analysis_id,
            "error": f"Cleanup failed: {str(e)}"
        }, indent=2)
