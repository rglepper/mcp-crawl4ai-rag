"""
Knowledge graph tools for the Crawl4AI MCP server.

This module provides MCP tool wrappers for knowledge graph operations including
hallucination detection, knowledge graph querying, and repository parsing.
"""
import json
from pathlib import Path

from mcp.server.fastmcp import Context

from src.models import HallucinationDetectionRequest
from src.services.hallucination_detector import HallucinationDetectorService
from src.services.knowledge_graph import KnowledgeGraphService
from src.services.neo4j_parser import Neo4jParserService


async def check_ai_script_hallucinations(ctx: Context, script_path: str) -> str:
    """
    Check an AI-generated Python script for hallucinations using the knowledge graph.

    This tool analyzes a Python script for potential AI hallucinations by validating
    imports, method calls, class instantiations, and function calls against a Neo4j
    knowledge graph containing real repository data.

    The tool performs comprehensive analysis including:
    - Import validation against known repositories
    - Method call validation on classes from the knowledge graph
    - Class instantiation parameter validation
    - Function call parameter validation
    - Attribute access validation

    Args:
        ctx: The MCP server provided context
        script_path: Absolute path to the Python script to analyze

    Returns:
        JSON string with hallucination detection results
    """
    try:
        # Get Neo4j driver from context
        neo4j_driver = ctx.request_context.lifespan_context.neo4j_driver
        
        # Create service
        service = HallucinationDetectorService(
            neo4j_driver,
            ctx.request_context.lifespan_context.settings
        )
        
        # Create request
        request = HallucinationDetectionRequest(
            script_path=Path(script_path),
            include_suggestions=True,
            confidence_threshold=0.5
        )
        
        # Process request
        result = await service.detect_hallucinations(request)
        
        # Convert to dictionary for JSON serialization
        result_dict = {
            "success": result.success,
            "script_path": result.script_path,
            "total_issues": result.total_issues,
            "confidence_score": result.confidence_score,
            "issues": result.issues,
            "recommendations": result.recommendations
        }
        
        if not result.success and result.error:
            result_dict["error"] = result.error
        
        return json.dumps(result_dict, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "script_path": script_path,
            "error": f"Hallucination detection failed: {str(e)}"
        }, indent=2)


async def query_knowledge_graph(ctx: Context, command: str) -> str:
    """
    Query and explore the Neo4j knowledge graph containing repository data.

    This tool provides comprehensive access to the knowledge graph for exploring repositories,
    classes, methods, functions, and their relationships. Perfect for understanding what data
    is available for hallucination detection and debugging validation results.

    **⚠️ IMPORTANT: Always start with the `repos` command first!**
    Before using any other commands, run `repos` to see what repositories are available
    in your knowledge graph. This will help you understand what data you can explore.

    ## Available Commands:
    - `repos`: List all repositories in the knowledge graph
    - `classes [repo_name]`: List classes in a specific repository
    - `methods [class_name]`: List methods of a specific class
    - `class [class_name]`: Get detailed information about a specific class
    - `method [method_name]`: Search for a specific method across all classes
    - `search [query]`: Search for classes, methods, or functions matching a query

    Args:
        ctx: The MCP server provided context
        command: The knowledge graph query command

    Returns:
        JSON string with query results
    """
    try:
        # Get Neo4j driver from context
        neo4j_driver = ctx.request_context.lifespan_context.neo4j_driver
        
        # Create service
        service = KnowledgeGraphService(
            neo4j_driver,
            ctx.request_context.lifespan_context.settings
        )
        
        # Parse command
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        result = {"success": True}
        
        # Process command
        if cmd == "repos":
            repositories = await service.list_repositories()
            result["repositories"] = repositories
            
        elif cmd == "classes" and args:
            repo_name = args[0]
            classes = await service.list_classes(repo_name)
            result["classes"] = classes
            result["repository"] = repo_name
            
        elif cmd == "methods" and args:
            class_name = args[0]
            methods = await service.get_methods_of_class(class_name)
            result["methods"] = methods
            result["class"] = class_name
            
        elif cmd == "class" and args:
            class_name = args[0]
            class_info = await service.explore_class(class_name)
            result["class_info"] = class_info
            
        elif cmd == "method" and args:
            method_name = args[0]
            class_name = args[1] if len(args) > 1 else None
            methods = await service.search_method(method_name, class_name)
            result["methods"] = methods
            result["search_term"] = method_name
            
        elif cmd == "search" and args:
            search_term = args[0]
            # Custom search implementation
            classes = await service.list_classes(limit=100)
            matching_classes = [c for c in classes if search_term.lower() in c["name"].lower()]
            result["matching_classes"] = matching_classes
            
            methods = []
            for cls in matching_classes[:5]:  # Limit to avoid too many queries
                cls_methods = await service.get_methods_of_class(cls["name"])
                methods.extend(cls_methods)
            
            matching_methods = [m for m in methods if search_term.lower() in m["method_name"].lower()]
            result["matching_methods"] = matching_methods
            result["search_term"] = search_term
            
        else:
            result = {
                "success": False,
                "error": f"Unknown command: {cmd}",
                "available_commands": [
                    "repos", "classes [repo_name]", "methods [class_name]",
                    "class [class_name]", "method [method_name]", "search [query]"
                ]
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Knowledge graph query failed: {str(e)}"
        }, indent=2)


async def parse_github_repository(ctx: Context, repo_url: str) -> str:
    """
    Parse a GitHub repository into the Neo4j knowledge graph.

    This tool clones a GitHub repository, analyzes its Python files, and stores
    the code structure (classes, methods, functions, imports) in Neo4j for use
    in hallucination detection. The tool:

    - Clones the repository to a temporary location
    - Analyzes Python files to extract code structure
    - Stores classes, methods, functions, and imports in Neo4j
    - Provides detailed statistics about the parsing results
    - Automatically handles module name detection for imports

    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')

    Returns:
        JSON string with parsing results
    """
    try:
        # Get Neo4j driver from context
        neo4j_driver = ctx.request_context.lifespan_context.neo4j_driver
        
        # Create service
        service = Neo4jParserService(
            neo4j_driver,
            ctx.request_context.lifespan_context.settings
        )
        
        # Process request
        result = await service.parse_repository(repo_url)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"Repository parsing failed: {str(e)}"
        }, indent=2)
