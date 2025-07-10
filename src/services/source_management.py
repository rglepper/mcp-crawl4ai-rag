"""
Source management service for the Crawl4AI MCP server.

This service handles source management operations including cleanup,
analysis, and knowledge base guide generation.
"""
import logging
from typing import Dict, List, Any, Optional

from src.config import Settings
from src.models import SourceCleanupRequest

logger = logging.getLogger(__name__)


class SourceManagementService:
    """Service for managing sources in the knowledge base."""
    
    def __init__(self, supabase_client, settings: Settings):
        """
        Initialize the source management service.
        
        Args:
            supabase_client: Supabase client for database operations
            settings: Application settings
        """
        self.supabase_client = supabase_client
        self.settings = settings
    
    async def cleanup_source(self, request: SourceCleanupRequest) -> Dict[str, Any]:
        """
        Clean up a source and all its associated data.
        
        Args:
            request: Source cleanup request
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            source_id = request.source_id
            
            # Get current counts before deletion
            pages_result = self.supabase_client.table('crawled_pages').select('id').eq('source_id', source_id).execute()
            pages_count = len(pages_result.data)
            
            code_result = self.supabase_client.table('code_examples').select('id').eq('source_id', source_id).execute()
            code_count = len(code_result.data)
            
            # Delete crawled pages
            self.supabase_client.table('crawled_pages').delete().eq('source_id', source_id).execute()
            
            # Delete code examples
            self.supabase_client.table('code_examples').delete().eq('source_id', source_id).execute()
            
            # Delete source metadata
            self.supabase_client.table('sources').delete().eq('source_id', source_id).execute()
            
            return {
                "success": True,
                "source_id": source_id,
                "pages_removed": pages_count,
                "code_examples_removed": code_count
            }
            
        except Exception as e:
            logger.error(f"Source cleanup failed: {e}")
            return {
                "success": False,
                "source_id": request.source_id,
                "error": f"Cleanup failed: {str(e)}"
            }
    
    async def analyze_crawl_types(self) -> Dict[str, Any]:
        """
        Analyze crawl types for all sources.
        
        Returns:
            Dictionary with crawl type analysis
        """
        try:
            # Get crawl type statistics for each source
            result = self.supabase_client.rpc('get_crawl_type_stats').execute()
            
            if result.data:
                crawl_types = {}
                for row in result.data:
                    crawl_types[row['source_id']] = row['crawl_type']
                
                return {
                    'success': True,
                    'crawl_types': crawl_types,
                    'total_sources': len(crawl_types)
                }
            else:
                # Fallback: get sources and infer crawl types
                sources_result = self.supabase_client.table('sources').select('*').execute()
                crawl_types = {}
                
                for source in sources_result.data:
                    crawl_types[source['source_id']] = source.get('crawl_type', 'unknown')
                
                return {
                    'success': True,
                    'crawl_types': crawl_types,
                    'total_sources': len(crawl_types)
                }
                
        except Exception as e:
            logger.error(f"Crawl type analysis failed: {e}")
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }
    
    async def get_knowledge_base_guide(self, neo4j_driver=None) -> Dict[str, Any]:
        """
        Generate a knowledge base guide with available resources.
        
        Args:
            neo4j_driver: Optional Neo4j driver for knowledge graph data
            
        Returns:
            Dictionary with knowledge base guide
        """
        try:
            # Get available sources from Supabase
            sources_result = self.supabase_client.table('sources').select('*').execute()
            sources = sources_result.data
            
            # Get knowledge graph repositories if Neo4j is available
            knowledge_graph_repos = []
            if neo4j_driver:
                try:
                    from src.services.knowledge_graph import KnowledgeGraphService
                    kg_service = KnowledgeGraphService(neo4j_driver, self.settings)
                    knowledge_graph_repos = await kg_service.list_repositories()
                except Exception as e:
                    logger.warning(f"Could not fetch knowledge graph data: {e}")
            
            # Build resource list
            resources = []
            
            # Add documentation sources
            for source in sources:
                resources.append({
                    "type": "documentation",
                    "source_id": source['source_id'],
                    "title": source.get('title', source['source_id']),
                    "description": source.get('description', 'Documentation source'),
                    "pages": source.get('pages_count', 0),
                    "when_to_use": f"Use for {source['source_id']} documentation and examples"
                })
            
            # Add knowledge graph repositories
            for repo in knowledge_graph_repos:
                resources.append({
                    "type": "code_repository",
                    "repo_name": repo['name'],
                    "description": f"Code repository with {repo.get('file_count', 0)} files",
                    "when_to_use": f"Use for {repo['name']} code structure and hallucination detection"
                })
            
            # Create the guide
            guide = {
                "summary": f"Available resources: {len(sources)} documentation sources, {len(knowledge_graph_repos)} code repositories",
                "resources": resources,
                "usage_notes": {
                    "documentation_search": "Use perform_rag_query('your question', source='source_id') for targeted searches",
                    "code_search": "Use search_code_examples('pattern') for implementation examples",
                    "graph_exploration": "Use query_knowledge_graph('repos') to start exploring code repositories",
                    "hallucination_check": "Use check_ai_script_hallucinations('/path/to/script.py') to validate AI-generated code"
                }
            }
            
            return {
                "success": True,
                "guide": guide,
                "sources_count": len(sources) + len(knowledge_graph_repos)
            }
            
        except Exception as e:
            logger.error(f"Knowledge base guide generation failed: {e}")
            return {
                "success": False,
                "error": f"Failed to generate knowledge base guide: {str(e)}"
            }
    
    async def get_available_sources(self) -> Dict[str, Any]:
        """
        Get all available sources from the database.
        
        Returns:
            Dictionary with available sources
        """
        try:
            # Get sources with statistics
            sources_result = self.supabase_client.table('sources').select('*').execute()
            sources = sources_result.data
            
            # Format sources for response
            formatted_sources = []
            for source in sources:
                formatted_sources.append({
                    "source_id": source['source_id'],
                    "title": source.get('title', source['source_id']),
                    "description": source.get('description', ''),
                    "url": source.get('url', ''),
                    "crawl_type": source.get('crawl_type', 'unknown'),
                    "pages": source.get('pages_count', 0),
                    "chunks": source.get('chunks_count', 0)
                })
            
            return {
                "success": True,
                "sources": formatted_sources,
                "total_sources": len(formatted_sources)
            }
            
        except Exception as e:
            logger.error(f"Get available sources failed: {e}")
            return {
                "success": False,
                "error": f"Failed to get available sources: {str(e)}"
            }
