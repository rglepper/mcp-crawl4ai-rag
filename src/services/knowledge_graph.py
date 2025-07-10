"""
Knowledge graph service for the Crawl4AI MCP server.

This service handles querying and exploring the Neo4j knowledge graph
containing repository information.
"""
import logging
from typing import List, Dict, Any, Optional

from src.config import Settings

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service for querying the Neo4j knowledge graph."""
    
    def __init__(self, neo4j_driver, settings: Settings):
        """
        Initialize the knowledge graph service.
        
        Args:
            neo4j_driver: Neo4j database driver
            settings: Application settings
        """
        self.neo4j_driver = neo4j_driver
        self.settings = settings
    
    async def list_repositories(self) -> List[Dict[str, Any]]:
        """
        List all repositories in the knowledge graph.
        
        Returns:
            List of repository information
        """
        async with self.neo4j_driver.session() as session:
            query = """
            MATCH (r:Repository)
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            RETURN r.name as name, count(f) as file_count
            ORDER BY r.name
            """
            result = await session.run(query)
            
            repositories = []
            async for record in result:
                repositories.append({
                    "name": record["name"],
                    "file_count": record["file_count"]
                })
            
            return repositories
    
    async def explore_repository(self, repo_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific repository.
        
        Args:
            repo_name: Name of the repository
            
        Returns:
            Repository details including statistics
        """
        async with self.neo4j_driver.session() as session:
            # Get file count
            files_query = """
            MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
            RETURN count(f) as file_count
            """
            result = await session.run(files_query, repo_name=repo_name)
            file_count = (await result.single())['file_count']
            
            # Get class count
            classes_query = """
            MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
            RETURN count(DISTINCT c) as class_count
            """
            result = await session.run(classes_query, repo_name=repo_name)
            class_count = (await result.single())['class_count']
            
            # Get function count
            functions_query = """
            MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
            RETURN count(DISTINCT func) as function_count
            """
            result = await session.run(functions_query, repo_name=repo_name)
            function_count = (await result.single())['function_count']
            
            return {
                "name": repo_name,
                "file_count": file_count,
                "class_count": class_count,
                "function_count": function_count
            }
    
    async def list_classes(self, repo_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List classes in the knowledge graph.
        
        Args:
            repo_name: Optional repository name to filter by
            limit: Maximum number of classes to return
            
        Returns:
            List of class information
        """
        async with self.neo4j_driver.session() as session:
            if repo_name:
                query = """
                MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
                RETURN c.name as name, c.full_name as full_name, f.path as file_path
                ORDER BY c.name
                LIMIT $limit
                """
                result = await session.run(query, repo_name=repo_name, limit=limit)
            else:
                query = """
                MATCH (c:Class)
                RETURN c.name as name, c.full_name as full_name
                ORDER BY c.name
                LIMIT $limit
                """
                result = await session.run(query, limit=limit)
            
            classes = []
            async for record in result:
                class_info = {
                    "name": record["name"],
                    "full_name": record["full_name"]
                }
                if "file_path" in record:
                    class_info["file_path"] = record["file_path"]
                classes.append(class_info)
            
            return classes
    
    async def explore_class(self, class_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific class.
        
        Args:
            class_name: Name of the class to explore
            
        Returns:
            Class details including methods and attributes
        """
        async with self.neo4j_driver.session() as session:
            # Find the class
            class_query = """
            MATCH (c:Class)
            WHERE c.name = $class_name OR c.full_name = $class_name
            RETURN c.name as name, c.full_name as full_name
            LIMIT 1
            """
            result = await session.run(class_query, class_name=class_name)
            class_record = await result.single()
            
            if not class_record:
                return {"error": f"Class '{class_name}' not found"}
            
            actual_name = class_record['name']
            full_name = class_record['full_name']
            
            # Get methods
            methods_query = """
            MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
            WHERE c.name = $class_name OR c.full_name = $class_name
            RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed, m.return_type as return_type
            ORDER BY m.name
            """
            result = await session.run(methods_query, class_name=class_name)
            
            methods = []
            async for record in result:
                # Use detailed params if available, fall back to simple params
                params_to_use = record['params_detailed'] or record['params_list'] or []
                methods.append({
                    'name': record['name'],
                    'parameters': params_to_use,
                    'return_type': record['return_type'] or 'Any'
                })
            
            return {
                "name": actual_name,
                "full_name": full_name,
                "methods": methods
            }
    
    async def search_method(self, method_name: str, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for methods in the knowledge graph.
        
        Args:
            method_name: Name of the method to search for
            class_name: Optional class name to filter by
            
        Returns:
            List of matching methods
        """
        async with self.neo4j_driver.session() as session:
            if class_name:
                query = """
                MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE (c.name = $class_name OR c.full_name = $class_name) AND m.name = $method_name
                RETURN c.name as class_name, c.full_name as class_full_name, 
                       m.name as method_name, m.params_list as params_list, 
                       m.params_detailed as params_detailed, m.return_type as return_type
                """
                result = await session.run(query, class_name=class_name, method_name=method_name)
            else:
                query = """
                MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
                WHERE m.name = $method_name
                RETURN c.name as class_name, c.full_name as class_full_name,
                       m.name as method_name, m.params_list as params_list,
                       m.params_detailed as params_detailed, m.return_type as return_type
                ORDER BY c.name
                LIMIT 20
                """
                result = await session.run(query, method_name=method_name)
            
            methods = []
            async for record in result:
                params_to_use = record['params_detailed'] or record['params_list'] or []
                methods.append({
                    'class_name': record['class_name'],
                    'class_full_name': record['class_full_name'],
                    'method_name': record['method_name'],
                    'parameters': params_to_use,
                    'return_type': record['return_type'] or 'Any'
                })
            
            return methods
    
    async def run_custom_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Run a custom Cypher query against the knowledge graph.
        
        Args:
            cypher_query: Cypher query to execute
            
        Returns:
            Query results
        """
        async with self.neo4j_driver.session() as session:
            try:
                result = await session.run(cypher_query)
                
                records = []
                async for record in result:
                    # Convert record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        record_dict[key] = record[key]
                    records.append(record_dict)
                
                return records
                
            except Exception as e:
                logger.error(f"Error executing custom query: {e}")
                return [{"error": str(e)}]
    
    async def search_files_importing(self, target_module: str) -> List[Dict[str, Any]]:
        """
        Find files that import a specific module.
        
        Args:
            target_module: Module name to search for
            
        Returns:
            List of files importing the module
        """
        async with self.neo4j_driver.session() as session:
            query = """
            MATCH (source:File)-[:IMPORTS]->(target:File)
            WHERE target.module_name CONTAINS $target
            RETURN source.path as file, target.module_name as imports
            """
            result = await session.run(query, target=target_module)
            
            files = []
            async for record in result:
                files.append({
                    "file": record["file"],
                    "imports": record["imports"]
                })
            
            return files
    
    async def get_classes_in_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get classes defined in a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of classes in the file
        """
        async with self.neo4j_driver.session() as session:
            query = """
            MATCH (f:File {path: $file_path})-[:DEFINES]->(c:Class)
            RETURN c.name as class_name, c.full_name as class_full_name
            ORDER BY c.name
            """
            result = await session.run(query, file_path=file_path)
            
            classes = []
            async for record in result:
                classes.append({
                    "class_name": record["class_name"],
                    "class_full_name": record["class_full_name"]
                })
            
            return classes
    
    async def get_methods_of_class(self, class_name: str) -> List[Dict[str, Any]]:
        """
        Get methods of a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of methods in the class
        """
        async with self.neo4j_driver.session() as session:
            query = """
            MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
            WHERE c.name = $class_name OR c.full_name = $class_name
            RETURN m.name as method_name, m.args as args, m.return_type as return_type
            ORDER BY m.name
            """
            result = await session.run(query, class_name=class_name)
            
            methods = []
            async for record in result:
                methods.append({
                    "method_name": record["method_name"],
                    "args": record["args"] or [],
                    "return_type": record["return_type"] or "Any"
                })
            
            return methods
    
    async def initialize(self):
        """Initialize the service."""
        logger.info("Knowledge graph service initialized")
    
    async def close(self):
        """Close the service."""
        logger.info("Knowledge graph service closed")
