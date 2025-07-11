"""
Neo4j parser service for the Crawl4AI MCP server.

This service handles parsing GitHub repositories and storing the extracted
code structure (classes, methods, functions, imports) in Neo4j.
"""
import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Set
import ast

from src.config import Settings

logger = logging.getLogger(__name__)


class Neo4jCodeAnalyzer:
    """Analyzes Python files and extracts structure for Neo4j storage."""
    
    def __init__(self):
        self.processed_files = set()
    
    def analyze_python_file(self, file_path: Path, repo_root: Path, project_modules: Set[str]) -> Dict[str, Any]:
        """Extract structure for direct Neo4j insertion."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            relative_path = str(file_path.relative_to(repo_root))
            module_name = self._get_importable_module_name(file_path, repo_root, relative_path)
            
            # Extract structure
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract class with its methods and attributes
                    methods = []
                    attributes = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = self._extract_function_info(item, is_method=True)
                            methods.append(method_info)
                        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            # Class attribute with type annotation
                            attr_name = item.target.id
                            attr_type = self._get_type_annotation(item.annotation)
                            attributes.append({
                                'name': attr_name,
                                'type': attr_type,
                                'line_number': getattr(item, 'lineno', 0)
                            })
                    
                    class_info = {
                        'name': node.name,
                        'full_name': f"{module_name}.{node.name}" if module_name else node.name,
                        'line_number': getattr(node, 'lineno', 0),
                        'methods': methods,
                        'attributes': attributes,
                        'base_classes': [self._get_name_from_node(base) for base in node.bases]
                    }
                    classes.append(class_info)
                
                elif isinstance(node, ast.FunctionDef) and self._is_top_level_function(node, tree):
                    # Top-level function
                    func_info = self._extract_function_info(node, is_method=False)
                    func_info['full_name'] = f"{module_name}.{node.name}" if module_name else node.name
                    functions.append(func_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Import statement
                    import_info = self._extract_import_info(node)
                    imports.extend(import_info)
            
            return {
                'file_path': relative_path,
                'module_name': module_name,
                'line_count': len(content.splitlines()),
                'classes': classes,
                'functions': functions,
                'imports': imports
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return {
                'file_path': str(file_path.relative_to(repo_root)),
                'module_name': '',
                'line_count': 0,
                'classes': [],
                'functions': [],
                'imports': [],
                'error': str(e)
            }
    
    def _extract_function_info(self, node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """Extract function/method information."""
        # Extract parameters
        args = []
        params_list = []
        params_detailed = []
        
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = self._get_type_annotation(arg.annotation) if arg.annotation else None
            
            args.append(arg_name)
            params_list.append(arg_name)
            
            param_detail = {'name': arg_name}
            if arg_type:
                param_detail['type'] = arg_type
            params_detailed.append(param_detail)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._get_type_annotation(node.returns)
        
        return {
            'name': node.name,
            'args': args,
            'params_list': params_list,
            'params_detailed': params_detailed,
            'return_type': return_type,
            'line_number': getattr(node, 'lineno', 0),
            'is_method': is_method
        }
    
    def _extract_import_info(self, node: ast.AST) -> List[Dict[str, Any]]:
        """Extract import information."""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'module': alias.name,
                    'name': alias.name,
                    'alias': alias.asname,
                    'is_from_import': False,
                    'line_number': getattr(node, 'lineno', 0)
                })
        
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    'module': module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'is_from_import': True,
                    'line_number': getattr(node, 'lineno', 0)
                })
        
        return imports
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Get string representation of type annotation."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            value = self._get_name_from_node(annotation.value)
            return f"{value}.{annotation.attr}" if value else annotation.attr
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        else:
            return "Any"
    
    def _get_name_from_node(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name_from_node(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        else:
            return "Unknown"
    
    def _is_top_level_function(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is at top level (not inside a class)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return False
        return True
    
    def _get_importable_module_name(self, file_path: Path, repo_root: Path, relative_path: str) -> str:
        """Generate importable module name from file path."""
        if file_path.name == '__init__.py':
            # For __init__.py files, use the parent directory
            parts = file_path.parent.relative_to(repo_root).parts
        else:
            # For regular .py files, use the file name without extension
            parts = file_path.relative_to(repo_root).with_suffix('').parts
        
        # Convert path parts to module name
        module_name = '.'.join(parts) if parts and parts != ('.',) else ''
        return module_name


class Neo4jParserService:
    """Service for parsing repositories and storing in Neo4j."""
    
    def __init__(self, neo4j_driver, settings: Settings):
        """
        Initialize the Neo4j parser service.
        
        Args:
            neo4j_driver: Neo4j database driver
            settings: Application settings
        """
        self.neo4j_driver = neo4j_driver
        self.settings = settings
        self.analyzer = Neo4jCodeAnalyzer()
    
    async def parse_repository(self, repo_url: str, temp_dir: str = None) -> Dict[str, Any]:
        """
        Parse a GitHub repository and store in Neo4j.
        
        Args:
            repo_url: GitHub repository URL
            temp_dir: Temporary directory for cloning (optional)
            
        Returns:
            Dictionary with parsing results
        """
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        logger.info(f"Parsing repository: {repo_name}")
        
        try:
            # Clear existing data for this repository before re-processing
            await self._clear_repository_data(repo_name)
            
            # Set default temp_dir
            if temp_dir is None:
                temp_dir = f"/tmp/neo4j_repos/{repo_name}"
            
            # Clone and analyze
            repo_path = Path(self._clone_repository(repo_url, temp_dir))
            
            # Find Python files
            python_files = list(repo_path.rglob("*.py"))
            logger.info(f"Found {len(python_files)} Python files")
            
            if not python_files:
                return {
                    "success": False,
                    "repo_url": repo_url,
                    "error": "No Python files found in repository"
                }
            
            # Analyze files
            modules_data = self._analyze_python_files(python_files, repo_path)
            
            # Store in Neo4j
            storage_result = await self._store_in_neo4j(repo_name, modules_data)
            
            # Cleanup
            self._cleanup_repository(temp_dir)
            
            return {
                "success": True,
                "repo_url": repo_url,
                "repo_name": repo_name,
                "files_processed": len(modules_data),
                "nodes_created": storage_result.get("nodes_created", 0),
                "relationships_created": storage_result.get("relationships_created", 0)
            }
            
        except Exception as e:
            logger.error(f"Error parsing repository {repo_url}: {str(e)}")
            return {
                "success": False,
                "repo_url": repo_url,
                "error": str(e)
            }
    
    def _clone_repository(self, repo_url: str, target_dir: str) -> str:
        """Clone repository with shallow clone."""
        logger.info(f"Cloning repository to: {target_dir}")
        
        if os.path.exists(target_dir):
            logger.info(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir, ignore_errors=True)
        
        # Create parent directory
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        
        # Clone with shallow depth for speed
        cmd = ["git", "clone", "--depth", "1", repo_url, target_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"Git clone failed: {result.stderr}")
        
        logger.info(f"Repository cloned successfully to {target_dir}")
        return target_dir
    
    def _analyze_python_files(self, python_files: List[Path], repo_root: Path) -> List[Dict[str, Any]]:
        """Analyze Python files and extract structure."""
        modules_data = []
        project_modules = set()
        
        # First pass: collect all module names
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            relative_path = str(file_path.relative_to(repo_root))
            module_name = self.analyzer._get_importable_module_name(file_path, repo_root, relative_path)
            if module_name:
                project_modules.add(module_name.split('.')[0])
        
        # Second pass: analyze files
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                module_data = self.analyzer.analyze_python_file(file_path, repo_root, project_modules)
                if module_data and not module_data.get('error'):
                    modules_data.append(module_data)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
                continue
        
        logger.info(f"Successfully analyzed {len(modules_data)} Python files")
        return modules_data
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'test_',
            '_test.py',
            'tests/',
            'venv/',
            '.venv/',
            'env/',
            '.env/',
            'build/',
            'dist/'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    async def _store_in_neo4j(self, repo_name: str, modules_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store analyzed data in Neo4j."""
        async with self.neo4j_driver.session() as session:
            # Create Repository node
            await session.run(
                "CREATE (r:Repository {name: $repo_name, created_at: datetime()})",
                repo_name=repo_name
            )
            
            nodes_created = 0
            relationships_created = 0
            
            for mod in modules_data:
                # Create File node
                await session.run("""
                    CREATE (f:File {
                        name: $name,
                        path: $path,
                        module_name: $module_name,
                        line_count: $line_count,
                        created_at: datetime()
                    })
                """, 
                    name=mod['file_path'].split('/')[-1],
                    path=mod['file_path'],
                    module_name=mod['module_name'],
                    line_count=mod['line_count']
                )
                nodes_created += 1
                
                # Connect Repository to File
                await session.run("""
                    MATCH (r:Repository {name: $repo_name})
                    MATCH (f:File {path: $file_path})
                    MERGE (r)-[:CONTAINS]->(f)
                """, repo_name=repo_name, file_path=mod['file_path'])
                relationships_created += 1
                
                # Create Class nodes and relationships
                for cls in mod['classes']:
                    await session.run("""
                        MERGE (c:Class {full_name: $full_name})
                        ON CREATE SET c.name = $name, c.created_at = datetime()
                    """, name=cls['name'], full_name=cls['full_name'])
                    nodes_created += 1
                    
                    # Connect File to Class
                    await session.run("""
                        MATCH (f:File {path: $file_path})
                        MATCH (c:Class {full_name: $class_full_name})
                        MERGE (f)-[:DEFINES]->(c)
                    """, file_path=mod['file_path'], class_full_name=cls['full_name'])
                    relationships_created += 1
                    
                    # Create Method nodes
                    for method in cls['methods']:
                        method_id = f"{cls['full_name']}::{method['name']}"
                        await session.run("""
                            MERGE (m:Method {method_id: $method_id})
                            ON CREATE SET m.name = $name,
                                         m.args = $args,
                                         m.params_list = $params_list,
                                         m.params_detailed = $params_detailed,
                                         m.return_type = $return_type,
                                         m.created_at = datetime()
                        """, 
                            name=method['name'],
                            method_id=method_id,
                            args=method['args'],
                            params_list=method.get('params_list', []),
                            params_detailed=method.get('params_detailed', []),
                            return_type=method['return_type']
                        )
                        nodes_created += 1
                        
                        # Connect Class to Method
                        await session.run("""
                            MATCH (c:Class {full_name: $class_full_name})
                            MATCH (m:Method {method_id: $method_id})
                            MERGE (c)-[:HAS_METHOD]->(m)
                        """, class_full_name=cls['full_name'], method_id=method_id)
                        relationships_created += 1
            
            logger.info(f"Created {nodes_created} nodes and {relationships_created} relationships")
            return {"nodes_created": nodes_created, "relationships_created": relationships_created}
    
    async def _clear_repository_data(self, repo_name: str):
        """Clear existing data for a repository."""
        async with self.neo4j_driver.session() as session:
            await session.run("""
                MATCH (r:Repository {name: $repo_name})
                OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
                OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
                OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
                DETACH DELETE r, f, c, m
            """, repo_name=repo_name)
    
    def _cleanup_repository(self, repo_path: str):
        """Clean up cloned repository."""
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
                logger.info(f"Cleaned up repository at {repo_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {repo_path}: {e}")
    
    async def initialize(self):
        """Initialize the service."""
        logger.info("Neo4j parser service initialized")
    
    async def close(self):
        """Close the service."""
        logger.info("Neo4j parser service closed")
