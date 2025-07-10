"""
Temporary analysis service for the Crawl4AI MCP server.

This service handles temporary repository analysis operations including
repository cloning, analysis, searching, and cleanup.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import Settings
from src.models import TemporaryAnalysisRequest, TemporaryAnalysisSearchRequest
from src.services.script_analyzer import ScriptAnalyzerService

logger = logging.getLogger(__name__)


class TemporaryAnalysisService:
    """Service for temporary repository analysis."""
    
    def __init__(self, settings: Settings):
        """
        Initialize the temporary analysis service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.script_analyzer = ScriptAnalyzerService(settings)
        self.analysis_dir = Path("knowledge_graphs/temp_analysis")
        self.repos_dir = Path("knowledge_graphs/repos")
        
        # Create directories if they don't exist
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.repos_dir, exist_ok=True)
    
    async def analyze_repository_temporarily(self, request: TemporaryAnalysisRequest) -> Dict[str, Any]:
        """
        Analyze a GitHub repository temporarily.
        
        Args:
            request: Temporary analysis request
            
        Returns:
            Dictionary with analysis results
        """
        try:
            repo_url = request.repo_url
            focus_areas = request.focus_areas or []
            
            # Extract repo name from URL
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            
            # Generate unique analysis ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_id = f"{repo_name}_{timestamp}"
            
            # Clone repository
            repo_path = self._clone_repository(repo_url)
            
            try:
                # Analyze repository
                analysis_data = self._analyze_repository(repo_path, focus_areas)
                
                # Save analysis data
                analysis_file = self.analysis_dir / f"{analysis_id}.json"
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2)
                
                # Prepare result
                result = {
                    "success": True,
                    "analysis_id": analysis_id,
                    "repo_url": repo_url,
                    "repo_name": repo_name,
                    "files_analyzed": analysis_data["file_count"],
                    "classes_found": len(analysis_data["classes"]),
                    "methods_found": sum(len(cls["methods"]) for cls in analysis_data["classes"]),
                    "functions_found": len(analysis_data["functions"]),
                    "focus_areas": focus_areas
                }
                
                # Add focus area stats if specified
                if focus_areas:
                    focus_stats = {}
                    for area in focus_areas:
                        area_lower = area.lower()
                        classes_matching = [cls for cls in analysis_data["classes"] 
                                          if area_lower in cls["name"].lower()]
                        methods_matching = sum(1 for cls in analysis_data["classes"] 
                                             for method in cls["methods"] 
                                             if area_lower in method["name"].lower())
                        functions_matching = [func for func in analysis_data["functions"] 
                                            if area_lower in func["name"].lower()]
                        
                        focus_stats[area] = {
                            "classes": len(classes_matching),
                            "methods": methods_matching,
                            "functions": len(functions_matching)
                        }
                    
                    result["focus_area_stats"] = focus_stats
                
                return result
                
            finally:
                # Clean up cloned repository
                self._cleanup_repository(repo_path)
            
        except Exception as e:
            logger.error(f"Temporary repository analysis failed: {e}")
            return {
                "success": False,
                "repo_url": request.repo_url,
                "error": f"Analysis failed: {str(e)}"
            }
    
    async def search_temporary_analysis(self, request: TemporaryAnalysisSearchRequest) -> Dict[str, Any]:
        """
        Search through temporary analysis data.
        
        Args:
            request: Temporary analysis search request
            
        Returns:
            Dictionary with search results
        """
        try:
            analysis_id = request.analysis_id
            search_query = request.search_query.lower()
            search_type = request.search_type.lower()
            
            # Load analysis data
            analysis_file = self.analysis_dir / f"{analysis_id}.json"
            if not analysis_file.exists():
                return {
                    "success": False,
                    "analysis_id": analysis_id,
                    "error": f"Analysis not found: {analysis_id}"
                }
            
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # Perform search
            matches = []
            
            # Search classes
            if search_type in ["all", "classes"]:
                for cls in analysis_data["classes"]:
                    if search_query in cls["name"].lower():
                        matches.append({
                            "type": "class",
                            "name": cls["name"],
                            "file": cls["file"],
                            "methods_count": len(cls["methods"])
                        })
            
            # Search methods
            if search_type in ["all", "methods"]:
                for cls in analysis_data["classes"]:
                    for method in cls["methods"]:
                        if search_query in method["name"].lower():
                            matches.append({
                                "type": "method",
                                "name": method["name"],
                                "class": cls["name"],
                                "file": cls["file"],
                                "params": method.get("params", []),
                                "return_type": method.get("return_type")
                            })
            
            # Search functions
            if search_type in ["all", "functions"]:
                for func in analysis_data["functions"]:
                    if search_query in func["name"].lower():
                        matches.append({
                            "type": "function",
                            "name": func["name"],
                            "file": func["file"],
                            "params": func.get("params", []),
                            "return_type": func.get("return_type")
                        })
            
            # Search modules
            if search_type in ["all", "modules"]:
                for module in analysis_data["modules"]:
                    if search_query in module["name"].lower():
                        matches.append({
                            "type": "module",
                            "name": module["name"],
                            "file": module["file"],
                            "imports": module.get("imports", [])[:5]
                        })
            
            return {
                "success": True,
                "analysis_id": analysis_id,
                "search_query": search_query,
                "search_type": search_type,
                "matches": matches,
                "total_matches": len(matches)
            }
            
        except Exception as e:
            logger.error(f"Temporary analysis search failed: {e}")
            return {
                "success": False,
                "analysis_id": request.analysis_id,
                "search_query": request.search_query,
                "error": f"Search failed: {str(e)}"
            }
    
    async def list_temporary_analyses(self) -> Dict[str, Any]:
        """
        List all available temporary analyses.
        
        Returns:
            Dictionary with list of analyses
        """
        try:
            analyses = []
            
            for file in self.analysis_dir.glob("*.json"):
                try:
                    analysis_id = file.stem
                    created_at = datetime.fromtimestamp(file.stat().st_mtime)
                    
                    # Extract basic info from file
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    repo_name = analysis_id.split('_')[0] if '_' in analysis_id else analysis_id
                    
                    analyses.append({
                        "analysis_id": analysis_id,
                        "repo_name": repo_name,
                        "created_at": created_at.isoformat(),
                        "file_size": file.stat().st_size,
                        "files_analyzed": data.get("file_count", 0),
                        "classes_count": len(data.get("classes", [])),
                        "functions_count": len(data.get("functions", []))
                    })
                except Exception as e:
                    logger.warning(f"Error processing analysis file {file}: {e}")
                    continue
            
            # Sort by creation time (newest first)
            analyses.sort(key=lambda x: x["created_at"], reverse=True)
            
            return {
                "success": True,
                "analyses": analyses,
                "total_analyses": len(analyses)
            }
            
        except Exception as e:
            logger.error(f"List temporary analyses failed: {e}")
            return {
                "success": False,
                "error": f"Failed to list analyses: {str(e)}"
            }
    
    async def cleanup_temporary_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """
        Clean up a specific temporary analysis.
        
        Args:
            analysis_id: Analysis ID to clean up
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            analysis_file = self.analysis_dir / f"{analysis_id}.json"
            
            if not analysis_file.exists():
                return {
                    "success": False,
                    "analysis_id": analysis_id,
                    "error": f"Analysis not found: {analysis_id}"
                }
            
            # Remove analysis file
            analysis_file.unlink()
            
            return {
                "success": True,
                "analysis_id": analysis_id,
                "files_removed": 1
            }
            
        except Exception as e:
            logger.error(f"Cleanup temporary analysis failed: {e}")
            return {
                "success": False,
                "analysis_id": analysis_id,
                "error": f"Cleanup failed: {str(e)}"
            }
    
    async def cleanup_all_temporary_analyses(self) -> Dict[str, Any]:
        """
        Clean up all temporary analyses.
        
        Returns:
            Dictionary with cleanup results
        """
        try:
            # Count files before deletion
            analysis_files = list(self.analysis_dir.glob("*.json"))
            file_count = len(analysis_files)
            
            # Remove all analysis files
            for file in analysis_files:
                file.unlink()
            
            return {
                "success": True,
                "analyses_removed": file_count,
                "files_removed": file_count
            }
            
        except Exception as e:
            logger.error(f"Cleanup all temporary analyses failed: {e}")
            return {
                "success": False,
                "error": f"Cleanup failed: {str(e)}"
            }
    
    def _clone_repository(self, repo_url: str) -> str:
        """Clone a GitHub repository to a temporary directory."""
        # Create a unique directory name
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_dir = self.repos_dir / f"{repo_name}_{timestamp}"
        
        # Create parent directory
        os.makedirs(self.repos_dir, exist_ok=True)
        
        # Clone with shallow depth for speed
        cmd = ["git", "clone", "--depth", "1", repo_url, str(repo_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"Git clone failed: {result.stderr}")
        
        return str(repo_dir)
    
    def _analyze_repository(self, repo_path: str, focus_areas: List[str]) -> Dict[str, Any]:
        """Analyze a repository and extract code structure."""
        repo_dir = Path(repo_path)
        
        # Find Python files
        python_files = list(repo_dir.rglob("*.py"))
        
        # Skip test files if not explicitly focused on tests
        if "tests" not in focus_areas and "test" not in focus_areas:
            python_files = [f for f in python_files if "test" not in str(f).lower()]
        
        # Analyze files
        classes = []
        functions = []
        modules = []
        
        for file_path in python_files:
            try:
                # Use script analyzer to analyze file
                result = self.script_analyzer.analyze_script(str(file_path))
                
                # Skip files with errors
                if result.errors:
                    continue
                
                # Extract relative path
                rel_path = str(file_path.relative_to(repo_dir))
                
                # Process classes and methods
                for cls in result.class_instantiations:
                    # Find methods for this class
                    methods = [
                        {
                            "name": m.method_name,
                            "params": m.args,
                            "kwargs": m.kwargs
                        }
                        for m in result.method_calls
                        if m.object_type == cls.class_name
                    ]
                    
                    classes.append({
                        "name": cls.class_name,
                        "file": rel_path,
                        "methods": methods
                    })
                
                # Process functions
                for func in result.function_calls:
                    functions.append({
                        "name": func.function_name,
                        "file": rel_path,
                        "params": func.args,
                        "kwargs": func.kwargs
                    })
                
                # Process module
                modules.append({
                    "name": file_path.stem,
                    "file": rel_path,
                    "imports": [
                        {
                            "module": imp.module,
                            "name": imp.name,
                            "is_from_import": imp.is_from_import
                        }
                        for imp in result.imports
                    ]
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
                continue
        
        # Compile analysis data
        analysis_data = {
            "repo_path": repo_path,
            "file_count": len(python_files),
            "classes": classes,
            "functions": functions,
            "modules": modules,
            "focus_areas": focus_areas
        }
        
        return analysis_data
    
    def _cleanup_repository(self, repo_path: str):
        """Clean up cloned repository."""
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup {repo_path}: {e}")
