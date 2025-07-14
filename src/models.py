"""
Pydantic models for the Crawl4AI MCP server.

This module provides type-safe data models for all MCP tool requests and responses,
ensuring proper validation and serialization throughout the application.
"""
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class CrawlType(str, Enum):
    """Enumeration of different crawl types supported by the system."""
    SINGLE_PAGE = "single_page"
    SITEMAP = "sitemap"
    TXT_FILE = "txt_file"
    RECURSIVE = "recursive"


# Crawling Models
class CrawlRequest(BaseModel):
    """Request model for crawling operations."""
    url: HttpUrl = Field(..., description="URL to crawl")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum recursion depth")
    max_concurrent: int = Field(default=10, ge=1, le=50, description="Maximum concurrent sessions")
    chunk_size: int = Field(default=5000, ge=100, le=10000, description="Maximum chunk size in characters")


class CrawlResult(BaseModel):
    """Response model for crawling operations."""
    success: bool = Field(..., description="Whether the crawl was successful")
    url: str = Field(..., description="The crawled URL")
    crawl_type: CrawlType = Field(..., description="Type of crawl performed")
    pages_crawled: int = Field(default=0, ge=0, description="Number of pages crawled")
    chunks_stored: int = Field(default=0, ge=0, description="Number of chunks stored")
    code_examples_stored: int = Field(default=0, ge=0, description="Number of code examples stored")
    error: Optional[str] = Field(default=None, description="Error message if crawl failed")


# Search Models
class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    source: Optional[str] = Field(default=None, description="Optional source filter")
    match_count: int = Field(default=5, ge=1, le=20, description="Maximum number of results")


class RAGResponse(BaseModel):
    """Response model for RAG search operations."""
    success: bool = Field(..., description="Whether the search was successful")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    search_mode: str = Field(..., description="Search mode used (vector, hybrid, etc.)")
    reranking_applied: bool = Field(..., description="Whether reranking was applied")
    count: int = Field(..., ge=0, description="Number of results returned")
    error: Optional[str] = Field(default=None, description="Error message if search failed")


# Directory Ingestion Models
class DirectoryIngestionRequest(BaseModel):
    """Request model for local directory ingestion."""
    directory_path: Path = Field(..., description="Path to directory containing files")
    source_name: Optional[str] = Field(default=None, description="Custom source name")
    file_extensions: str = Field(default=".md,.txt,.markdown", description="File extensions to process")
    recursive: bool = Field(default=True, description="Whether to search subdirectories")
    chunk_size: int = Field(default=5000, ge=100, le=10000, description="Maximum chunk size")


# Knowledge Graph Models
class RepositoryAnalysisRequest(BaseModel):
    """Request model for repository analysis operations."""
    repo_url: HttpUrl = Field(..., description="GitHub repository URL")
    focus_areas: Optional[str] = Field(default=None, description="Comma-separated focus areas")


class HallucinationDetectionRequest(BaseModel):
    """Request model for AI script hallucination detection."""
    script_path: Path = Field(..., description="Path to Python script to analyze")


class HallucinationResult(BaseModel):
    """Response model for hallucination detection results."""
    success: bool = Field(..., description="Whether the analysis was successful")
    script_path: str = Field(..., description="Path to the analyzed script")
    total_issues: int = Field(..., ge=0, description="Total number of issues found")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of analysis")
    issues: List[Dict[str, Any]] = Field(..., description="List of detected issues")
    recommendations: List[str] = Field(..., description="List of recommendations")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")


class KnowledgeGraphQuery(BaseModel):
    """Request model for knowledge graph queries."""
    command: str = Field(..., min_length=1, description="Query command to execute")


# Source Management Models
class SourceCleanupRequest(BaseModel):
    """Request model for source cleanup operations."""
    source_id: str = Field(..., description="Source ID to clean up")
    confirm: bool = Field(default=False, description="Confirmation flag for deletion")





# Temporary Analysis Models
class TemporaryAnalysisRequest(BaseModel):
    """Request model for temporary repository analysis operations."""
    repo_url: str = Field(..., description="GitHub repository URL to analyze")
    focus_areas: Optional[List[str]] = Field(default=None, description="Optional list of areas to focus on")
    include_tests: bool = Field(default=False, description="Whether to include test files in analysis")


class TemporaryAnalysisSearchRequest(BaseModel):
    """Request model for searching temporary analysis data."""
    analysis_id: str = Field(..., description="Analysis ID to search within")
    search_query: str = Field(..., description="Search query")
    search_type: str = Field(default="all", description="Type of search (all, classes, methods, etc.)")


# Application Context Models
class Crawl4AIContext(BaseModel):
    """Context model containing all application dependencies."""
    crawler: Any = Field(..., description="AsyncWebCrawler instance")
    supabase_client: Any = Field(..., description="Supabase client instance")
    reranking_model: Optional[Any] = Field(None, description="Cross-encoder reranking model")
    neo4j_driver: Optional[Any] = Field(None, description="Neo4j async driver")
    settings: Any = Field(..., description="Application settings")

    # Core services (initialized once during startup)
    database_service: Any = Field(..., description="Database service for Supabase operations")
    embedding_service: Any = Field(..., description="Embedding service for vector operations")
    search_service: Any = Field(..., description="Search service for RAG operations")
    web_crawling_service: Any = Field(..., description="Web crawling service")
    directory_ingestion_service: Any = Field(..., description="Directory ingestion service")
    source_management_service: Any = Field(..., description="Source management service")
    temporary_analysis_service: Any = Field(..., description="Temporary analysis service")

    # Neo4j services (initialized once during startup if Neo4j is available)
    hallucination_detector_service: Optional[Any] = Field(None, description="Hallucination detector service")
    knowledge_graph_service: Optional[Any] = Field(None, description="Knowledge graph service")
    neo4j_parser_service: Optional[Any] = Field(None, description="Neo4j parser service")

    class Config:
        arbitrary_types_allowed = True
