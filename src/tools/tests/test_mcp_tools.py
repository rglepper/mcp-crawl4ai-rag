"""
Tests for MCP tool wrappers.

This module tests all 16 MCP tool wrapper functions to ensure they properly
delegate to the appropriate services and return correctly formatted responses.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
import json

from src.tools.crawling_tools import (
    crawl_single_page,
    smart_crawl_url,
    ingest_local_directory
)
from src.tools.source_management_tools import (
    cleanup_source,
    analyze_crawl_types,
    get_knowledge_base_guide,
    get_available_sources
)
from src.tools.search_tools import (
    perform_rag_query,
    search_code_examples
)
from src.tools.knowledge_graph_tools import (
    check_ai_script_hallucinations,
    query_knowledge_graph,
    parse_github_repository
)
from src.tools.temporary_analysis_tools import (
    analyze_repository_temporarily,
    search_temporary_analysis,
    list_temporary_analyses,
    cleanup_temporary_analysis
)


@pytest.fixture
def mock_context():
    """Create mock MCP context for testing."""
    context = Mock()
    context.request_context = Mock()
    context.request_context.lifespan_context = Mock()

    # Mock Supabase client
    supabase_client = Mock()
    context.request_context.lifespan_context.supabase_client = supabase_client

    # Mock Neo4j driver
    neo4j_driver = AsyncMock()
    context.request_context.lifespan_context.neo4j_driver = neo4j_driver

    # Mock AsyncWebCrawler
    crawler = AsyncMock()
    context.request_context.lifespan_context.crawler = crawler

    return context


class TestCrawlingTools:
    """Test crawling tool wrappers."""

    @patch('src.tools.crawling_tools.WebCrawlingService')
    async def test_crawl_single_page_success(self, mock_service_class, mock_context):
        """Test successful single page crawling."""
        # Mock service instance and response
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = Mock()
        mock_result.success = True
        mock_result.url = "https://example.com"
        mock_result.crawl_type = "single_page"
        mock_result.pages_crawled = 1
        mock_result.chunks_stored = 5
        mock_result.code_examples_stored = 2

        mock_service.process_crawl_request.return_value = mock_result

        # Call the tool
        result = await crawl_single_page(mock_context, "https://example.com")

        # Parse and verify result
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["url"] == "https://example.com"
        assert result_data["crawl_type"] == "single_page"
        assert result_data["pages_crawled"] == 1

        # Verify service was called correctly
        mock_service.process_crawl_request.assert_called_once()

    @patch('src.tools.crawling_tools.WebCrawlingService')
    async def test_crawl_single_page_failure(self, mock_service_class, mock_context):
        """Test single page crawling failure."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = Mock()
        mock_result.success = False
        mock_result.url = "https://example.com"
        mock_result.crawl_type = "single_page"
        mock_result.pages_crawled = 0
        mock_result.chunks_stored = 0
        mock_result.code_examples_stored = 0
        mock_result.error = "Network error"

        mock_service.process_crawl_request.return_value = mock_result

        result = await crawl_single_page(mock_context, "https://example.com")

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert result_data["error"] == "Network error"

    @patch('src.tools.crawling_tools.WebCrawlingService')
    async def test_smart_crawl_url_success(self, mock_service_class, mock_context):
        """Test successful smart URL crawling."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = Mock()
        mock_result.success = True
        mock_result.url = "https://example.com/sitemap.xml"
        mock_result.crawl_type = "sitemap"
        mock_result.pages_crawled = 10
        mock_result.chunks_stored = 50
        mock_result.code_examples_stored = 5

        mock_service.process_crawl_request.return_value = mock_result

        result = await smart_crawl_url(
            mock_context,
            "https://example.com/sitemap.xml",
            max_depth=3,
            max_concurrent=10,
            chunk_size=5000
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["crawl_type"] == "sitemap"
        assert result_data["pages_crawled"] == 10

    @patch('src.services.directory_ingestion.DirectoryIngestionService')
    async def test_ingest_local_directory_success(self, mock_service_class, mock_context):
        """Test successful local directory ingestion."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock the context structure
        mock_context.request_context.lifespan_context.directory_ingestion_service = None
        mock_context.request_context.lifespan_context.supabase_client = Mock()
        mock_context.request_context.lifespan_context.settings = Mock()

        mock_result = {
            "success": True,
            "files_processed": 5,
            "chunks_stored": 25,
            "source_name": "test_docs"
        }

        mock_service.ingest_directory.return_value = mock_result

        result = await ingest_local_directory(
            mock_context,
            "/path/to/docs",
            source_name="test_docs",
            file_extensions=".md,.txt",
            recursive=True,
            chunk_size=5000
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["files_processed"] == 5
        assert result_data["source_name"] == "test_docs"


class TestSourceManagementTools:
    """Test source management tool wrappers."""

    @patch('src.tools.source_management_tools.SourceManagementService')
    async def test_cleanup_source_success(self, mock_service_class, mock_context):
        """Test successful source cleanup."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "source_id": "example.com",
            "pages_removed": 10,
            "code_examples_removed": 5
        }

        mock_service.cleanup_source.return_value = mock_result

        result = await cleanup_source(mock_context, "example.com", confirm=True)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["source_id"] == "example.com"
        assert result_data["pages_removed"] == 10

    @patch('src.tools.source_management_tools.SourceManagementService')
    async def test_cleanup_source_not_confirmed(self, mock_service_class, mock_context):
        """Test source cleanup without confirmation."""
        result = await cleanup_source(mock_context, "example.com", confirm=False)

        result_data = json.loads(result)
        assert result_data["success"] is False
        assert "confirmation required" in result_data["error"].lower()

    @patch('src.tools.source_management_tools.SourceManagementService')
    async def test_analyze_crawl_types(self, mock_service_class, mock_context):
        """Test crawl types analysis."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "crawl_types": {
                "example.com": "sitemap",
                "docs.example.com": "webpage"
            }
        }

        mock_service.analyze_crawl_types.return_value = mock_result

        result = await analyze_crawl_types(mock_context)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "crawl_types" in result_data

    @patch('src.tools.source_management_tools.SourceManagementService')
    async def test_get_knowledge_base_guide(self, mock_service_class, mock_context):
        """Test knowledge base guide generation."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "guide": "Knowledge base guide content",
            "sources_count": 5
        }

        mock_service.get_knowledge_base_guide.return_value = mock_result

        result = await get_knowledge_base_guide(mock_context)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "guide" in result_data
        assert result_data["sources_count"] == 5

    @patch('src.tools.source_management_tools.SourceManagementService')
    async def test_get_available_sources(self, mock_service_class, mock_context):
        """Test getting available sources."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "sources": [
                {"source_id": "example.com", "pages": 10},
                {"source_id": "docs.example.com", "pages": 5}
            ]
        }

        mock_service.get_available_sources.return_value = mock_result

        result = await get_available_sources(mock_context)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["sources"]) == 2


class TestSearchTools:
    """Test search tool wrappers."""

    @patch('src.tools.search_tools.SearchService')
    async def test_perform_rag_query_success(self, mock_service_class, mock_context):
        """Test successful RAG query."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "query": "test query",
            "results": [
                {"content": "Test content 1", "source": "example.com"},
                {"content": "Test content 2", "source": "example.com"}
            ],
            "total_results": 2
        }

        mock_service.perform_rag_query.return_value = mock_result

        result = await perform_rag_query(
            mock_context,
            "test query",
            source="example.com",
            match_count=5
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["query"] == "test query"
        assert len(result_data["results"]) == 2

    @patch('src.tools.search_tools.SearchService')
    async def test_search_code_examples_success(self, mock_service_class, mock_context):
        """Test successful code examples search."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "query": "authentication",
            "code_examples": [
                {"code": "def authenticate():", "summary": "Auth function"},
                {"code": "class AuthManager:", "summary": "Auth class"}
            ],
            "total_examples": 2
        }

        mock_service.search_code_examples.return_value = mock_result

        result = await search_code_examples(
            mock_context,
            "authentication",
            source_id="example.com",
            match_count=5
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["query"] == "authentication"
        assert len(result_data["code_examples"]) == 2


class TestKnowledgeGraphTools:
    """Test knowledge graph tool wrappers."""

    @patch('src.tools.knowledge_graph_tools.HallucinationDetectorService')
    async def test_check_ai_script_hallucinations_success(self, mock_service_class, mock_context):
        """Test successful AI script hallucination check."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = Mock()
        mock_result.success = True
        mock_result.script_path = "/path/to/script.py"
        mock_result.total_issues = 2
        mock_result.confidence_score = 0.75
        mock_result.issues = [
            {"type": "invalid_import", "description": "Module not found"},
            {"type": "invalid_method", "description": "Method not found"}
        ]
        mock_result.recommendations = ["Check imports", "Verify method names"]

        mock_service.detect_hallucinations.return_value = mock_result

        result = await check_ai_script_hallucinations(mock_context, "/path/to/script.py")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["script_path"] == "/path/to/script.py"
        assert result_data["total_issues"] == 2
        assert result_data["confidence_score"] == 0.75

    @patch('src.tools.knowledge_graph_tools.KnowledgeGraphService')
    async def test_query_knowledge_graph_repos(self, mock_service_class, mock_context):
        """Test knowledge graph repository query."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_repositories = [
            {"name": "repo1", "file_count": 10},
            {"name": "repo2", "file_count": 15}
        ]

        mock_service.list_repositories.return_value = mock_repositories

        result = await query_knowledge_graph(mock_context, "repos")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "repositories" in result_data
        assert len(result_data["repositories"]) == 2

    @patch('src.tools.knowledge_graph_tools.KnowledgeGraphService')
    async def test_query_knowledge_graph_classes(self, mock_service_class, mock_context):
        """Test knowledge graph classes query."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_classes = [
            {"name": "TestClass", "full_name": "module.TestClass"},
            {"name": "AnotherClass", "full_name": "module.AnotherClass"}
        ]

        mock_service.list_classes.return_value = mock_classes

        result = await query_knowledge_graph(mock_context, "classes repo1")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "classes" in result_data
        assert len(result_data["classes"]) == 2

    @patch('src.tools.knowledge_graph_tools.Neo4jParserService')
    async def test_parse_github_repository_success(self, mock_service_class, mock_context):
        """Test successful GitHub repository parsing."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "repo_url": "https://github.com/test/repo.git",
            "repo_name": "repo",
            "files_processed": 10,
            "nodes_created": 50,
            "relationships_created": 25
        }

        mock_service.parse_repository.return_value = mock_result

        result = await parse_github_repository(mock_context, "https://github.com/test/repo.git")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["repo_name"] == "repo"
        assert result_data["files_processed"] == 10


class TestTemporaryAnalysisTools:
    """Test temporary analysis tool wrappers."""

    @patch('src.tools.temporary_analysis_tools.TemporaryAnalysisService')
    async def test_analyze_repository_temporarily_success(self, mock_service_class, mock_context):
        """Test successful temporary repository analysis."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "analysis_id": "test_repo_20250109_143022",
            "repo_url": "https://github.com/test/repo.git",
            "files_analyzed": 15,
            "classes_found": 10,
            "methods_found": 50
        }

        mock_service.analyze_repository_temporarily.return_value = mock_result

        result = await analyze_repository_temporarily(
            mock_context,
            "https://github.com/test/repo.git",
            focus_areas="authentication,database"
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "test_repo" in result_data["analysis_id"]
        assert result_data["files_analyzed"] == 15

    @patch('src.tools.temporary_analysis_tools.TemporaryAnalysisService')
    async def test_search_temporary_analysis_success(self, mock_service_class, mock_context):
        """Test successful temporary analysis search."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "analysis_id": "test_repo_20250109_143022",
            "search_query": "auth",
            "matches": [
                {"type": "class", "name": "AuthManager", "file": "auth.py"},
                {"type": "method", "name": "authenticate", "class": "AuthManager"}
            ],
            "total_matches": 2
        }

        mock_service.search_temporary_analysis.return_value = mock_result

        result = await search_temporary_analysis(
            mock_context,
            "test_repo_20250109_143022",
            "auth",
            search_type="all"
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["search_query"] == "auth"
        assert len(result_data["matches"]) == 2

    @patch('src.tools.temporary_analysis_tools.TemporaryAnalysisService')
    async def test_list_temporary_analyses(self, mock_service_class, mock_context):
        """Test listing temporary analyses."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "analyses": [
                {
                    "analysis_id": "test_repo_20250109_143022",
                    "repo_name": "test_repo",
                    "created_at": "2025-01-09T14:30:22Z",
                    "files_analyzed": 15
                },
                {
                    "analysis_id": "another_repo_20250109_150000",
                    "repo_name": "another_repo",
                    "created_at": "2025-01-09T15:00:00Z",
                    "files_analyzed": 8
                }
            ],
            "total_analyses": 2
        }

        mock_service.list_temporary_analyses.return_value = mock_result

        result = await list_temporary_analyses(mock_context)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["analyses"]) == 2
        assert result_data["total_analyses"] == 2

    @patch('src.tools.temporary_analysis_tools.TemporaryAnalysisService')
    async def test_cleanup_temporary_analysis_single(self, mock_service_class, mock_context):
        """Test cleanup of single temporary analysis."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "analysis_id": "test_repo_20250109_143022",
            "files_removed": 1
        }

        mock_service.cleanup_temporary_analysis.return_value = mock_result

        result = await cleanup_temporary_analysis(
            mock_context,
            analysis_id="test_repo_20250109_143022"
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["analysis_id"] == "test_repo_20250109_143022"
        assert result_data["files_removed"] == 1

    @patch('src.tools.temporary_analysis_tools.TemporaryAnalysisService')
    async def test_cleanup_temporary_analysis_all(self, mock_service_class, mock_context):
        """Test cleanup of all temporary analyses."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        mock_result = {
            "success": True,
            "analyses_removed": 3,
            "files_removed": 3
        }

        mock_service.cleanup_all_temporary_analyses.return_value = mock_result

        result = await cleanup_temporary_analysis(
            mock_context,
            all_analyses=True
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["analyses_removed"] == 3
        assert result_data["files_removed"] == 3
