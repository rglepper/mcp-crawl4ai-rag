"""
Tests for the models module.

Following TDD principles - these tests define the expected behavior
of all Pydantic models before implementation.
"""
import pytest
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import ValidationError, HttpUrl
from enum import Enum


def test_crawl_type_enum_has_correct_values():
    """Test that CrawlType enum has all expected values."""
    from src.models import CrawlType
    
    # Assert: All expected crawl types are present
    assert CrawlType.SINGLE_PAGE == "single_page"
    assert CrawlType.SITEMAP == "sitemap"
    assert CrawlType.TXT_FILE == "txt_file"
    assert CrawlType.RECURSIVE == "recursive"


def test_crawl_request_validates_required_fields():
    """Test that CrawlRequest validates required URL field."""
    from src.models import CrawlRequest
    
    # Act & Assert: Should raise ValidationError when URL is missing
    with pytest.raises(ValidationError) as exc_info:
        CrawlRequest()
    
    assert "url" in str(exc_info.value)


def test_crawl_request_has_correct_defaults():
    """Test that CrawlRequest has correct default values."""
    from src.models import CrawlRequest
    
    # Arrange & Act
    request = CrawlRequest(url="https://example.com")
    
    # Assert: Check default values
    assert request.max_depth == 3
    assert request.max_concurrent == 10
    assert request.chunk_size == 5000


def test_crawl_request_validates_field_ranges():
    """Test that CrawlRequest validates field ranges."""
    from src.models import CrawlRequest
    
    # Test max_depth validation
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", max_depth=0)  # Below minimum
    
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", max_depth=11)  # Above maximum
    
    # Test max_concurrent validation
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", max_concurrent=0)  # Below minimum
    
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", max_concurrent=51)  # Above maximum
    
    # Test chunk_size validation
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", chunk_size=99)  # Below minimum
    
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", chunk_size=10001)  # Above maximum


def test_crawl_result_model_structure():
    """Test that CrawlResult model has correct structure and validation."""
    from src.models import CrawlResult, CrawlType
    
    # Test successful result
    result = CrawlResult(
        success=True,
        url="https://example.com",
        crawl_type=CrawlType.SINGLE_PAGE,
        pages_crawled=1,
        chunks_stored=5,
        code_examples_stored=2
    )
    
    assert result.success is True
    assert result.url == "https://example.com"
    assert result.crawl_type == CrawlType.SINGLE_PAGE
    assert result.pages_crawled == 1
    assert result.chunks_stored == 5
    assert result.code_examples_stored == 2
    assert result.error is None


def test_crawl_result_error_case():
    """Test that CrawlResult handles error cases properly."""
    from src.models import CrawlResult, CrawlType
    
    # Test error result
    result = CrawlResult(
        success=False,
        url="https://example.com",
        crawl_type=CrawlType.SINGLE_PAGE,
        error="Connection failed"
    )
    
    assert result.success is False
    assert result.error == "Connection failed"
    assert result.pages_crawled == 0  # Default value
    assert result.chunks_stored == 0  # Default value
    assert result.code_examples_stored == 0  # Default value


def test_search_request_validates_query_length():
    """Test that SearchRequest validates query length constraints."""
    from src.models import SearchRequest
    
    # Test empty query
    with pytest.raises(ValidationError):
        SearchRequest(query="")
    
    # Test query too long
    long_query = "a" * 1001
    with pytest.raises(ValidationError):
        SearchRequest(query=long_query)
    
    # Test valid query
    request = SearchRequest(query="test query")
    assert request.query == "test query"
    assert request.source is None
    assert request.match_count == 5


def test_search_request_validates_match_count_range():
    """Test that SearchRequest validates match_count range."""
    from src.models import SearchRequest
    
    # Test below minimum
    with pytest.raises(ValidationError):
        SearchRequest(query="test", match_count=0)
    
    # Test above maximum
    with pytest.raises(ValidationError):
        SearchRequest(query="test", match_count=21)
    
    # Test valid range
    request = SearchRequest(query="test", match_count=10)
    assert request.match_count == 10


def test_rag_response_model_structure():
    """Test that RAGResponse model has correct structure."""
    from src.models import RAGResponse
    
    # Test successful response
    response = RAGResponse(
        success=True,
        results=[{"content": "test", "score": 0.9}],
        search_mode="vector",
        reranking_applied=True,
        count=1
    )
    
    assert response.success is True
    assert len(response.results) == 1
    assert response.search_mode == "vector"
    assert response.reranking_applied is True
    assert response.count == 1
    assert response.error is None


def test_directory_ingestion_request_validates_path():
    """Test that DirectoryIngestionRequest validates directory path."""
    from src.models import DirectoryIngestionRequest
    
    # Test with valid path
    request = DirectoryIngestionRequest(directory_path=Path("/tmp/test"))
    assert request.directory_path == Path("/tmp/test")
    assert request.source_name is None
    assert request.file_extensions == ".md,.txt,.markdown"
    assert request.recursive is True
    assert request.chunk_size == 5000


def test_directory_ingestion_request_validates_chunk_size():
    """Test that DirectoryIngestionRequest validates chunk_size range."""
    from src.models import DirectoryIngestionRequest
    
    # Test below minimum
    with pytest.raises(ValidationError):
        DirectoryIngestionRequest(directory_path=Path("/tmp"), chunk_size=99)
    
    # Test above maximum
    with pytest.raises(ValidationError):
        DirectoryIngestionRequest(directory_path=Path("/tmp"), chunk_size=10001)


def test_repository_analysis_request_validates_url():
    """Test that RepositoryAnalysisRequest validates repository URL."""
    from src.models import RepositoryAnalysisRequest
    
    # Test valid GitHub URL
    request = RepositoryAnalysisRequest(repo_url="https://github.com/user/repo.git")
    assert str(request.repo_url) == "https://github.com/user/repo.git"
    assert request.focus_areas is None
    
    # Test with focus areas
    request_with_focus = RepositoryAnalysisRequest(
        repo_url="https://github.com/user/repo.git",
        focus_areas="auth,api,database"
    )
    assert request_with_focus.focus_areas == "auth,api,database"


def test_hallucination_detection_request_validates_path():
    """Test that HallucinationDetectionRequest validates script path."""
    from src.models import HallucinationDetectionRequest
    
    # Test with valid path
    request = HallucinationDetectionRequest(script_path=Path("/path/to/script.py"))
    assert request.script_path == Path("/path/to/script.py")


def test_hallucination_result_model_structure():
    """Test that HallucinationResult model has correct structure."""
    from src.models import HallucinationResult
    
    # Test successful result
    result = HallucinationResult(
        success=True,
        script_path="/path/to/script.py",
        total_issues=2,
        confidence_score=0.85,
        issues=[{"type": "import", "message": "Unknown import"}],
        recommendations=["Check import spelling"]
    )
    
    assert result.success is True
    assert result.script_path == "/path/to/script.py"
    assert result.total_issues == 2
    assert result.confidence_score == 0.85
    assert len(result.issues) == 1
    assert len(result.recommendations) == 1
    assert result.error is None


def test_knowledge_graph_query_validates_command():
    """Test that KnowledgeGraphQuery validates command field."""
    from src.models import KnowledgeGraphQuery
    
    # Test empty command
    with pytest.raises(ValidationError):
        KnowledgeGraphQuery(command="")
    
    # Test valid command
    query = KnowledgeGraphQuery(command="repos")
    assert query.command == "repos"


def test_temporary_analysis_request_validates_fields():
    """Test that TemporaryAnalysisRequest validates all fields."""
    from src.models import TemporaryAnalysisRequest
    
    # Test valid request
    request = TemporaryAnalysisRequest(
        analysis_id="pydantic_20250109_143022",
        search_query="auth",
        search_type="classes"
    )
    
    assert request.analysis_id == "pydantic_20250109_143022"
    assert request.search_query == "auth"
    assert request.search_type == "classes"


def test_temporary_analysis_request_has_default_search_type():
    """Test that TemporaryAnalysisRequest has correct default search_type."""
    from src.models import TemporaryAnalysisRequest
    
    request = TemporaryAnalysisRequest(
        analysis_id="test_id",
        search_query="test"
    )
    
    assert request.search_type == "all"


def test_source_cleanup_request_validates_fields():
    """Test that SourceCleanupRequest validates fields correctly."""
    from src.models import SourceCleanupRequest
    
    # Test with defaults
    request = SourceCleanupRequest(source_id="example.com")
    assert request.source_id == "example.com"
    assert request.confirm is False
    
    # Test with confirm=True
    request_confirmed = SourceCleanupRequest(source_id="example.com", confirm=True)
    assert request_confirmed.confirm is True


def test_all_models_can_be_serialized_to_json():
    """Test that all models can be serialized to JSON."""
    from src.models import (
        CrawlRequest, CrawlResult, CrawlType, SearchRequest, RAGResponse,
        DirectoryIngestionRequest, RepositoryAnalysisRequest,
        HallucinationDetectionRequest, HallucinationResult,
        KnowledgeGraphQuery, TemporaryAnalysisRequest, SourceCleanupRequest
    )
    
    # Test each model can be serialized
    models_to_test = [
        CrawlRequest(url="https://example.com"),
        CrawlResult(success=True, url="https://example.com", crawl_type=CrawlType.SINGLE_PAGE),
        SearchRequest(query="test"),
        RAGResponse(success=True, results=[], search_mode="vector", reranking_applied=False, count=0),
        DirectoryIngestionRequest(directory_path=Path("/tmp")),
        RepositoryAnalysisRequest(repo_url="https://github.com/user/repo.git"),
        HallucinationDetectionRequest(script_path=Path("/path/to/script.py")),
        HallucinationResult(success=True, script_path="/path", total_issues=0, confidence_score=1.0, issues=[], recommendations=[]),
        KnowledgeGraphQuery(command="repos"),
        TemporaryAnalysisRequest(analysis_id="test", search_query="test"),
        SourceCleanupRequest(source_id="test.com")
    ]
    
    for model in models_to_test:
        # Should not raise an exception
        json_str = model.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
