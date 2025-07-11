"""
Tests for the SearchService class.

Following TDD principles - these tests define the expected behavior
of the SearchService before implementation.
"""
from unittest.mock import Mock, patch


def create_mock_settings():
    """Helper function to create a properly configured mock Settings object."""
    from src.config import Settings
    mock_settings = Mock(spec=Settings)
    mock_settings.use_hybrid_search = False
    mock_settings.use_reranking = False
    return mock_settings


def test_search_service_initializes_with_dependencies():
    """Test that SearchService initializes properly with required dependencies."""
    from src.services.rag_search import SearchService
    from src.services.database import DatabaseService
    from src.services.embedding import EmbeddingService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock(spec=DatabaseService)
    mock_embedding_service = Mock(spec=EmbeddingService)
    mock_reranking_model = None
    
    # Act
    service = SearchService(
        settings=mock_settings,
        database_service=mock_database_service,
        embedding_service=mock_embedding_service,
        reranking_model=mock_reranking_model
    )
    
    # Assert
    assert service.settings == mock_settings
    assert service.database_service == mock_database_service
    assert service.embedding_service == mock_embedding_service
    assert service.reranking_model == mock_reranking_model


def test_search_documents_vector_only():
    """Test basic vector search for documents without hybrid or reranking."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    
    # Mock database search results
    mock_results = [
        {"url": "https://example.com", "content": "Test content", "similarity": 0.9}
    ]
    mock_database_service.search_documents.return_value = mock_results
    
    # Act
    results = service.search_documents("test query", match_count=5)
    
    # Assert
    assert results["success"] is True
    assert results["search_mode"] == "vector"
    assert results["reranking_applied"] is False
    assert len(results["results"]) == 1
    assert results["results"][0]["content"] == "Test content"
    mock_database_service.search_documents.assert_called_once_with(
        "test query", match_count=5, source_filter=None
    )


def test_search_documents_with_source_filter():
    """Test document search with source filtering."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    mock_database_service.search_documents.return_value = []
    
    # Act
    service.search_documents("test query", source_filter="example.com")
    
    # Assert
    mock_database_service.search_documents.assert_called_once_with(
        "test query", match_count=10, source_filter="example.com"
    )


def test_search_documents_hybrid_mode():
    """Test hybrid search combining vector and keyword search."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_settings.use_hybrid_search = True
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    
    # Mock vector and keyword results
    vector_results = [{"url": "url1", "content": "vector content", "similarity": 0.9}]
    keyword_results = [{"url": "url2", "content": "keyword content"}]
    
    mock_database_service.search_documents.return_value = vector_results
    
    with patch.object(service, '_perform_keyword_search') as mock_keyword:
        mock_keyword.return_value = keyword_results
        
        # Act
        results = service.search_documents("test query", match_count=5)
    
    # Assert
    assert results["search_mode"] == "hybrid"
    mock_database_service.search_documents.assert_called_once_with(
        "test query", match_count=10, source_filter=None  # Double match_count for hybrid
    )
    mock_keyword.assert_called_once()


def test_search_documents_with_reranking():
    """Test document search with reranking enabled."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_settings.use_reranking = True
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    mock_reranking_model = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, mock_reranking_model)
    
    # Mock search results
    original_results = [
        {"url": "url1", "content": "content1", "similarity": 0.8},
        {"url": "url2", "content": "content2", "similarity": 0.9}
    ]
    reranked_results = [
        {"url": "url2", "content": "content2", "similarity": 0.9, "rerank_score": 0.95},
        {"url": "url1", "content": "content1", "similarity": 0.8, "rerank_score": 0.85}
    ]
    
    mock_database_service.search_documents.return_value = original_results
    
    with patch.object(service, '_rerank_results') as mock_rerank:
        mock_rerank.return_value = reranked_results
        
        # Act
        results = service.search_documents("test query")
    
    # Assert
    assert results["reranking_applied"] is True
    assert results["results"][0]["rerank_score"] == 0.95
    mock_rerank.assert_called_once_with("test query", original_results, "content")


def test_search_code_examples_vector_only():
    """Test basic vector search for code examples."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    
    # Mock database search results
    mock_results = [
        {"url": "url1", "content": "def test(): pass", "summary": "Test function", "similarity": 0.9}
    ]
    mock_database_service.search_code_examples.return_value = mock_results
    
    # Act
    results = service.search_code_examples("test function", match_count=3)
    
    # Assert
    assert results["success"] is True
    assert results["search_mode"] == "vector"
    assert len(results["results"]) == 1
    assert results["results"][0]["code"] == "def test(): pass"
    mock_database_service.search_code_examples.assert_called_once_with(
        "test function", match_count=3, source_filter=None
    )


def test_search_code_examples_hybrid_mode():
    """Test hybrid search for code examples."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_settings.use_hybrid_search = True
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    
    vector_results = [{"url": "url1", "content": "vector code", "summary": "summary", "similarity": 0.9}]
    keyword_results = [{"url": "url2", "content": "keyword code", "summary": "summary"}]
    
    mock_database_service.search_code_examples.return_value = vector_results
    
    with patch.object(service, '_perform_keyword_search_code') as mock_keyword:
        mock_keyword.return_value = keyword_results
        
        # Act
        results = service.search_code_examples("test", match_count=5)
    
    # Assert
    assert results["search_mode"] == "hybrid"
    mock_keyword.assert_called_once()


def test_rerank_results_success():
    """Test successful reranking of search results."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    mock_reranking_model = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, mock_reranking_model)
    
    results = [
        {"content": "First result", "similarity": 0.8},
        {"content": "Second result", "similarity": 0.9}
    ]
    
    # Mock reranking scores (reversed order)
    mock_reranking_model.predict.return_value = [0.95, 0.85]
    
    # Act
    reranked = service._rerank_results("test query", results, "content")
    
    # Assert
    assert len(reranked) == 2
    assert reranked[0]["rerank_score"] == 0.95
    assert reranked[1]["rerank_score"] == 0.85
    assert reranked[0]["content"] == "First result"  # Should be reordered by score


def test_rerank_results_handles_errors():
    """Test that reranking handles errors gracefully."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    mock_reranking_model = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, mock_reranking_model)
    
    results = [{"content": "Test result"}]
    mock_reranking_model.predict.side_effect = Exception("Reranking failed")
    
    # Act
    reranked = service._rerank_results("test query", results, "content")
    
    # Assert
    assert reranked == results  # Should return original results on error


def test_perform_keyword_search_documents():
    """Test keyword search for documents."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    
    # Mock Supabase client and query
    mock_client = Mock()
    service.database_service.client = mock_client
    
    mock_query_result = Mock()
    mock_query_result.data = [{"url": "url1", "content": "keyword match"}]
    mock_client.from_.return_value.select.return_value.ilike.return_value.limit.return_value.execute.return_value = mock_query_result
    
    # Act
    results = service._perform_keyword_search("test query", match_count=5)
    
    # Assert
    assert len(results) == 1
    assert results[0]["content"] == "keyword match"
    mock_client.from_.assert_called_with('crawled_pages')


def test_perform_keyword_search_code_examples():
    """Test keyword search for code examples."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    
    # Mock Supabase client and query
    mock_client = Mock()
    service.database_service.client = mock_client
    
    mock_query_result = Mock()
    mock_query_result.data = [{"url": "url1", "content": "def test(): pass", "summary": "Test function"}]
    mock_client.from_.return_value.select.return_value.or_.return_value.limit.return_value.execute.return_value = mock_query_result
    
    # Act
    results = service._perform_keyword_search_code("test function", match_count=3)
    
    # Assert
    assert len(results) == 1
    assert results[0]["content"] == "def test(): pass"
    mock_client.from_.assert_called_with('code_examples')


def test_combine_and_deduplicate_results():
    """Test combining and deduplicating search results."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    
    vector_results = [
        {"url": "url1", "content": "content1", "similarity": 0.9},
        {"url": "url2", "content": "content2", "similarity": 0.8}
    ]
    keyword_results = [
        {"url": "url2", "content": "content2"},  # Duplicate
        {"url": "url3", "content": "content3"}   # New
    ]
    
    # Act
    combined = service._combine_and_deduplicate_results(vector_results, keyword_results, max_results=5)
    
    # Assert
    assert len(combined) == 3  # Should deduplicate url2
    urls = [result["url"] for result in combined]
    assert "url1" in urls
    assert "url2" in urls
    assert "url3" in urls


def test_search_handles_database_errors():
    """Test that search methods handle database errors gracefully."""
    from src.services.rag_search import SearchService
    
    # Arrange
    mock_settings = create_mock_settings()
    mock_database_service = Mock()
    mock_embedding_service = Mock()
    
    service = SearchService(mock_settings, mock_database_service, mock_embedding_service, None)
    mock_database_service.search_documents.side_effect = Exception("Database error")
    
    # Act
    results = service.search_documents("test query")
    
    # Assert
    assert results["success"] is False
    assert "error" in results
    assert "Database error" in results["error"]
