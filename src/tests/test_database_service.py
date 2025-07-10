"""
Tests for the DatabaseService class.

Following TDD principles - these tests define the expected behavior
of the DatabaseService before implementation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
from supabase import Client


def create_mock_settings():
    """Helper function to create a properly configured mock Settings object."""
    from src.config import Settings
    mock_settings = Mock(spec=Settings)
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_service_key = "test-key"
    return mock_settings


def test_database_service_initializes_with_settings():
    """Test that DatabaseService initializes properly with settings."""
    from src.services.database import DatabaseService
    from src.config import Settings

    # Arrange
    mock_settings = Mock(spec=Settings)
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_service_key = "test-key"

    # Act
    with patch('src.services.database.create_client') as mock_create_client:
        mock_client = Mock(spec=Client)
        mock_create_client.return_value = mock_client

        service = DatabaseService(mock_settings)

    # Assert
    assert service.settings == mock_settings
    assert service.client == mock_client
    mock_create_client.assert_called_once_with("https://test.supabase.co", "test-key")


def test_database_service_raises_error_on_invalid_credentials():
    """Test that DatabaseService raises error when Supabase credentials are invalid."""
    from src.services.database import DatabaseService
    from src.config import Settings

    # Arrange
    mock_settings = Mock(spec=Settings)
    mock_settings.supabase_url = ""
    mock_settings.supabase_service_key = ""

    # Act & Assert
    with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"):
        DatabaseService(mock_settings)


def test_add_documents_batch_success():
    """Test successful batch addition of documents to crawled_pages table."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    # Mock table operations
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    mock_table.delete.return_value.in_.return_value.execute.return_value = None
    mock_table.insert.return_value.execute.return_value = None

    urls = ["https://example.com/page1", "https://example.com/page2"]
    chunk_numbers = [1, 2]
    contents = ["Content 1", "Content 2"]
    metadatas = [{"title": "Page 1"}, {"title": "Page 2"}]
    url_to_full_document = {"https://example.com/page1": "Full doc 1", "https://example.com/page2": "Full doc 2"}

    # Act
    with patch('src.services.database.create_embeddings_batch') as mock_embeddings:
        mock_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

        service.add_documents_batch(urls, chunk_numbers, contents, metadatas, url_to_full_document)

    # Assert
    mock_client.table.assert_called_with("crawled_pages")
    # Check that delete was called with the correct URLs (order doesn't matter)
    delete_call_args = mock_table.delete.return_value.in_.call_args
    assert delete_call_args[0][0] == "url"
    assert set(delete_call_args[0][1]) == {"https://example.com/page1", "https://example.com/page2"}
    mock_table.insert.assert_called_once()


def test_add_documents_batch_handles_embedding_failure():
    """Test that add_documents_batch handles embedding creation failures gracefully."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_table = Mock()
    mock_client.table.return_value = mock_table
    mock_table.delete.return_value.in_.return_value.execute.return_value = None

    # Act
    with patch('src.services.database.create_embeddings_batch') as mock_embeddings:
        mock_embeddings.side_effect = Exception("Embedding API error")

        # Should not raise exception, should handle gracefully
        service.add_documents_batch(["https://example.com"], [1], ["content"], [{}], {})

    # Assert: Should not call insert if embeddings fail
    mock_table.insert.assert_not_called()


def test_search_documents_vector_search():
    """Test vector search functionality in search_documents."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    # Mock RPC call
    mock_rpc_result = Mock()
    mock_rpc_result.data = [{"content": "test result", "similarity": 0.9}]
    mock_client.rpc.return_value.execute.return_value = mock_rpc_result

    # Act
    with patch('src.services.database.create_embedding') as mock_embedding:
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        results = service.search_documents("test query", match_count=5)

    # Assert
    assert len(results) == 1
    assert results[0]["content"] == "test result"
    mock_client.rpc.assert_called_once_with('match_crawled_pages', {
        'query_embedding': [0.1, 0.2, 0.3],
        'match_count': 5
    })


def test_search_documents_with_source_filter():
    """Test search_documents with source filtering."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_rpc_result = Mock()
    mock_rpc_result.data = []
    mock_client.rpc.return_value.execute.return_value = mock_rpc_result

    # Act
    with patch('src.services.database.create_embedding') as mock_embedding:
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        service.search_documents("test query", match_count=5, source_filter="example.com")

    # Assert
    mock_client.rpc.assert_called_once_with('match_crawled_pages', {
        'query_embedding': [0.1, 0.2, 0.3],
        'match_count': 5,
        'filter': {'source_id': 'example.com'}
    })


def test_add_code_examples_batch_success():
    """Test successful batch addition of code examples."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_table = Mock()
    mock_client.table.return_value = mock_table
    mock_table.delete.return_value.eq.return_value.execute.return_value = None
    mock_table.insert.return_value.execute.return_value = None

    urls = ["https://example.com/code1"]
    chunk_numbers = [1]
    code_examples = ["def hello(): pass"]
    summaries = ["Simple hello function"]
    metadatas = [{"language": "python"}]

    # Act
    with patch('src.services.database.create_embeddings_batch') as mock_embeddings:
        mock_embeddings.return_value = [[0.1, 0.2]]

        service.add_code_examples_batch(urls, chunk_numbers, code_examples, summaries, metadatas)

    # Assert
    mock_client.table.assert_called_with("code_examples")
    mock_table.insert.assert_called_once()


def test_search_code_examples_success():
    """Test successful code example search."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_rpc_result = Mock()
    mock_rpc_result.data = [{"content": "def test(): pass", "summary": "Test function"}]
    mock_client.rpc.return_value.execute.return_value = mock_rpc_result

    # Act
    with patch('src.services.database.create_embedding') as mock_embedding:
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        results = service.search_code_examples("test function", match_count=3)

    # Assert
    assert len(results) == 1
    assert results[0]["content"] == "def test(): pass"
    mock_client.rpc.assert_called_once_with('match_code_examples', {
        'query_embedding': [0.1, 0.2, 0.3],
        'match_count': 3
    })


def test_update_source_info_new_source():
    """Test updating source info for a new source."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_table = Mock()
    mock_client.table.return_value = mock_table

    # Mock update returning no data (source doesn't exist)
    mock_update_result = Mock()
    mock_update_result.data = []
    mock_table.update.return_value.eq.return_value.execute.return_value = mock_update_result
    mock_table.insert.return_value.execute.return_value = None

    # Act
    service.update_source_info("example.com", "Test summary", 1000)

    # Assert
    mock_table.update.assert_called_once()
    mock_table.insert.assert_called_once_with({
        'source_id': 'example.com',
        'summary': 'Test summary',
        'total_word_count': 1000
    })


def test_update_source_info_existing_source():
    """Test updating source info for an existing source."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_table = Mock()
    mock_client.table.return_value = mock_table

    # Mock update returning data (source exists)
    mock_update_result = Mock()
    mock_update_result.data = [{"source_id": "example.com"}]
    mock_table.update.return_value.eq.return_value.execute.return_value = mock_update_result

    # Act
    service.update_source_info("example.com", "Updated summary", 2000)

    # Assert
    mock_table.update.assert_called_once()
    mock_table.insert.assert_not_called()  # Should not insert if update succeeded


def test_cleanup_source_success():
    """Test successful source cleanup."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_table = Mock()
    mock_client.table.return_value = mock_table
    mock_table.delete.return_value.eq.return_value.execute.return_value = None

    # Act
    result = service.cleanup_source("example.com")

    # Assert
    assert result["success"] is True
    assert result["source_id"] == "example.com"
    # Should delete from all three tables
    assert mock_client.table.call_count == 3


def test_get_available_sources_success():
    """Test getting available sources."""
    from src.services.database import DatabaseService

    # Arrange
    mock_settings = create_mock_settings()
    mock_client = Mock(spec=Client)

    with patch('src.services.database.create_client', return_value=mock_client):
        service = DatabaseService(mock_settings)

    mock_table = Mock()
    mock_client.table.return_value = mock_table
    mock_select_result = Mock()
    mock_select_result.data = [
        {"source_id": "example.com", "summary": "Test site", "total_word_count": 1000}
    ]
    mock_table.select.return_value.execute.return_value = mock_select_result

    # Act
    sources = service.get_available_sources()

    # Assert
    assert len(sources) == 1
    assert sources[0]["source_id"] == "example.com"
    mock_client.table.assert_called_with("sources")


def test_database_service_handles_connection_errors():
    """Test that DatabaseService handles connection errors gracefully."""
    from src.services.database import DatabaseService
    from src.config import Settings

    # Arrange
    mock_settings = Mock(spec=Settings)
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_service_key = "test-key"

    # Act & Assert
    with patch('src.services.database.create_client') as mock_create_client:
        mock_create_client.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            DatabaseService(mock_settings)
