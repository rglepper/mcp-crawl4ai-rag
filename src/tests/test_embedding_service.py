"""
Tests for the EmbeddingService class.

Following TDD principles - these tests define the expected behavior
of the EmbeddingService before implementation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple
import openai


def create_mock_settings():
    """Helper function to create a properly configured mock Settings object."""
    from src.config import Settings
    mock_settings = Mock(spec=Settings)
    mock_settings.openai_api_key = "test-openai-key"
    mock_settings.model_choice = "gpt-4o-mini"
    mock_settings.use_contextual_embeddings = False
    return mock_settings


def test_embedding_service_initializes_with_settings():
    """Test that EmbeddingService initializes properly with settings."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()

    # Act
    service = EmbeddingService(mock_settings)

    # Assert
    assert service.settings == mock_settings
    assert service.embedding_model == "text-embedding-3-small"
    assert service.embedding_dimension == 1536


def test_embedding_service_raises_error_on_missing_api_key():
    """Test that EmbeddingService raises error when OpenAI API key is missing."""
    from src.services.embedding import EmbeddingService
    from src.config import Settings

    # Arrange
    mock_settings = Mock(spec=Settings)
    mock_settings.openai_api_key = ""

    # Act & Assert
    with pytest.raises(ValueError, match="OpenAI API key must be set"):
        EmbeddingService(mock_settings)


def test_create_embeddings_batch_success():
    """Test successful batch embedding creation."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    texts = ["Hello world", "Test text"]
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6])
    ]

    # Act
    with patch('openai.embeddings.create', return_value=mock_response) as mock_create:
        embeddings = service.create_embeddings_batch(texts)

    # Assert
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    mock_create.assert_called_once_with(
        model="text-embedding-3-small",
        input=texts
    )


def test_create_embeddings_batch_empty_input():
    """Test that create_embeddings_batch handles empty input gracefully."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    # Act
    embeddings = service.create_embeddings_batch([])

    # Assert
    assert embeddings == []


def test_create_embeddings_batch_with_retry_logic():
    """Test that create_embeddings_batch retries on failure and eventually succeeds."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    texts = ["Test text"]
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    # Act
    with patch('openai.embeddings.create') as mock_create:
        with patch('time.sleep') as mock_sleep:
            # First call fails, second succeeds
            mock_create.side_effect = [Exception("API Error"), mock_response]

            embeddings = service.create_embeddings_batch(texts)

    # Assert
    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert mock_create.call_count == 2
    mock_sleep.assert_called_once_with(1.0)  # First retry delay


def test_create_embeddings_batch_fallback_to_individual():
    """Test that create_embeddings_batch falls back to individual creation after max retries."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    texts = ["Text 1", "Text 2"]
    mock_individual_response = Mock()
    mock_individual_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    # Act
    with patch('openai.embeddings.create') as mock_create:
        with patch('time.sleep'):
            # Batch calls fail, individual calls succeed
            mock_create.side_effect = [
                Exception("Batch Error 1"),
                Exception("Batch Error 2"),
                Exception("Batch Error 3"),
                mock_individual_response,  # First individual call
                mock_individual_response   # Second individual call
            ]

            embeddings = service.create_embeddings_batch(texts)

    # Assert
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.1, 0.2, 0.3]
    assert mock_create.call_count == 5  # 3 batch attempts + 2 individual


def test_create_embeddings_batch_complete_failure_returns_zeros():
    """Test that create_embeddings_batch returns zero embeddings when everything fails."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    texts = ["Text 1", "Text 2"]

    # Act
    with patch('openai.embeddings.create') as mock_create:
        with patch('time.sleep'):
            # All calls fail
            mock_create.side_effect = Exception("Complete failure")

            embeddings = service.create_embeddings_batch(texts)

    # Assert
    assert len(embeddings) == 2
    assert embeddings[0] == [0.0] * 1536
    assert embeddings[1] == [0.0] * 1536


def test_create_embedding_single_text():
    """Test creating embedding for a single text."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    # Act
    with patch.object(service, 'create_embeddings_batch') as mock_batch:
        mock_batch.return_value = [[0.1, 0.2, 0.3]]

        embedding = service.create_embedding("Test text")

    # Assert
    assert embedding == [0.1, 0.2, 0.3]
    mock_batch.assert_called_once_with(["Test text"])


def test_create_embedding_handles_empty_batch_result():
    """Test that create_embedding handles empty batch result gracefully."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    # Act
    with patch.object(service, 'create_embeddings_batch') as mock_batch:
        mock_batch.return_value = []

        embedding = service.create_embedding("Test text")

    # Assert
    assert embedding == [0.0] * 1536


def test_generate_contextual_embedding_success():
    """Test successful contextual embedding generation."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    mock_settings.use_contextual_embeddings = True
    service = EmbeddingService(mock_settings)

    full_document = "This is a long document about Python programming. It covers functions and classes."
    chunk = "It covers functions and classes."

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Enhanced context: This chunk discusses Python programming concepts including functions and classes within a broader programming tutorial."))]

    # Act
    with patch('openai.chat.completions.create', return_value=mock_response) as mock_chat:
        contextual_text, success = service.generate_contextual_embedding(full_document, chunk)

    # Assert
    assert success is True
    assert "Enhanced context:" in contextual_text
    mock_chat.assert_called_once()


def test_generate_contextual_embedding_disabled():
    """Test that contextual embedding is skipped when disabled."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    mock_settings.use_contextual_embeddings = False
    service = EmbeddingService(mock_settings)

    full_document = "This is a document."
    chunk = "This is a chunk."

    # Act
    contextual_text, success = service.generate_contextual_embedding(full_document, chunk)

    # Assert
    assert success is False
    assert contextual_text == chunk  # Returns original chunk


def test_generate_contextual_embedding_handles_api_error():
    """Test that contextual embedding handles API errors gracefully."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    mock_settings.use_contextual_embeddings = True
    service = EmbeddingService(mock_settings)

    full_document = "This is a document."
    chunk = "This is a chunk."

    # Act
    with patch('openai.chat.completions.create') as mock_chat:
        mock_chat.side_effect = Exception("API Error")

        contextual_text, success = service.generate_contextual_embedding(full_document, chunk)

    # Assert
    assert success is False
    assert contextual_text == chunk  # Falls back to original chunk


def test_process_chunks_with_context_parallel():
    """Test parallel processing of chunks with contextual embeddings."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    mock_settings.use_contextual_embeddings = True  # Enable contextual embeddings
    service = EmbeddingService(mock_settings)

    chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
    url_to_full_document = {
        "url1": "Full document 1",
        "url2": "Full document 2",
        "url3": "Full document 3"
    }
    urls = ["url1", "url2", "url3"]

    # Act
    with patch.object(service, 'generate_contextual_embedding') as mock_contextual:
        mock_contextual.return_value = ("Enhanced chunk", True)

        results = service.process_chunks_with_context(chunks, urls, url_to_full_document)

    # Assert
    assert len(results) == 3
    assert all(result == "Enhanced chunk" for result in results)
    assert mock_contextual.call_count == 3


def test_enhance_query_for_code_search():
    """Test query enhancement for code example search."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    query = "authentication function"

    # Act
    enhanced_query = service.enhance_query_for_code_search(query)

    # Assert
    assert "Code example for authentication function" in enhanced_query
    assert "Summary: Example code showing authentication function" in enhanced_query


def test_validate_embeddings_filters_invalid():
    """Test that validate_embeddings filters out invalid embeddings."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    embeddings = [
        [0.1, 0.2, 0.3],  # Valid
        [0.0, 0.0, 0.0],  # Invalid (all zeros)
        [],               # Invalid (empty)
        [0.4, 0.5, 0.6],  # Valid
        None              # Invalid (None)
    ]
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

    # Act
    with patch.object(service, 'create_embedding') as mock_single:
        mock_single.return_value = [0.7, 0.8, 0.9]  # Fallback embedding

        valid_embeddings = service.validate_embeddings(embeddings, texts)

    # Assert
    assert len(valid_embeddings) == 5
    assert valid_embeddings[0] == [0.1, 0.2, 0.3]  # Original valid
    assert valid_embeddings[1] == [0.7, 0.8, 0.9]  # Replaced invalid
    assert valid_embeddings[2] == [0.7, 0.8, 0.9]  # Replaced invalid
    assert valid_embeddings[3] == [0.4, 0.5, 0.6]  # Original valid
    assert valid_embeddings[4] == [0.7, 0.8, 0.9]  # Replaced invalid
    assert mock_single.call_count == 3  # Called for 3 invalid embeddings


def test_embedding_service_uses_correct_model_from_settings():
    """Test that EmbeddingService uses the correct embedding model from settings."""
    from src.services.embedding import EmbeddingService

    # Arrange
    mock_settings = create_mock_settings()
    service = EmbeddingService(mock_settings)

    texts = ["Test text"]
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    # Act
    with patch('openai.embeddings.create', return_value=mock_response) as mock_create:
        service.create_embeddings_batch(texts)

    # Assert
    mock_create.assert_called_once_with(
        model="text-embedding-3-small",  # Should use the configured model
        input=texts
    )
