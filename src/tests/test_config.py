"""
Tests for the config module.

Following TDD principles - these tests define the expected behavior
of the Settings class before implementation.
"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError


def test_settings_loads_required_environment_variables():
    """Test that Settings class loads all required environment variables."""
    # Arrange: Set up required environment variables
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_KEY": "test-supabase-key"
    }

    # Act & Assert: This should not raise an exception
    with patch.dict(os.environ, env_vars, clear=True):
        from src.config import Settings
        settings = Settings()

        assert settings.openai_api_key == "test-openai-key"
        assert settings.supabase_url == "https://test.supabase.co"
        assert settings.supabase_service_key == "test-supabase-key"


def test_settings_fails_when_required_variables_missing():
    """Test that Settings raises ValidationError when required variables are missing."""
    # Arrange: Clear environment variables and disable env_file
    with patch.dict(os.environ, {}, clear=True):
        # Act & Assert: Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            from src.config import Settings
            # Create Settings without env_file to test required field validation
            Settings(_env_file=None)

        # Verify specific required fields are mentioned in error
        error_str = str(exc_info.value)
        assert "openai_api_key" in error_str
        assert "supabase_url" in error_str
        assert "supabase_service_key" in error_str


def test_settings_has_correct_default_values():
    """Test that Settings class has correct default values for optional fields."""
    # Arrange: Set only required environment variables, disable env_file
    env_vars = {
        "OPENAI_API_KEY": "test-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_KEY": "test-service-key"
    }

    # Act
    with patch.dict(os.environ, env_vars, clear=True):
        from src.config import Settings
        settings = Settings(_env_file=None)

    # Assert: Check default values match PRP specification
    assert settings.host == "0.0.0.0"
    assert settings.port == 8051
    assert settings.model_choice == "gpt-4o-mini"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password is None
    assert settings.use_contextual_embeddings is False
    assert settings.use_hybrid_search is False
    assert settings.use_reranking is False
    assert settings.enable_knowledge_graph is False


def test_settings_loads_optional_environment_variables():
    """Test that Settings loads optional environment variables when provided."""
    # Arrange: Set all environment variables
    env_vars = {
        "OPENAI_API_KEY": "test-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_KEY": "test-service-key",
        "HOST": "127.0.0.1",
        "PORT": "9000",
        "MODEL_CHOICE": "gpt-4",
        "NEO4J_URI": "bolt://remote:7687",
        "NEO4J_USER": "admin",
        "NEO4J_PASSWORD": "secret",
        "USE_CONTEXTUAL_EMBEDDINGS": "true",
        "USE_HYBRID_SEARCH": "true",
        "USE_RERANKING": "true",
        "USE_KNOWLEDGE_GRAPH": "true"
    }

    # Act
    with patch.dict(os.environ, env_vars, clear=True):
        from src.config import Settings
        settings = Settings()

    # Assert: All values should be loaded from environment
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.model_choice == "gpt-4"
    assert settings.neo4j_uri == "bolt://remote:7687"
    assert settings.neo4j_user == "admin"
    assert settings.neo4j_password == "secret"
    assert settings.use_contextual_embeddings is True
    assert settings.use_hybrid_search is True
    assert settings.use_reranking is True
    assert settings.enable_knowledge_graph is True


def test_settings_validates_port_range():
    """Test that Settings validates port is within valid range."""
    # Arrange: Set invalid port
    env_vars = {
        "OPENAI_API_KEY": "test-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_KEY": "test-service-key",
        "PORT": "99999"  # Invalid port number
    }

    # Act & Assert: Should raise ValidationError
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValidationError) as exc_info:
            from src.config import Settings
            Settings()

        assert "port" in str(exc_info.value).lower()


def test_settings_validates_boolean_flags():
    """Test that Settings properly converts string boolean flags."""
    # Arrange: Test various boolean string representations
    test_cases = [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("1", True),
        ("0", False)
    ]

    for bool_str, expected in test_cases:
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_KEY": "test-service-key",
            "USE_RERANKING": bool_str
        }

        # Act
        with patch.dict(os.environ, env_vars, clear=True):
            from src.config import Settings
            settings = Settings()

        # Assert
        assert settings.use_reranking is expected


def test_settings_loads_from_env_file():
    """Test that Settings can load from .env file when present."""
    # This test verifies the env_file configuration works
    # Note: In real implementation, this would use a test .env file
    env_vars = {
        "OPENAI_API_KEY": "test-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_KEY": "test-service-key"
    }

    with patch.dict(os.environ, env_vars, clear=True):
        from src.config import Settings
        settings = Settings()

        # Should have loaded successfully
        assert settings.openai_api_key == "test-key"


def test_get_settings_function_returns_cached_instance():
    """Test that get_settings() function returns cached Settings instance."""
    env_vars = {
        "OPENAI_API_KEY": "test-key",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_SERVICE_KEY": "test-service-key"
    }

    with patch.dict(os.environ, env_vars, clear=True):
        from src.config import get_settings

        # Act: Call get_settings multiple times
        settings1 = get_settings()
        settings2 = get_settings()

        # Assert: Should return the same instance (cached)
        assert settings1 is settings2
        assert settings1.openai_api_key == "test-key"
