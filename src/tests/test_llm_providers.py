"""Tests for LLM provider abstractions."""

import os
import pytest
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch


class TestLLMProviderBase:
    """Test cases for the LLMProvider base class."""
    
    def test_llm_provider_is_abstract(self):
        """Test that LLMProvider cannot be instantiated directly."""
        from src.llm_providers import LLMProvider
        
        # Should raise TypeError when trying to instantiate abstract class
        with pytest.raises(TypeError) as exc_info:
            LLMProvider()
        
        # Verify the error mentions abstract methods
        assert "Can't instantiate abstract class" in str(exc_info.value)
    
    def test_llm_provider_requires_abstract_methods(self):
        """Test that LLMProvider subclasses must implement all abstract methods."""
        from src.llm_providers import LLMProvider
        
        # Create a partial implementation missing some methods
        class IncompleteProvider(LLMProvider):
            @property
            def name(self):
                return "incomplete"
            
            def is_available(self):
                return True
            # Missing chat_completion method
        
        # Should fail to instantiate
        with pytest.raises(TypeError) as exc_info:
            IncompleteProvider()
        
        assert "chat_completion" in str(exc_info.value)
    
    def test_llm_provider_with_all_methods_can_instantiate(self):
        """Test that a complete LLMProvider implementation can be instantiated."""
        from src.llm_providers import LLMProvider
        
        # Create a complete implementation
        class CompleteProvider(LLMProvider):
            @property
            def name(self):
                return "complete"
            
            def is_available(self):
                return True
                
            async def chat_completion(self, messages, model, temperature=0.3, max_tokens=None, **kwargs):
                return "test response"
        
        # Should instantiate successfully
        provider = CompleteProvider()
        assert provider.name == "complete"
        assert provider.is_available() is True


class TestOpenAIProvider:
    """Test cases for the OpenAI provider implementation."""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        return mock_response
    
    def test_openai_provider_name(self):
        """Test that OpenAI provider returns correct name."""
        from src.llm_providers import OpenAIProvider
        
        provider = OpenAIProvider()
        assert provider.name == "openai"
    
    def test_openai_provider_is_available_with_api_key(self):
        """Test that OpenAI provider is available when API key is set."""
        from src.llm_providers import OpenAIProvider
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            provider = OpenAIProvider()
            assert provider.is_available() is True
    
    def test_openai_provider_is_not_available_without_api_key(self):
        """Test that OpenAI provider is not available without API key."""
        from src.llm_providers import OpenAIProvider
        
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()
            assert provider.is_available() is False
    
    @pytest.mark.asyncio
    async def test_openai_provider_chat_completion_success(self, mock_openai_response):
        """Test successful chat completion with OpenAI provider."""
        from src.llm_providers import OpenAIProvider
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            provider = OpenAIProvider()
            
            # Mock the openai client
            with patch("openai.chat.completions.create") as mock_create:
                mock_create.return_value = mock_openai_response
                
                messages = [{"role": "user", "content": "Hello"}]
                result = await provider.chat_completion(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    temperature=0.5,
                    max_tokens=100
                )
                
                assert result == "Test response"
                mock_create.assert_called_once()
                
                # Verify call arguments
                call_args = mock_create.call_args[1]
                assert call_args["model"] == "gpt-3.5-turbo"
                assert call_args["messages"] == messages
                assert call_args["temperature"] == 0.5
                assert call_args["max_tokens"] == 100
    
    @pytest.mark.asyncio
    async def test_openai_provider_chat_completion_without_api_key(self):
        """Test chat completion fails without API key."""
        from src.llm_providers import OpenAIProvider
        
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()
            
            with pytest.raises(RuntimeError) as exc_info:
                await provider.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-3.5-turbo"
                )
            
            assert "OpenAI API key not configured" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_provider_chat_completion_error_handling(self):
        """Test error handling in OpenAI provider."""
        from src.llm_providers import OpenAIProvider
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            provider = OpenAIProvider()
            
            # Mock API error
            with patch("openai.chat.completions.create") as mock_create:
                mock_create.side_effect = Exception("API Error: Rate limit exceeded")
                
                with pytest.raises(Exception) as exc_info:
                    await provider.chat_completion(
                        messages=[{"role": "user", "content": "Hello"}],
                        model="gpt-3.5-turbo"
                    )
                
                assert "API Error: Rate limit exceeded" in str(exc_info.value)


class TestClaudeCodeProvider:
    """Test cases for the Claude Code CLI provider implementation."""
    
    def test_claude_code_provider_name(self):
        """Test that Claude Code provider returns correct name."""
        # Will implement after creating the provider
        pass
    
    def test_claude_code_provider_finds_claude_cli_in_path(self):
        """Test that provider can find claude CLI in system PATH."""
        # Will mock subprocess to check for claude command
        pass
    
    def test_claude_code_provider_finds_claude_cli_in_common_locations(self):
        """Test that provider checks common installation locations."""
        # Will verify it checks ~/.claude/bin/claude, etc.
        pass
    
    def test_claude_code_provider_is_not_available_without_cli(self):
        """Test that provider is not available when CLI is not found."""
        # Will verify it returns False when claude CLI is not found
        pass
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_chat_completion_success(self):
        """Test successful chat completion with Claude Code provider."""
        # Will mock subprocess call to claude CLI
        pass
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_formats_messages_correctly(self):
        """Test that messages are formatted correctly for CLI."""
        # Will verify message formatting for the CLI interface
        pass
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_handles_cli_errors(self):
        """Test error handling for CLI failures."""
        # Will verify proper error handling for subprocess errors
        pass
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_respects_model_parameter(self):
        """Test that provider uses the specified model."""
        # Will verify --model parameter is passed correctly
        pass


class TestProviderFactory:
    """Test cases for the provider factory/configuration."""
    
    def test_factory_returns_openai_provider_by_default(self):
        """Test that factory returns OpenAI provider when no config is set."""
        # Will verify default behavior
        pass
    
    def test_factory_returns_claude_code_provider_when_configured(self):
        """Test that factory returns Claude Code provider when configured."""
        # Will verify LLM_PROVIDER=claude-code works
        pass
    
    def test_factory_falls_back_to_openai_if_claude_not_available(self):
        """Test fallback behavior when Claude Code is not available."""
        # Will verify graceful fallback with warning
        pass
    
    def test_factory_raises_error_for_unknown_provider(self):
        """Test that factory raises error for unknown provider names."""
        # Will verify proper error handling
        pass