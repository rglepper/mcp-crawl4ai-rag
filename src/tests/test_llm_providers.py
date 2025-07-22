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
        from src.llm_providers import ClaudeCodeProvider
        
        provider = ClaudeCodeProvider()
        assert provider.name == "claude-code"
    
    def test_claude_code_provider_finds_claude_cli_in_path(self):
        """Test that provider can find claude CLI in system PATH."""
        from src.llm_providers import ClaudeCodeProvider
        
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"
            
            provider = ClaudeCodeProvider()
            assert provider.is_available() is True
            assert str(provider.claude_path) == "/usr/local/bin/claude"
    
    def test_claude_code_provider_finds_claude_cli_in_common_locations(self):
        """Test that provider checks common installation locations."""
        from src.llm_providers import ClaudeCodeProvider
        from pathlib import Path
        
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Not in PATH
            
            # Mock the specific path we want to find
            test_path = Path.home() / ".claude" / "bin" / "claude"
            
            # Mock Path operations
            original_exists = Path.exists
            original_is_file = Path.is_file
            
            def mock_exists(self):
                if self == test_path:
                    return True
                return False
            
            def mock_is_file(self):
                if self == test_path:
                    return True
                return False
            
            with patch.object(Path, "exists", mock_exists):
                with patch.object(Path, "is_file", mock_is_file):
                    provider = ClaudeCodeProvider()
                    assert provider.is_available() is True
                    assert str(provider.claude_path) == str(test_path)
    
    def test_claude_code_provider_is_not_available_without_cli(self):
        """Test that provider is not available when CLI is not found."""
        from src.llm_providers import ClaudeCodeProvider
        from pathlib import Path
        
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            
            with patch.object(Path, "exists") as mock_exists:
                mock_exists.return_value = False
                
                provider = ClaudeCodeProvider()
                assert provider.is_available() is False
                assert provider.claude_path is None
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_chat_completion_success(self):
        """Test successful chat completion with Claude Code provider."""
        from src.llm_providers import ClaudeCodeProvider
        
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"
            
            provider = ClaudeCodeProvider()
            
            # Mock subprocess
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.communicate.return_value = (
                    '{"content": "Hello from Claude!"}',
                    ""
                )
                mock_process.returncode = 0
                mock_popen.return_value = mock_process
                
                messages = [{"role": "user", "content": "Hello"}]
                result = await provider.chat_completion(
                    messages=messages,
                    model="claude-code/opus"
                )
                
                assert result == "Hello from Claude!"
                
                # Verify command construction
                call_args = mock_popen.call_args[0][0]
                assert "/usr/local/bin/claude" in call_args
                assert "--model" in call_args
                assert "claude-code/opus" in call_args
                assert "--no-interactive" in call_args
                assert "--json-output" in call_args
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_formats_messages_correctly(self):
        """Test that messages are formatted correctly for CLI."""
        from src.llm_providers import ClaudeCodeProvider
        
        provider = ClaudeCodeProvider()
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = provider._format_messages_for_cli(messages)
        
        expected = "System: You are helpful\n\nHuman: Hello\n\nAssistant: Hi there\n\nHuman: How are you?\n\nAssistant:"
        assert formatted == expected
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_handles_cli_errors(self):
        """Test error handling for CLI failures."""
        from src.llm_providers import ClaudeCodeProvider
        
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"
            
            provider = ClaudeCodeProvider()
            
            # Mock subprocess error
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.communicate.return_value = ("", "Error: Invalid model")
                mock_process.returncode = 1
                mock_popen.return_value = mock_process
                
                with pytest.raises(RuntimeError) as exc_info:
                    await provider.chat_completion(
                        messages=[{"role": "user", "content": "Hello"}],
                        model="invalid-model"
                    )
                
                assert "Claude Code CLI error" in str(exc_info.value)
                assert "Invalid model" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_claude_code_provider_respects_temperature_and_max_tokens(self):
        """Test that provider passes temperature and max_tokens correctly."""
        from src.llm_providers import ClaudeCodeProvider
        
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"
            
            provider = ClaudeCodeProvider()
            
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.communicate.return_value = ('{"content": "Response"}', "")
                mock_process.returncode = 0
                mock_popen.return_value = mock_process
                
                await provider.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="claude-code/sonnet",
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Verify command includes temperature and max_tokens
                call_args = mock_popen.call_args[0][0]
                assert "--temperature" in call_args
                assert "0.7" in call_args
                assert "--max-tokens" in call_args
                assert "500" in call_args


class TestProviderFactory:
    """Test cases for the provider factory/configuration."""
    
    def test_factory_returns_openai_provider_by_default(self):
        """Test that factory returns OpenAI provider when no config is set."""
        from src.llm_providers import get_llm_provider, OpenAIProvider
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Remove LLM_PROVIDER if set
            os.environ.pop("LLM_PROVIDER", None)
            
            provider = get_llm_provider()
            assert isinstance(provider, OpenAIProvider)
            assert provider.name == "openai"
    
    def test_factory_returns_claude_code_provider_when_configured(self):
        """Test that factory returns Claude Code provider when configured."""
        from src.llm_providers import get_llm_provider, ClaudeCodeProvider
        
        with patch.dict(os.environ, {"LLM_PROVIDER": "claude-code"}):
            with patch("shutil.which") as mock_which:
                mock_which.return_value = "/usr/local/bin/claude"
                
                provider = get_llm_provider()
                assert isinstance(provider, ClaudeCodeProvider)
                assert provider.name == "claude-code"
    
    def test_factory_falls_back_to_openai_if_claude_not_available(self):
        """Test fallback behavior when Claude Code is not available."""
        from src.llm_providers import get_llm_provider, OpenAIProvider
        
        with patch.dict(os.environ, {"LLM_PROVIDER": "claude-code", "OPENAI_API_KEY": "test-key"}):
            with patch("shutil.which") as mock_which:
                mock_which.return_value = None  # Claude not found
                
                # Mock Path operations to ensure claude is not found
                from pathlib import Path
                with patch.object(Path, "exists", return_value=False):
                    # Capture logs to verify warning
                    with patch("src.llm_providers.logger") as mock_logger:
                        provider = get_llm_provider()
                        
                        # Should fall back to OpenAI
                        assert isinstance(provider, OpenAIProvider)
                        assert provider.name == "openai"
                        
                        # Should log warning about fallback
                        mock_logger.warning.assert_called_once()
                        warning_msg = mock_logger.warning.call_args[0][0]
                        assert "Claude Code CLI not found" in warning_msg
                        assert "falling back to OpenAI" in warning_msg
    
    def test_factory_raises_error_for_unknown_provider(self):
        """Test that factory raises error for unknown provider names."""
        from src.llm_providers import get_llm_provider
        
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown-provider"}):
            with pytest.raises(ValueError) as exc_info:
                get_llm_provider()
            
            assert "Unknown LLM provider: unknown-provider" in str(exc_info.value)
    
    def test_factory_raises_error_if_openai_not_available(self):
        """Test that factory raises error if OpenAI API key is not set."""
        from src.llm_providers import get_llm_provider
        
        with patch.dict(os.environ, {}, clear=True):
            # No API key set
            with pytest.raises(RuntimeError) as exc_info:
                get_llm_provider()
            
            assert "OpenAI API key not configured" in str(exc_info.value)