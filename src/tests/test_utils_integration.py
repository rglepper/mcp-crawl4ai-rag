"""Integration tests for utils.py refactoring to use LLM providers."""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any


class TestUtilsIntegrationWithProviders:
    """Integration tests for utils.py functions with LLM provider abstraction."""

    @pytest.mark.asyncio
    async def test_generate_contextual_embedding_uses_provider(self):
        """Test that generate_contextual_embedding uses the configured LLM provider."""
        # Import after mocking to avoid import-time configuration
        mock_provider = Mock()
        mock_provider.chat_completion = AsyncMock(return_value="Enhanced context for the chunk")
        mock_provider.name = "test-provider"

        with patch("src.utils.get_llm_provider", return_value=mock_provider):
            with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true", "MODEL_CHOICE": "test-model"}):
                from src.utils import generate_contextual_embedding
                
                full_document = "This is a long document with multiple sections."
                chunk = "This is a specific chunk from the document."
                
                result_text, used_llm = await generate_contextual_embedding(full_document, chunk)
                
                # Should have called the provider
                mock_provider.chat_completion.assert_called_once()
                call_args = mock_provider.chat_completion.call_args
                
                # Verify the messages structure
                messages = call_args[1]["messages"]
                assert len(messages) == 2
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert chunk in messages[1]["content"]
                
                # Verify model parameter
                assert call_args[1]["model"] == "test-model"
                
                # Verify result
                assert "Enhanced context for the chunk" in result_text
                assert used_llm is True

    @pytest.mark.asyncio 
    async def test_generate_contextual_embedding_handles_provider_error(self):
        """Test error handling when LLM provider fails."""
        mock_provider = Mock()
        mock_provider.chat_completion = AsyncMock(side_effect=Exception("API Error"))
        mock_provider.name = "test-provider"

        with patch("src.utils.get_llm_provider", return_value=mock_provider):
            with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true", "MODEL_CHOICE": "test-model"}):
                from src.utils import generate_contextual_embedding
                
                full_document = "This is a document."
                chunk = "This is a chunk."
                
                # Should fall back to original chunk when LLM fails
                result_text, used_llm = await generate_contextual_embedding(full_document, chunk)
                
                assert result_text == chunk
                assert used_llm is False

    @pytest.mark.asyncio
    async def test_generate_contextual_embedding_skips_when_disabled(self):
        """Test that contextual embeddings are skipped when disabled."""
        with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "false"}):
            from src.utils import generate_contextual_embedding
            
            full_document = "This is a document."
            chunk = "This is a chunk."
            
            result_text, used_llm = await generate_contextual_embedding(full_document, chunk)
            
            assert result_text == chunk
            assert used_llm is False

    @pytest.mark.asyncio
    async def test_generate_code_example_summary_uses_provider(self):
        """Test that generate_code_example_summary uses the configured LLM provider."""
        mock_provider = Mock()
        mock_provider.chat_completion = AsyncMock(return_value="Summary of the code example")
        mock_provider.name = "test-provider"

        with patch("src.utils.get_llm_provider", return_value=mock_provider):
            with patch.dict(os.environ, {"MODEL_CHOICE": "test-model"}):
                from src.utils import generate_code_example_summary
                
                code = "def hello(): return 'world'"
                context_before = "This is setup code"
                context_after = "This is cleanup code"
                
                result = await generate_code_example_summary(code, context_before, context_after)
                
                # Should have called the provider
                mock_provider.chat_completion.assert_called_once()
                call_args = mock_provider.chat_completion.call_args
                
                # Verify the prompt contains all parts
                messages = call_args[1]["messages"]
                user_message = messages[1]["content"]
                assert code in user_message
                assert context_before in user_message
                assert context_after in user_message
                
                assert result == "Summary of the code example"

    @pytest.mark.asyncio
    async def test_extract_source_summary_uses_provider(self):
        """Test that extract_source_summary uses the configured LLM provider."""
        mock_provider = Mock()
        mock_provider.chat_completion = AsyncMock(return_value="Concise summary of the content")
        mock_provider.name = "test-provider"

        with patch("src.utils.get_llm_provider", return_value=mock_provider):
            with patch.dict(os.environ, {"MODEL_CHOICE": "test-model"}):
                from src.utils import extract_source_summary
                
                source_id = "test-source-123"
                content = "This is a long piece of content that needs to be summarized."
                
                result = await extract_source_summary(source_id, content)
                
                # Should have called the provider
                mock_provider.chat_completion.assert_called_once()
                call_args = mock_provider.chat_completion.call_args
                
                # Verify the prompt
                messages = call_args[1]["messages"]
                user_message = messages[1]["content"]
                assert content in user_message
                assert "summarize" in user_message.lower()
                
                assert result == "Concise summary of the content"

    @pytest.mark.asyncio
    async def test_provider_switching_between_functions(self):
        """Test that different functions can use different providers based on configuration."""
        # First call with OpenAI provider
        openai_provider = Mock()
        openai_provider.chat_completion = AsyncMock(return_value="OpenAI response")
        openai_provider.name = "openai"
        
        # Second call with Claude Code provider
        claude_provider = Mock()
        claude_provider.chat_completion = AsyncMock(return_value="Claude response")
        claude_provider.name = "claude-code"

        with patch("src.utils.get_llm_provider") as mock_get_provider:
            # First call returns OpenAI
            mock_get_provider.return_value = openai_provider
            
            with patch.dict(os.environ, {"MODEL_CHOICE": "gpt-4"}):
                from src.utils import generate_code_example_summary
                
                result1 = await generate_code_example_summary("code", "before", "after")
                assert result1 == "OpenAI response"
                openai_provider.chat_completion.assert_called_once()
            
            # Second call returns Claude Code
            mock_get_provider.return_value = claude_provider
            
            with patch.dict(os.environ, {"MODEL_CHOICE": "claude-code/sonnet"}):
                from src.utils import extract_source_summary
                
                result2 = await extract_source_summary("source", "content")
                assert result2 == "Claude response" 
                claude_provider.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_model_choice(self):
        """Test that existing MODEL_CHOICE environment variable still works."""
        # The provider should still receive the MODEL_CHOICE value
        mock_provider = Mock()
        mock_provider.chat_completion = AsyncMock(return_value="Response")
        mock_provider.name = "test-provider"

        with patch("src.utils.get_llm_provider", return_value=mock_provider):
            with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true", "MODEL_CHOICE": "gpt-4-turbo"}):
                from src.utils import generate_contextual_embedding
                
                await generate_contextual_embedding("doc", "chunk")
                
                # Verify the model parameter is passed through
                call_args = mock_provider.chat_completion.call_args
                assert call_args[1]["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_provider_error_handling_maintains_functionality(self):
        """Test that provider errors don't break the overall functionality."""
        failing_provider = Mock()
        failing_provider.chat_completion = AsyncMock(side_effect=RuntimeError("Provider unavailable"))
        failing_provider.name = "failing-provider"

        with patch("src.utils.get_llm_provider", return_value=failing_provider):
            with patch.dict(os.environ, {"USE_CONTEXTUAL_EMBEDDINGS": "true", "MODEL_CHOICE": "test-model"}):
                from src.utils import generate_contextual_embedding
                
                # Should handle error gracefully
                result_text, used_llm = await generate_contextual_embedding("document", "chunk")
                
                # Should fall back to original chunk
                assert result_text == "chunk"
                assert used_llm is False