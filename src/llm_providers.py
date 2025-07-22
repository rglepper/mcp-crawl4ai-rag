"""LLM provider abstractions for flexible model support."""

import os
import json
import subprocess
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import openai
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If completion fails
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name for identification."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available for use."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self):
        """Initialize OpenAI provider."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "openai"
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a chat completion using OpenAI API."""
        if not self.is_available():
            raise RuntimeError("OpenAI API key not configured")
        
        try:
            # Prepare completion arguments
            completion_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                completion_args["max_tokens"] = max_tokens
                
            # Add any additional kwargs
            completion_args.update(kwargs)
            
            # Call OpenAI API
            response = openai.chat.completions.create(**completion_args)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class ClaudeCodeProvider(LLMProvider):
    """Claude Code CLI provider implementation."""
    
    def __init__(self):
        """Initialize Claude Code provider."""
        self.claude_path = self._find_claude_cli()
        self.default_model = os.getenv("CLAUDE_CODE_MODEL", "claude-code/sonnet")
    
    def _find_claude_cli(self) -> Optional[Path]:
        """Find the claude CLI executable."""
        # Check if claude is in PATH
        claude_in_path = shutil.which("claude")
        if claude_in_path:
            return Path(claude_in_path)
        
        # Check common installation locations
        common_locations = [
            Path.home() / ".claude" / "bin" / "claude",
            Path.home() / "node_modules" / ".bin" / "claude",
            Path.home() / ".yarn" / "bin" / "claude",
            Path("/usr/local/bin/claude"),
            Path("/opt/claude/bin/claude"),
        ]
        
        for location in common_locations:
            if location.exists() and location.is_file():
                return location
        
        return None
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "claude-code"
    
    def is_available(self) -> bool:
        """Check if Claude Code CLI is available."""
        return self.claude_path is not None
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a chat completion using Claude Code CLI."""
        if not self.is_available():
            raise RuntimeError("Claude Code CLI not found. Please install claude-code.")
        
        # Use provided model or default
        model_to_use = model or self.default_model
        
        # Format messages for CLI input
        formatted_messages = self._format_messages_for_cli(messages)
        
        # Build command
        cmd = [
            str(self.claude_path),
            "--model", model_to_use,
            "--no-interactive",  # Non-interactive mode
            "--json-output",     # Get structured output
        ]
        
        # Add temperature if not default
        if temperature != 0.3:
            cmd.extend(["--temperature", str(temperature)])
        
        # Add max tokens if specified
        if max_tokens:
            cmd.extend(["--max-tokens", str(max_tokens)])
        
        try:
            # Run the command with the formatted messages as input
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=formatted_messages)
            
            if process.returncode != 0:
                error_msg = f"Claude Code CLI error: {stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Parse JSON output
            try:
                result = json.loads(stdout)
                return result.get("content", "").strip()
            except json.JSONDecodeError:
                # Fallback to plain text if not JSON
                return stdout.strip()
                
        except Exception as e:
            logger.error(f"Claude Code CLI execution error: {e}")
            raise
    
    def _format_messages_for_cli(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for CLI input."""
        # Claude Code CLI expects a specific format
        # System messages are prefixed, user/assistant messages follow
        formatted_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        # Add final "Assistant:" to prompt for response
        formatted_parts.append("Assistant:")
        
        return "\n\n".join(formatted_parts)


def get_llm_provider() -> LLMProvider:
    """
    Get the configured LLM provider.
    
    Returns the appropriate provider based on environment configuration.
    Falls back to OpenAI if Claude Code is selected but not available.
    """
    provider_name = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider_name == "claude-code":
        provider = ClaudeCodeProvider()
        if provider.is_available():
            logger.info(f"Using {provider.name} provider")
            return provider
        else:
            logger.warning(
                "Claude Code CLI not found, falling back to OpenAI. "
                "Install claude-code CLI to use Claude models."
            )
            provider_name = "openai"
    
    if provider_name == "openai":
        provider = OpenAIProvider()
        if not provider.is_available():
            raise RuntimeError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY environment variable."
            )
        logger.info(f"Using {provider.name} provider")
        return provider
    
    raise ValueError(f"Unknown LLM provider: {provider_name}")