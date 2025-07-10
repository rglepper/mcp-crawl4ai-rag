"""
Configuration management for the Crawl4AI MCP server.

This module centralizes all environment variable handling and provides
type-safe configuration using Pydantic Settings.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with validation.

    All configuration is loaded from environment variables with sensible defaults.
    Required variables will raise ValidationError if not provided.
    """

    # MCP Server Configuration
    host: str = Field(default="0.0.0.0", description="Host to bind the MCP server to")
    port: int = Field(default=8051, ge=1, le=65535, description="Port for the MCP server")

    # OpenAI Configuration - REQUIRED
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings and LLM calls")
    model_choice: str = Field(default="gpt-4o-mini", description="LLM model for summaries and contextual embeddings")

    # Supabase Configuration - REQUIRED
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_key: str = Field(..., description="Supabase service role key")

    # Neo4j Configuration - OPTIONAL (for knowledge graph features)
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password")

    # Feature Flags - Control optional functionality
    use_contextual_embeddings: bool = Field(default=False, description="Enable contextual embeddings")
    use_hybrid_search: bool = Field(default=False, description="Enable hybrid search combining vector and keyword")
    use_reranking: bool = Field(default=False, description="Enable cross-encoder reranking")
    enable_knowledge_graph: bool = Field(default=False, alias="USE_KNOWLEDGE_GRAPH", description="Enable Neo4j knowledge graph features")

    @field_validator('port')
    @classmethod
    def validate_port_range(cls, v):
        """Validate port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v

    @field_validator('use_contextual_embeddings', 'use_hybrid_search', 'use_reranking', 'enable_knowledge_graph', mode='before')
    @classmethod
    def parse_boolean_strings(cls, v):
        """Parse string boolean values from environment variables."""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return v

    model_config = ConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
        populate_by_name=True  # Allow both field names and aliases
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are loaded only once per application run.
    This is the recommended way to access settings throughout the application.

    Returns:
        Settings: Validated settings instance
    """
    return Settings()
