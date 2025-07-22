# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read .project/rules/workflow.md for any development tasks
Follow .project/rules/python-guidelines.md when writing Python code
Follow .project/rules/unit-testing.md when writing tests

## Commit Message Guidelines
- Write clear, descriptive commit messages using conventional commits format.
- Don't mention tests in the commit message, only in the body.
- Always explain the implementation (not the tests) in commit messages
- Describe what was added/changed
- Explain why you chose to implement it this way (what were your logical reasoning steps that lead you to make the decisions you made, and what options you considered)
- Detail how the implementation works (provide the logic behind the code)
- Reference the example in .project/rules/workflow.md

## Core Principles
IMPORTANT: You MUST follow these principles in all code changes and PRP generations:

### KISS (Keep It Simple, Stupid)
- Simplicity should be a key goal in design
- Choose straightforward solutions over complex ones whenever possible
- Simple solutions are easier to understand, maintain, and debug
### YAGNI (You Aren't Gonna Need It)
- Avoid building functionality on speculation
- Implement features only when they are needed, not when you anticipate they might be useful in the future
### Open/Closed Principle
- Software entities should be open for extension but closed for modification
- Design systems so that new functionality can be added with minimal changes to existing code

## Project Overview

This is a Python-based MCP (Model Context Protocol) server that provides web crawling and RAG (Retrieval-Augmented Generation) capabilities for AI agents and coding assistants. The server integrates Crawl4AI for web crawling, Supabase for vector storage, and Neo4j for knowledge graph functionality.

## Key Commands

### Development
- `uv run src/crawl4ai_mcp.py` - Run the MCP server directly
- `uv run python -m pytest` - Run all tests
- `uv run ruff check` - Run linting
- `uv run ruff format .` - Format code
- `uv run mypy src/` - Run type checking

### Docker
- `docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .` - Build Docker image
- `docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag` - Run with Docker

### Setup
- `uv pip install -e .` - Install in development mode
- `crawl4ai-setup` - Initialize Crawl4AI after installation

## Architecture

### Core Components

The codebase is organized as a single monolithic MCP server file with supporting utilities:

1. **MCP Server (`src/crawl4ai_mcp.py`)** - Main FastMCP server with all tool implementations
2. **LLM Providers (`src/llm_providers.py`)** - Abstraction layer for different LLM providers (OpenAI, Claude Code)
3. **Utilities (`src/utils.py`)** - Helper functions for embeddings, database operations, and search
4. **Knowledge Graph Modules (`knowledge_graphs/`)** - Neo4j integration and AI hallucination detection
5. **Runner Script (`run_mcp_server.py`)** - Python launcher for virtual environment management

### Main Server File Structure

The `src/crawl4ai_mcp.py` file contains:
- Configuration and environment setup
- Neo4j validation helpers
- Context dataclass for shared state
- All 16 MCP tool implementations in a single file
- FastMCP server initialization and lifespan management

### LLM Provider Abstraction

The `src/llm_providers.py` file provides:
- `LLMProvider` - Abstract base class for all LLM providers
- `OpenAIProvider` - OpenAI API integration with automatic retry logic
- `ClaudeCodeProvider` - Claude Code CLI integration with subprocess management
- `get_llm_provider()` - Factory function that returns the configured provider

### Utility Functions

The `src/utils.py` file provides:
- `get_supabase_client()` - Supabase client initialization
- `create_embeddings_batch()` / `create_embedding()` - OpenAI embedding generation (always uses OpenAI)
- `generate_contextual_embedding()` - LLM-enhanced chunk context using configured provider (async)
- `generate_code_example_summary()` - Code example summarization using configured provider (async)
- `extract_source_summary()` - Source content summarization using configured provider (async)
- `add_documents_to_supabase()` - Document storage with chunking
- `search_documents()` - Hybrid search implementation
- `extract_code_blocks()` / `add_code_examples_to_supabase()` - Code example extraction
- `search_code_examples()` - Code-specific search

### Knowledge Graph Components

Located in `knowledge_graphs/` directory:
- `parse_repo_into_neo4j.py` - GitHub repository parser
- `ai_hallucination_detector.py` - Hallucination detection logic
- `ai_script_analyzer.py` - Python script analysis
- `hallucination_reporter.py` - Report generation
- `knowledge_graph_validator.py` - Graph validation
- `query_knowledge_graph.py` - Interactive graph queries

### Configuration System

The system uses environment variables for configuration with these key settings:
- LLM provider selection (`LLM_PROVIDER`, `CLAUDE_CODE_MODEL`)
- RAG strategy toggles (`USE_CONTEXTUAL_EMBEDDINGS`, `USE_HYBRID_SEARCH`, `USE_AGENTIC_RAG`, `USE_RERANKING`, `USE_KNOWLEDGE_GRAPH`)
- Database connections (Supabase, Neo4j)
- API keys (OpenAI)
- Server settings (HOST, PORT, TRANSPORT)

### Context Management

The server uses a shared context pattern:
- AsyncWebCrawler instance for web crawling
- Supabase client for vector storage
- Neo4j driver for knowledge graph (optional)
- CrossEncoder for reranking (optional)

## Key Features

### RAG Strategies
The system supports multiple RAG enhancement strategies that can be enabled independently:
- **Contextual Embeddings** - LLM-enhanced chunk context
- **Hybrid Search** - Combined vector and keyword search
- **Agentic RAG** - Code example extraction and specialized search
- **Reranking** - Cross-encoder result reordering
- **Knowledge Graph** - AI hallucination detection via Neo4j

### Crawling Capabilities
- Smart URL detection (sitemaps, text files, regular pages)
- Recursive crawling with parallel processing
- Content chunking by headers and size
- Source filtering and management
- Local directory ingestion

### Knowledge Graph
- GitHub repository parsing into Neo4j
- Python code structure analysis (classes, methods, functions)
- AI hallucination detection for generated code
- Interactive graph querying

## Development Guidelines

### Code Organization
- The codebase uses a monolithic architecture with all tools in `src/crawl4ai_mcp.py`
- Utility functions are separated in `src/utils.py`
- Knowledge graph functionality is modularized in `knowledge_graphs/` directory
- Use the shared context pattern for dependency management
- Avoid circular imports between modules

### Testing Strategy
- Follow TDD workflow (Write test → Run test → Write code → Refactor)
- Use pytest with descriptive test names
- Mock external dependencies (databases, APIs)
- Test both success and failure scenarios
- Keep tests isolated and independent

### Code Style
- Use type hints for all function signatures
- Follow PEP8 with 100 character line limit
- Use Pydantic for data validation
- Format with `ruff format`, lint with `ruff check`
- Document public functions with Google-style docstrings

### Important Anti-Patterns to Avoid
- Don't move code without understanding dependencies (especially Neo4j/knowledge graph)
- Don't create circular imports between modules
- Don't refactor the monolithic structure without careful consideration
- Don't skip writing tests "to save time" (especially for knowledge graph features)
- Don't use synchronous Supabase calls in async functions
- Don't forget to validate all inputs with proper error handling
- Don't ignore Neo4j connection failures (graceful degradation required)
- Don't leave temporary files uncleaned (repository analysis cleanup)
- Don't hardcode file paths (use Path objects and proper configuration)
- Don't mix knowledge graph logic with core RAG functionality
- Don't break the existing MCP tool interface (16 tools must work identically)

## Database Schema

The system uses two main databases:
1. **Supabase** - Vector storage with tables for crawled content and code examples
2. **Neo4j** - Knowledge graph for repository structure and relationships **bolt://localhost:7689**

Refer to `crawled_pages.sql` for the Supabase schema setup.

## Available Tools (16 Total)

The system provides 16 MCP tools organized into categories:

### Core Tools (4)
- `crawl_single_page` - Crawl a single web page
- `smart_crawl_url` - Intelligently crawl websites based on URL type
- `get_available_sources` - Get list of available sources in database
- `perform_rag_query` - Search for relevant content using semantic search

### Conditional Tools (1)
- `search_code_examples` - Search for code examples (requires `USE_AGENTIC_RAG=true`)

### Knowledge Graph Tools (3)
- `parse_github_repository` - Parse GitHub repository into Neo4j (requires `USE_KNOWLEDGE_GRAPH=true`)
- `check_ai_script_hallucinations` - Analyze scripts for AI hallucinations (requires `USE_KNOWLEDGE_GRAPH=true`)
- `query_knowledge_graph` - Query Neo4j knowledge graph (requires `USE_KNOWLEDGE_GRAPH=true`)

### Source Management Tools (4)
- `cleanup_source` - Remove source data from database
- `analyze_crawl_types` - Analyze different crawl types
- `get_knowledge_base_guide` - Get knowledge base usage guide
- `get_available_sources` - Get available data sources

### Temporary Analysis Tools (4)
- `analyze_repository_temporarily` - Temporarily analyze repository
- `search_temporary_analysis` - Search temporary analysis results
- `list_temporary_analyses` - List all temporary analyses
- `cleanup_temporary_analysis` - Clean up temporary analysis data

### Local Directory Tools (1)
- `ingest_local_directory` - Ingest local directory content into database

## Entry Points

### Main Entry Points
- `src/crawl4ai_mcp.py` - Direct execution of the MCP server
- `run_mcp_server.py` - Wrapper script that ensures proper virtual environment usage
- `run_mcp_server.sh` - Shell script for Unix-like systems