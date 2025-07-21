# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- `uv run src/mcp_server.py` - Run the MCP server directly
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

The codebase follows a vertical slice architecture organized into these main layers:

1. **MCP Server (`src/mcp_server.py`)** - FastMCP server setup with lifespan management
2. **Configuration (`src/config.py`)** - Centralized settings using Pydantic
3. **Models (`src/models.py`)** - Data models and context objects
4. **Services (`src/services/`)** - Business logic layer
5. **Tools (`src/tools/`)** - MCP tool implementations

### Service Layer

Services handle core business logic:
- `database.py` - Supabase vector database operations
- `web_crawling.py` - Web crawling and content extraction
- `embedding.py` - Text embedding and vectorization
- `rag_search.py` - RAG query processing and hybrid search
- `knowledge_graph.py` - Neo4j graph operations
- `source_management.py` - Source data management
- `temporary_analysis.py` - Temporary repository analysis
- `directory_ingestion.py` - Local directory content ingestion
- `graph_validator.py` - Knowledge graph validation
- `hallucination_detector.py` - AI hallucination detection
- `neo4j_parser.py` - Neo4j data parsing
- `report_generator.py` - Analysis report generation
- `script_analyzer.py` - Python script analysis

### Tool Layer

MCP tools expose functionality to AI agents:
- `crawling_tools.py` - Web crawling operations (`crawl_single_page`, `smart_crawl_url`, `ingest_local_directory`)
- `search_tools.py` - RAG search and code example retrieval (`perform_rag_query`, `search_code_examples`)
- `knowledge_graph_tools.py` - Graph database operations (`check_ai_script_hallucinations`, `query_knowledge_graph`, `parse_github_repository`)
- `source_management_tools.py` - Source management (`cleanup_source`, `analyze_crawl_types`, `get_knowledge_base_guide`, `get_available_sources`)
- `temporary_analysis_tools.py` - Temporary analysis operations (`analyze_repository_temporarily`, `search_temporary_analysis`, `list_temporary_analyses`, `cleanup_temporary_analysis`)

### Configuration System

The system uses environment variables for configuration with these key settings:
- RAG strategy toggles (`USE_CONTEXTUAL_EMBEDDINGS`, `USE_HYBRID_SEARCH`, `USE_AGENTIC_RAG`, `USE_RERANKING`, `USE_KNOWLEDGE_GRAPH`)
- Database connections (Supabase, Neo4j)
- API keys (OpenAI)
- Server settings (HOST, PORT, TRANSPORT)

### Context Management

The `Crawl4AIContext` model manages shared dependencies:
- AsyncWebCrawler instance
- Database service
- Embedding service
- Neo4j driver (optional)
- Cross-encoder for reranking (optional)

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
- Follow vertical slice architecture with tests next to code
  - One entry point @main.py
  - Shared files @src/config.py @src/db.py @src/models.py
  - Services @src/services/ handle business logic
  - Tools @src/tools/  are a thin wrapper that get exposed in @src/mcp_server.py as MCP tools
- Use dependency injection via context management
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
- Don't put business logic in tool files (keep tools as thin wrappers)
- Don't skip writing tests "to save time" (especially for knowledge graph features)
- Don't use synchronous Supabase calls in async functions
- Don't forget to validate all inputs with Pydantic models
- Don't ignore Neo4j connection failures (graceful degradation required)
- Don't leave temporary files uncleaned (repository analysis cleanup)
- Don't hardcode file paths (use Path objects and proper configuration)
- Don't mix knowledge graph logic with core RAG functionality
- Don't break the existing MCP tool interface (16 tools must work identically)
- Don't try to recreate the original architecture, use the current code organisation

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

## Entry Point Guidance

### Main Entry Point
- The only entry point should be @main.py