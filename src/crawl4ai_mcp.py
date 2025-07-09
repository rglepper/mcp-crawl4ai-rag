"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Also includes AI hallucination detection and repository parsing tools using Neo4j knowledge graphs.
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import subprocess
import shutil
import concurrent.futures
import sys

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# Add knowledge_graphs folder to path for importing knowledge graph modules
knowledge_graphs_path = Path(__file__).resolve().parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))

from utils import (
    get_supabase_client,
    add_documents_to_supabase,
    search_documents,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    search_code_examples
)

# Import knowledge graph modules
from knowledge_graph_validator import KnowledgeGraphValidator
from parse_repo_into_neo4j import DirectNeo4jExtractor, Neo4jCodeAnalyzer
from ai_script_analyzer import AIScriptAnalyzer
from hallucination_reporter import HallucinationReporter

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Helper functions for Neo4j validation and error handling
def validate_neo4j_connection() -> bool:
    """Check if Neo4j environment variables are configured."""
    return all([
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    ])

def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j connection errors for user-friendly messages."""
    error_str = str(error).lower()
    if "authentication" in error_str or "unauthorized" in error_str:
        return "Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD."
    elif "connection" in error_str or "refused" in error_str or "timeout" in error_str:
        return "Cannot connect to Neo4j. Check NEO4J_URI and ensure Neo4j is running."
    elif "database" in error_str:
        return "Neo4j database error. Check if the database exists and is accessible."
    else:
        return f"Neo4j error: {str(error)}"

def validate_script_path(script_path: str) -> Dict[str, Any]:
    """Validate script path and return error info if invalid."""
    if not script_path or not isinstance(script_path, str):
        return {"valid": False, "error": "Script path is required"}

    if not os.path.exists(script_path):
        return {"valid": False, "error": f"Script not found: {script_path}"}

    if not script_path.endswith('.py'):
        return {"valid": False, "error": "Only Python (.py) files are supported"}

    try:
        # Check if file is readable
        with open(script_path, 'r', encoding='utf-8') as f:
            f.read(1)  # Read first character to test
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Cannot read script file: {str(e)}"}

def validate_github_url(repo_url: str) -> Dict[str, Any]:
    """Validate GitHub repository URL."""
    if not repo_url or not isinstance(repo_url, str):
        return {"valid": False, "error": "Repository URL is required"}

    repo_url = repo_url.strip()

    # Basic GitHub URL validation
    if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
        return {"valid": False, "error": "Please provide a valid GitHub repository URL"}

    # Check URL format
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        return {"valid": False, "error": "Repository URL must start with https:// or git@"}

    return {"valid": True, "repo_name": repo_url.split('/')[-1].replace('.git', '')}

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None       # DirectNeo4jExtractor when available

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    # Initialize Supabase client
    supabase_client = get_supabase_client()

    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None

    # Initialize Neo4j components if configured and enabled
    knowledge_validator = None
    repo_extractor = None

    # Check if knowledge graph functionality is enabled
    knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"

    if knowledge_graph_enabled:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                print("Initializing knowledge graph components...")

                # Initialize knowledge graph validator
                knowledge_validator = KnowledgeGraphValidator(neo4j_uri, neo4j_user, neo4j_password)
                await knowledge_validator.initialize()
                print("✓ Knowledge graph validator initialized")

                # Initialize repository extractor
                repo_extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
                await repo_extractor.initialize()
                print("✓ Repository extractor initialized")

            except Exception as e:
                print(f"Failed to initialize Neo4j components: {format_neo4j_error(e)}")
                knowledge_validator = None
                repo_extractor = None
        else:
            print("Neo4j credentials not configured - knowledge graph tools will be unavailable")
    else:
        print("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor
        )
    finally:
        # Clean up all components
        await crawler.__aexit__(None, None, None)
        if knowledge_validator:
            try:
                await knowledge_validator.close()
                print("✓ Knowledge graph validator closed")
            except Exception as e:
                print(f"Error closing knowledge validator: {e}")
        if repo_extractor:
            try:
                await repo_extractor.close()
                print("✓ Repository extractor closed")
            except Exception as e:
                print(f"Error closing repository extractor: {e}")

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.

    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content

    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results

    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]

        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]

        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)

        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.

    Args:
        url: URL to check

    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

async def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.

    Args:
        sitemap_url: URL of the sitemap

    Returns:
        List of URLs found in the sitemap
    """
    print(f"📋 Fetching sitemap from: {sitemap_url}")
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        print(f"✅ Sitemap fetched successfully (status: {resp.status_code})")
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
            print(f"✅ Parsed {len(urls)} URLs from sitemap")
        except Exception as e:
            print(f"❌ Error parsing sitemap XML: {e}")
    else:
        print(f"❌ Failed to fetch sitemap (status: {resp.status_code})")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.

    Args:
        chunk: Markdown chunk

    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (code, context_before, context_after)

    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path

            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)

            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0

            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                # Accumulate word count
                total_word_count += meta.get("word_count", 0)

            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}

            # Update source information FIRST (before inserting documents)
            source_summary = extract_source_summary(source_id, result.markdown[:5000])  # Use first 5000 chars for summary
            update_source_info(supabase_client, source_id, source_summary, total_word_count)

            # Add documentation chunks to Supabase (AFTER source exists)
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)

            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            if extract_code_examples:
                code_blocks = extract_code_blocks(result.markdown)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []

                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after'])
                                        for block in code_blocks]

                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))

                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)

                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)

                    # Add code examples to Supabase
                    add_code_examples_to_supabase(
                        supabase_client,
                        code_urls,
                        code_chunk_numbers,
                        code_examples,
                        code_summaries,
                        code_metadatas
                    )

            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": len(code_blocks) if code_blocks else 0,
                "content_length": len(result.markdown),
                "total_word_count": total_word_count,
                "source_id": source_id,
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.

    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth

    All crawled content is chunked and stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)

    Returns:
        JSON string with crawl summary and storage information
    """
    print(f"🎯 FUNCTION ENTRY: smart_crawl_url called with URL: {url}")
    try:
        print(f"🚀 Starting smart_crawl_url for: {url}")
        print(f"📊 Parameters: max_depth={max_depth}, max_concurrent={max_concurrent}, chunk_size={chunk_size}")

        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        print("✅ Got crawler and supabase client from context")

        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None

        print(f"🔍 Detecting URL type for: {url}")
        if is_txt(url):
            print("📄 Detected: Text file")
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            print("🗺️ Detected: Sitemap")
            # For sitemaps, extract URLs and crawl in parallel
            print("📋 Parsing sitemap...")
            sitemap_urls = await parse_sitemap(url)
            if not sitemap_urls:
                print("❌ No URLs found in sitemap")
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            print(f"✅ Found {len(sitemap_urls)} URLs in sitemap")
            print("🚀 Starting batch crawl...")
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            print("🌐 Detected: Regular webpage - using recursive crawling")
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"

        print(f"📊 Crawl completed. Found {len(crawl_results)} results")
        if not crawl_results:
            print("❌ No content found during crawl")
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)

        print("🔄 Processing results and preparing for storage...")
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0

        # Track sources and their content
        source_content_map = {}
        source_word_counts = {}

        # Process documentation chunks
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

            # Extract source_id
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path

            # Store content for source summary generation
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                source_word_counts[source_id] = 0

            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                # Accumulate word count
                source_word_counts[source_id] += meta.get("word_count", 0)

                chunk_count += 1

        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']

        # Update source information for each unique source FIRST (before inserting documents)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
            source_summaries = list(executor.map(lambda args: extract_source_summary(args[0], args[1]), source_summary_args))

        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            update_source_info(supabase_client, source_id, summary, word_count)

        # Add documentation chunks to Supabase (AFTER sources exist)
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=batch_size)

        # Extract and process code examples from all documents only if enabled
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        if extract_code_examples_enabled:
            all_code_blocks = []
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []

            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                code_blocks = extract_code_blocks(md)

                if code_blocks:
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after'])
                                        for block in code_blocks]

                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))

                    # Prepare code example data
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path

                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(source_url)
                        code_chunk_numbers.append(len(code_examples))  # Use global code example index
                        code_examples.append(block['code'])
                        code_summaries.append(summary)

                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)

            # Add all code examples to Supabase
            if code_examples:
                add_code_examples_to_supabase(
                    supabase_client,
                    code_urls,
                    code_chunk_numbers,
                    code_examples,
                    code_summaries,
                    code_metadatas,
                    batch_size=batch_size
                )

        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "code_examples_stored": len(code_examples),
            "sources_updated": len(source_content_map),
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def ingest_local_directory(ctx: Context, directory_path: str, source_name: str = None, file_extensions: str = ".md,.txt,.markdown", recursive: bool = True, chunk_size: int = 5000) -> str:
    """
    Ingest markdown and text files from a local filesystem directory into the knowledge base.

    This tool reads local markdown/text files, processes them with the same pipeline as web crawling
    (chunking, embedding, code extraction), and stores them in Supabase for semantic search.

    Args:
        ctx: The MCP server provided context
        directory_path: Absolute path to the directory containing markdown files
        source_name: Optional custom source name (defaults to directory name)
        file_extensions: Comma-separated file extensions to process (default: ".md,.txt,.markdown")
        recursive: Whether to search subdirectories recursively (default: True)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)

    Returns:
        JSON string with ingestion summary and storage information
    """
    try:
        import os
        import glob
        from pathlib import Path

        print(f"🚀 Starting local directory ingestion: {directory_path}")

        # Get the supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Validate directory path
        if not os.path.exists(directory_path):
            return json.dumps({
                "success": False,
                "error": f"Directory not found: {directory_path}"
            }, indent=2)

        if not os.path.isdir(directory_path):
            return json.dumps({
                "success": False,
                "error": f"Path is not a directory: {directory_path}"
            }, indent=2)

        # Parse file extensions
        extensions = [ext.strip() for ext in file_extensions.split(',')]
        print(f"📄 Looking for files with extensions: {extensions}")

        # Find all matching files
        files_found = []
        directory_path = Path(directory_path)

        for ext in extensions:
            if recursive:
                pattern = f"**/*{ext}"
                files_found.extend(directory_path.glob(pattern))
            else:
                pattern = f"*{ext}"
                files_found.extend(directory_path.glob(pattern))

        if not files_found:
            return json.dumps({
                "success": False,
                "error": f"No files found with extensions {extensions} in {directory_path}"
            }, indent=2)

        print(f"✅ Found {len(files_found)} files to process")

        # Determine source_id
        if source_name:
            source_id = source_name
        else:
            source_id = f"local:{directory_path.name}"

        print(f"📊 Using source_id: {source_id}")

        # Process files
        all_urls = []
        all_chunk_numbers = []
        all_contents = []
        all_metadatas = []
        url_to_full_document = {}
        total_word_count = 0
        files_processed = 0

        # Code examples storage (for agentic RAG)
        code_urls = []
        code_chunk_numbers = []
        code_examples = []
        code_summaries = []
        code_metadatas = []

        for file_path in files_found:
            try:
                print(f"📖 Processing: {file_path}")

                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    print(f"⚠️ Skipping empty file: {file_path}")
                    continue

                # Create a pseudo-URL for the file
                relative_path = file_path.relative_to(directory_path)
                file_url = f"file://{source_id}/{relative_path}"

                # Chunk the content
                chunks = smart_chunk_markdown(content, chunk_size=chunk_size)
                print(f"  ✂️ Created {len(chunks)} chunks")

                # Store full document for contextual embeddings
                url_to_full_document[file_url] = content

                # Process each chunk
                for i, chunk in enumerate(chunks):
                    all_urls.append(file_url)
                    all_chunk_numbers.append(i)
                    all_contents.append(chunk)

                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = file_url
                    meta["source"] = source_id
                    meta["crawl_time"] = "local_directory_ingest"
                    meta["crawl_type"] = "local_directory"
                    meta["file_path"] = str(file_path)
                    meta["relative_path"] = str(relative_path)
                    all_metadatas.append(meta)

                    # Accumulate word count
                    total_word_count += meta.get("word_count", 0)

                # Extract code examples if agentic RAG is enabled
                use_agentic_rag = os.getenv("USE_AGENTIC_RAG", "false") == "true"
                if use_agentic_rag:
                    code_blocks = extract_code_blocks(content)
                    print(f"  💻 Found {len(code_blocks)} code examples")

                    for j, code_block in enumerate(code_blocks):
                        code_urls.append(file_url)
                        code_chunk_numbers.append(j)
                        code_examples.append(code_block['code'])

                        # Generate summary for code example
                        summary = generate_code_example_summary(
                            code_block['code'],
                            code_block['context_before'],
                            code_block['context_after']
                        )
                        code_summaries.append(summary)

                        # Create metadata for code example
                        code_meta = {
                            "language": code_block['language'],
                            "file_path": str(file_path),
                            "relative_path": str(relative_path),
                            "source": source_id,
                            "crawl_type": "local_directory",
                            "code_example": True
                        }
                        code_metadatas.append(code_meta)

                files_processed += 1

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue

        if not all_contents:
            return json.dumps({
                "success": False,
                "error": "No content was successfully processed from any files"
            }, indent=2)

        print(f"🔄 Processing {len(all_contents)} chunks for storage...")

        # Generate source summary
        sample_content = "\n\n".join(all_contents[:3])  # Use first 3 chunks for summary
        source_summary = extract_source_summary(source_id, sample_content[:5000])

        # Update source information
        update_source_info(supabase_client, source_id, source_summary, total_word_count)
        print(f"✅ Updated source info for: {source_id}")

        # Add documentation chunks to Supabase
        add_documents_to_supabase(supabase_client, all_urls, all_chunk_numbers, all_contents, all_metadatas, url_to_full_document)
        print(f"✅ Stored {len(all_contents)} documentation chunks")

        # Add code examples if any were found
        code_examples_stored = 0
        if code_examples:
            add_code_examples_to_supabase(supabase_client, code_urls, code_chunk_numbers, code_examples, code_summaries, code_metadatas)
            code_examples_stored = len(code_examples)
            print(f"✅ Stored {code_examples_stored} code examples")

        return json.dumps({
            "success": True,
            "source_id": source_id,
            "directory_path": str(directory_path),
            "files_processed": files_processed,
            "total_files_found": len(files_found),
            "chunks_stored": len(all_contents),
            "code_examples_stored": code_examples_stored,
            "total_word_count": total_word_count,
            "crawl_type": "local_directory",
            "message": f"Successfully ingested {files_processed} files from local directory"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Directory ingestion failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def cleanup_source(ctx: Context, source_id: str, confirm: bool = False) -> str:
    """
    Remove a source and all its associated data from the knowledge base.

    This tool removes all crawled pages, code examples, and source metadata
    for a specified source_id. Use with caution as this action is irreversible.

    Args:
        ctx: The MCP server provided context
        source_id: The source ID to remove (e.g., 'example.com')
        confirm: Must be set to True to actually perform the deletion

    Returns:
        JSON string with cleanup results
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        if not confirm:
            # Just show what would be deleted
            pages_count = supabase_client.table('crawled_pages').select('id', count='exact').eq('source_id', source_id).execute()
            code_count = supabase_client.table('code_examples').select('id', count='exact').eq('source_id', source_id).execute()
            source_exists = supabase_client.table('sources').select('source_id').eq('source_id', source_id).execute()

            return json.dumps({
                "success": True,
                "action": "preview",
                "source_id": source_id,
                "would_delete": {
                    "crawled_pages": pages_count.count,
                    "code_examples": code_count.count,
                    "source_record": len(source_exists.data) > 0
                },
                "message": "Set confirm=True to actually perform the deletion"
            }, indent=2)

        # Perform actual deletion
        pages_result = supabase_client.table('crawled_pages').delete().eq('source_id', source_id).execute()
        code_result = supabase_client.table('code_examples').delete().eq('source_id', source_id).execute()
        source_result = supabase_client.table('sources').delete().eq('source_id', source_id).execute()

        return json.dumps({
            "success": True,
            "action": "deleted",
            "source_id": source_id,
            "deleted": {
                "crawled_pages": len(pages_result.data) if pages_result.data else 0,
                "code_examples": len(code_result.data) if code_result.data else 0,
                "source_record": len(source_result.data) if source_result.data else 0
            },
            "message": f"Successfully removed all data for source: {source_id}"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "source_id": source_id,
            "error": f"Cleanup failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def analyze_crawl_types(ctx: Context) -> str:
    """
    Analyze the crawl types used for each source to understand crawling scope.

    This tool shows whether each source was crawled as:
    - 'sitemap': Full site crawl from sitemap (many pages)
    - 'webpage': Recursive crawl from a starting page (multiple pages)
    - 'text_file': Single file crawl (one page)

    Returns:
        JSON string with crawl type analysis for each source
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Get crawl type statistics for each source
        result = supabase_client.rpc('get_crawl_type_stats').execute()

        if not result.data:
            # Fallback: query directly if RPC doesn't exist
            sources_result = supabase_client.table('sources').select('source_id').execute()
            crawl_analysis = []

            for source in sources_result.data:
                source_id = source['source_id']

                # Get crawl types and counts for this source
                pages_result = supabase_client.table('crawled_pages')\
                    .select('metadata')\
                    .eq('source_id', source_id)\
                    .execute()

                crawl_types = {}
                total_pages = len(pages_result.data)

                for page in pages_result.data:
                    metadata = page.get('metadata', {})
                    crawl_type = metadata.get('crawl_type', 'unknown')
                    crawl_types[crawl_type] = crawl_types.get(crawl_type, 0) + 1

                # Determine primary crawl type
                primary_type = max(crawl_types.items(), key=lambda x: x[1])[0] if crawl_types else 'unknown'

                crawl_analysis.append({
                    'source_id': source_id,
                    'total_pages': total_pages,
                    'primary_crawl_type': primary_type,
                    'crawl_type_breakdown': crawl_types,
                    'scope': 'single_page' if primary_type == 'text_file' or total_pages == 1 else 'multi_page'
                })

            return json.dumps({
                'success': True,
                'analysis': crawl_analysis,
                'summary': {
                    'total_sources': len(crawl_analysis),
                    'single_page_sources': len([s for s in crawl_analysis if s['scope'] == 'single_page']),
                    'multi_page_sources': len([s for s in crawl_analysis if s['scope'] == 'multi_page'])
                }
            }, indent=2)

        return json.dumps({
            'success': True,
            'analysis': result.data
        }, indent=2)

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }, indent=2)

@mcp.tool()
async def get_knowledge_base_guide(ctx: Context) -> str:
    """
    Get a concise guide to available knowledge base resources with usage recommendations.

    This tool provides LLMs with a simple list of available resources including:
    - Documentation sources in Supabase
    - Code repositories in the knowledge graph
    - Brief description and when to use each resource

    Returns:
        JSON string with concise resource guide and usage instructions
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Get available sources
        sources_result = supabase_client.table('sources').select('*').execute()
        sources = sources_result.data

        # Get knowledge graph repositories if available
        knowledge_graph_repos = []
        try:
            from neo4j import GraphDatabase
            import os

            neo4j_uri = os.getenv('NEO4J_URI')
            neo4j_user = os.getenv('NEO4J_USER')
            neo4j_password = os.getenv('NEO4J_PASSWORD')

            if neo4j_uri and neo4j_user and neo4j_password:
                driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                with driver.session() as session:
                    result = session.run("MATCH (r:Repository) RETURN r.name as name")
                    knowledge_graph_repos = [record["name"] for record in result]
                driver.close()
        except Exception:
            pass  # Neo4j not available or not configured

        # Create concise resource guide
        resources = []

        # Add documentation sources from Supabase
        for source in sources:
            source_id = source['source_id']
            summary = source.get('summary', 'No summary available')

            # Create concise description based on source
            description = ""
            use_for = ""

            if 'pydantic' in source_id.lower():
                description = "Pydantic documentation - Data validation and parsing library"
                use_for = "Data validation, type hints, API models, configuration management"
            elif 'qdrant' in source_id.lower():
                description = "Qdrant documentation - Vector database for similarity search"
                use_for = "Vector operations, similarity search, collection management"
            elif 'neo4j' in source_id.lower():
                description = "Neo4j documentation - Graph database platform"
                use_for = "Graph queries, Cypher language, relationship modeling"
            elif 'graphiti' in source_id.lower() or 'zep' in source_id.lower():
                description = "Graphiti documentation - Temporal knowledge graphs for AI"
                use_for = "AI agent memory, temporal graphs, knowledge representation"
            elif 'crawl4ai' in source_id.lower():
                description = "Crawl4AI documentation - LLM-friendly web scraping"
                use_for = "Web scraping, content extraction, documentation gathering"
            elif source_id.startswith('local:'):
                description = f"Local documentation - {source_id.replace('local:', '')}"
                use_for = "Project-specific docs, internal knowledge, custom implementations"
            elif any(domain in source_id for domain in ['docs.', 'help.', 'api.', 'guide.']):
                description = f"Official documentation - {source_id}"
                use_for = "API reference, official examples, best practices"
            else:
                description = f"Web content - {source_id}"
                use_for = "General information, tutorials, community content"

            resources.append({
                "resource_id": source_id,
                "type": "documentation",
                "description": description,
                "use_for": use_for,
                "search_tool": "perform_rag_query() or search_code_examples()"
            })

        # Add knowledge graph repositories
        for repo_name in knowledge_graph_repos:
            resources.append({
                "resource_id": repo_name,
                "type": "code_repository",
                "description": f"Code repository - {repo_name} (parsed into knowledge graph)",
                "use_for": "AI hallucination detection, code structure exploration, method validation",
                "search_tool": "query_knowledge_graph() or check_ai_script_hallucinations()"
            })

        # Create the guide
        guide = {
            "summary": f"Available resources: {len(sources)} documentation sources, {len(knowledge_graph_repos)} code repositories",
            "resources": resources,
            "usage_notes": {
                "documentation_search": "Use perform_rag_query('your question', source='source_id') for targeted searches",
                "code_search": "Use search_code_examples('pattern') for implementation examples",
                "graph_exploration": "Use query_knowledge_graph('repos') to start exploring code repositories",
                "hallucination_check": "Use check_ai_script_hallucinations('/path/to/script.py') to validate AI-generated code"
            }
        }

        return json.dumps(guide, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to generate knowledge base guide: {str(e)}"
        }, indent=2)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Query the sources table directly
        result = supabase_client.from_('sources')\
            .select('*')\
            .order('source_id')\
            .execute()

        # Format the sources with their details
        sources = []
        if result.data:
            for source in result.data:
                sources.append({
                    "source_id": source.get("source_id"),
                    "summary": source.get("summary"),
                    "total_words": source.get("total_words"),
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at")
                })

        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search

            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )

            # 2. Get keyword search results using ILIKE
            keyword_query = supabase_client.from_('crawled_pages')\
                .select('id, url, chunk_number, content, metadata, source_id')\
                .ilike('content', f'%{query}%')

            # Apply source filter if provided
            if source and source.strip():
                keyword_query = keyword_query.eq('source_id', source)

            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []

            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []

            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break

            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])

            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])

            # Use combined results
            results = combined_results[:match_count]

        else:
            # Standard vector search only
            results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )

        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")

        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)

        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def search_code_examples(ctx: Context, query: str, source_id: str = None, match_count: int = 5) -> str:
    """
    Search for code examples relevant to the query.

    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps({
            "success": False,
            "error": "Code example extraction is disabled. Perform a normal RAG search."
        }, indent=2)

    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}

        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search

            # Import the search function from utils
            from utils import search_code_examples as search_code_examples_impl

            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )

            # 2. Get keyword search results using ILIKE on both content and summary
            keyword_query = supabase_client.from_('code_examples')\
                .select('id, url, chunk_number, content, summary, metadata, source_id')\
                .or_(f'content.ilike.%{query}%,summary.ilike.%{query}%')

            # Apply source filter if provided
            if source_id and source_id.strip():
                keyword_query = keyword_query.eq('source_id', source_id)

            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []

            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []

            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break

            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])

            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'summary': kr['summary'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])

            # Use combined results
            results = combined_results[:match_count]

        else:
            # Standard vector search only
            from utils import search_code_examples as search_code_examples_impl

            results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )

        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")

        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)

        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source_id,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def check_ai_script_hallucinations(ctx: Context, script_path: str) -> str:
    """
    Check an AI-generated Python script for hallucinations using the knowledge graph.

    This tool analyzes a Python script for potential AI hallucinations by validating
    imports, method calls, class instantiations, and function calls against a Neo4j
    knowledge graph containing real repository data.

    The tool performs comprehensive analysis including:
    - Import validation against known repositories
    - Method call validation on classes from the knowledge graph
    - Class instantiation parameter validation
    - Function call parameter validation
    - Attribute access validation

    Args:
        ctx: The MCP server provided context
        script_path: Absolute path to the Python script to analyze

    Returns:
        JSON string with hallucination detection results, confidence scores, and recommendations
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)

        # Get the knowledge validator from context
        knowledge_validator = ctx.request_context.lifespan_context.knowledge_validator

        if not knowledge_validator:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph validator not available. Check Neo4j configuration in environment variables."
            }, indent=2)

        # Validate script path
        validation = validate_script_path(script_path)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "script_path": script_path,
                "error": validation["error"]
            }, indent=2)

        # Step 1: Analyze script structure using AST
        analyzer = AIScriptAnalyzer()
        analysis_result = analyzer.analyze_script(script_path)

        if analysis_result.errors:
            print(f"Analysis warnings for {script_path}: {analysis_result.errors}")

        # Step 2: Validate against knowledge graph
        validation_result = await knowledge_validator.validate_script(analysis_result)

        # Step 3: Generate comprehensive report
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation_result)

        # Format response with comprehensive information
        return json.dumps({
            "success": True,
            "script_path": script_path,
            "overall_confidence": validation_result.overall_confidence,
            "validation_summary": {
                "total_validations": report["validation_summary"]["total_validations"],
                "valid_count": report["validation_summary"]["valid_count"],
                "invalid_count": report["validation_summary"]["invalid_count"],
                "uncertain_count": report["validation_summary"]["uncertain_count"],
                "not_found_count": report["validation_summary"]["not_found_count"],
                "hallucination_rate": report["validation_summary"]["hallucination_rate"]
            },
            "hallucinations_detected": report["hallucinations_detected"],
            "recommendations": report["recommendations"],
            "analysis_metadata": {
                "total_imports": report["analysis_metadata"]["total_imports"],
                "total_classes": report["analysis_metadata"]["total_classes"],
                "total_methods": report["analysis_metadata"]["total_methods"],
                "total_attributes": report["analysis_metadata"]["total_attributes"],
                "total_functions": report["analysis_metadata"]["total_functions"]
            },
            "libraries_analyzed": report.get("libraries_analyzed", [])
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "script_path": script_path,
            "error": f"Analysis failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def query_knowledge_graph(ctx: Context, command: str) -> str:
    """
    Query and explore the Neo4j knowledge graph containing repository data.

    This tool provides comprehensive access to the knowledge graph for exploring repositories,
    classes, methods, functions, and their relationships. Perfect for understanding what data
    is available for hallucination detection and debugging validation results.

    **⚠️ IMPORTANT: Always start with the `repos` command first!**
    Before using any other commands, run `repos` to see what repositories are available
    in your knowledge graph. This will help you understand what data you can explore.

    ## Available Commands:

    **Repository Commands:**
    - `repos` - **START HERE!** List all repositories in the knowledge graph
    - `explore <repo_name>` - Get detailed overview of a specific repository

    **Class Commands:**
    - `classes` - List all classes across all repositories (limited to 20)
    - `classes <repo_name>` - List classes in a specific repository
    - `class <class_name>` - Get detailed information about a specific class including methods and attributes

    **Method Commands:**
    - `method <method_name>` - Search for methods by name across all classes
    - `method <method_name> <class_name>` - Search for a method within a specific class

    **Custom Query:**
    - `query <cypher_query>` - Execute a custom Cypher query (results limited to 20 records)

    ## Knowledge Graph Schema:

    **Node Types:**
    - Repository: `(r:Repository {name: string})`
    - File: `(f:File {path: string, module_name: string})`
    - Class: `(c:Class {name: string, full_name: string})`
    - Method: `(m:Method {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
    - Function: `(func:Function {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
    - Attribute: `(a:Attribute {name: string, type: string})`

    **Relationships:**
    - `(r:Repository)-[:CONTAINS]->(f:File)`
    - `(f:File)-[:DEFINES]->(c:Class)`
    - `(c:Class)-[:HAS_METHOD]->(m:Method)`
    - `(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)`
    - `(f:File)-[:DEFINES]->(func:Function)`

    ## Example Workflow:
    ```
    1. repos                                    # See what repositories are available
    2. explore pydantic-ai                      # Explore a specific repository
    3. classes pydantic-ai                      # List classes in that repository
    4. class Agent                              # Explore the Agent class
    5. method run_stream                        # Search for run_stream method
    6. method __init__ Agent                    # Find Agent constructor
    7. query "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) WHERE m.name = 'run' RETURN c.name, m.name LIMIT 5"
    ```

    Args:
        ctx: The MCP server provided context
        command: Command string to execute (see available commands above)

    Returns:
        JSON string with query results, statistics, and metadata
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)

        # Get Neo4j driver from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        if not repo_extractor or not repo_extractor.driver:
            return json.dumps({
                "success": False,
                "error": "Neo4j connection not available. Check Neo4j configuration in environment variables."
            }, indent=2)

        # Parse command
        command = command.strip()
        if not command:
            return json.dumps({
                "success": False,
                "command": "",
                "error": "Command cannot be empty. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
            }, indent=2)

        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        async with repo_extractor.driver.session() as session:
            # Route to appropriate handler
            if cmd == "repos":
                return await _handle_repos_command(session, command)
            elif cmd == "explore":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Repository name required. Usage: explore <repo_name>"
                    }, indent=2)
                return await _handle_explore_command(session, command, args[0])
            elif cmd == "classes":
                repo_name = args[0] if args else None
                return await _handle_classes_command(session, command, repo_name)
            elif cmd == "class":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Class name required. Usage: class <class_name>"
                    }, indent=2)
                return await _handle_class_command(session, command, args[0])
            elif cmd == "method":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Method name required. Usage: method <method_name> [class_name]"
                    }, indent=2)
                method_name = args[0]
                class_name = args[1] if len(args) > 1 else None
                return await _handle_method_command(session, command, method_name, class_name)
            elif cmd == "query":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Cypher query required. Usage: query <cypher_query>"
                    }, indent=2)
                cypher_query = " ".join(args)
                return await _handle_query_command(session, command, cypher_query)
            else:
                return json.dumps({
                    "success": False,
                    "command": command,
                    "error": f"Unknown command '{cmd}'. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
                }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Query execution failed: {str(e)}"
        }, indent=2)


async def _handle_repos_command(session, command: str) -> str:
    """Handle 'repos' command - list all repositories"""
    query = "MATCH (r:Repository) RETURN r.name as name ORDER BY r.name"
    result = await session.run(query)

    repos = []
    async for record in result:
        repos.append(record['name'])

    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "repositories": repos
        },
        "metadata": {
            "total_results": len(repos),
            "limited": False
        }
    }, indent=2)


async def _handle_explore_command(session, command: str, repo_name: str) -> str:
    """Handle 'explore <repo>' command - get repository overview"""
    # Check if repository exists
    repo_check_query = "MATCH (r:Repository {name: $repo_name}) RETURN r.name as name"
    result = await session.run(repo_check_query, repo_name=repo_name)
    repo_record = await result.single()

    if not repo_record:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Repository '{repo_name}' not found in knowledge graph"
        }, indent=2)

    # Get file count
    files_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
    RETURN count(f) as file_count
    """
    result = await session.run(files_query, repo_name=repo_name)
    file_count = (await result.single())['file_count']

    # Get class count
    classes_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
    RETURN count(DISTINCT c) as class_count
    """
    result = await session.run(classes_query, repo_name=repo_name)
    class_count = (await result.single())['class_count']

    # Get function count
    functions_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
    RETURN count(DISTINCT func) as function_count
    """
    result = await session.run(functions_query, repo_name=repo_name)
    function_count = (await result.single())['function_count']

    # Get method count
    methods_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
    RETURN count(DISTINCT m) as method_count
    """
    result = await session.run(methods_query, repo_name=repo_name)
    method_count = (await result.single())['method_count']

    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "repository": repo_name,
            "statistics": {
                "files": file_count,
                "classes": class_count,
                "functions": function_count,
                "methods": method_count
            }
        },
        "metadata": {
            "total_results": 1,
            "limited": False
        }
    }, indent=2)


async def _handle_classes_command(session, command: str, repo_name: str = None) -> str:
    """Handle 'classes [repo]' command - list classes"""
    limit = 20

    if repo_name:
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, repo_name=repo_name, limit=limit)
    else:
        query = """
        MATCH (c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, limit=limit)

    classes = []
    async for record in result:
        classes.append({
            'name': record['name'],
            'full_name': record['full_name']
        })

    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "classes": classes,
            "repository_filter": repo_name
        },
        "metadata": {
            "total_results": len(classes),
            "limited": len(classes) >= limit
        }
    }, indent=2)


async def _handle_class_command(session, command: str, class_name: str) -> str:
    """Handle 'class <name>' command - explore specific class"""
    # Find the class
    class_query = """
    MATCH (c:Class)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN c.name as name, c.full_name as full_name
    LIMIT 1
    """
    result = await session.run(class_query, class_name=class_name)
    class_record = await result.single()

    if not class_record:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Class '{class_name}' not found in knowledge graph"
        }, indent=2)

    actual_name = class_record['name']
    full_name = class_record['full_name']

    # Get methods
    methods_query = """
    MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed, m.return_type as return_type
    ORDER BY m.name
    """
    result = await session.run(methods_query, class_name=class_name)

    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record['params_detailed'] or record['params_list'] or []
        methods.append({
            'name': record['name'],
            'parameters': params_to_use,
            'return_type': record['return_type'] or 'Any'
        })

    # Get attributes
    attributes_query = """
    MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN a.name as name, a.type as type
    ORDER BY a.name
    """
    result = await session.run(attributes_query, class_name=class_name)

    attributes = []
    async for record in result:
        attributes.append({
            'name': record['name'],
            'type': record['type'] or 'Any'
        })

    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "class": {
                "name": actual_name,
                "full_name": full_name,
                "methods": methods,
                "attributes": attributes
            }
        },
        "metadata": {
            "total_results": 1,
            "methods_count": len(methods),
            "attributes_count": len(attributes),
            "limited": False
        }
    }, indent=2)


async def _handle_method_command(session, command: str, method_name: str, class_name: str = None) -> str:
    """Handle 'method <name> [class]' command - search for methods"""
    if class_name:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list,
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        """
        result = await session.run(query, class_name=class_name, method_name=method_name)
    else:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list,
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        ORDER BY c.name
        LIMIT 20
        """
        result = await session.run(query, method_name=method_name)

    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record['params_detailed'] or record['params_list'] or []
        methods.append({
            'class_name': record['class_name'],
            'class_full_name': record['class_full_name'],
            'method_name': record['method_name'],
            'parameters': params_to_use,
            'return_type': record['return_type'] or 'Any',
            'legacy_args': record['args'] or []
        })

    if not methods:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Method '{method_name}'" + (f" in class '{class_name}'" if class_name else "") + " not found"
        }, indent=2)

    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "methods": methods,
            "class_filter": class_name
        },
        "metadata": {
            "total_results": len(methods),
            "limited": len(methods) >= 20 and not class_name
        }
    }, indent=2)


async def _handle_query_command(session, command: str, cypher_query: str) -> str:
    """Handle 'query <cypher>' command - execute custom Cypher query"""
    try:
        # Execute the query with a limit to prevent overwhelming responses
        result = await session.run(cypher_query)

        records = []
        count = 0
        async for record in result:
            records.append(dict(record))
            count += 1
            if count >= 20:  # Limit results to prevent overwhelming responses
                break

        return json.dumps({
            "success": True,
            "command": command,
            "data": {
                "query": cypher_query,
                "results": records
            },
            "metadata": {
                "total_results": len(records),
                "limited": len(records) >= 20
            }
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Cypher query error: {str(e)}",
            "data": {
                "query": cypher_query
            }
        }, indent=2)


@mcp.tool()
async def analyze_repository_temporarily(ctx: Context, repo_url: str, focus_areas: str = None) -> str:
    """
    Temporarily analyze a GitHub repository for research and example purposes without storing in knowledge graph.

    This tool clones a repository, analyzes its Python code structure, and provides detailed information
    about classes, methods, functions, and patterns. Perfect for research when you want to understand
    how a codebase implements certain features or patterns to inform your own development.

    The repository is cloned to a temporary location and deleted after analysis - nothing is stored
    permanently in the knowledge graph.

    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
        focus_areas: Optional comma-separated keywords to focus analysis on (e.g., 'auth,api,database')

    Returns:
        JSON string with comprehensive repository analysis including classes, methods, functions,
        and code patterns relevant for research and example purposes
    """
    if not validate_neo4j_connection():
        return json.dumps({
            "success": False,
            "error": "Neo4j connection not configured. Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables."
        }, indent=2)

    try:
        # Validate repository URL
        validation = validate_github_url(repo_url)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "repo_url": repo_url,
                "error": validation["error"]
            }, indent=2)

        repo_name = validation["repo_name"]

        # Create timestamp for unique directory and file names
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create temporary analyzer (no Neo4j persistence)
        temp_analyzer = Neo4jCodeAnalyzer()

        # Use the same repos directory structure as existing code
        project_root = Path.cwd()
        knowledge_graphs_dir = project_root / "knowledge_graphs"
        temp_repo_dir = knowledge_graphs_dir / "repos" / f"temp_{repo_name}_{timestamp}"

        print(f"Temporarily analyzing repository: {repo_name}")

        # Clone repository to knowledge_graphs/repos directory
        print(f"Cloning repository to {temp_repo_dir}...")
        subprocess.run(['git', 'clone', '--depth', '1', repo_url, str(temp_repo_dir)],
                     check=True, capture_output=True)

        repo_path = Path(temp_repo_dir)

        # Get Python files
        python_files = []
        exclude_dirs = {
            'tests', 'test', '__pycache__', '.git', 'venv', 'env',
            'node_modules', 'build', 'dist', '.pytest_cache', 'docs',
            'examples', 'example', 'demo', 'benchmark'
        }

        for root, dirs, files in os.walk(temp_repo_dir):
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

                for file in files:
                    if file.endswith('.py') and not file.startswith('test_'):
                        file_path = Path(root) / file
                        if (file_path.stat().st_size < 500_000 and
                            file not in ['setup.py', 'conftest.py']):
                            python_files.append(file_path)

        print(f"Found {len(python_files)} Python files to analyze")

        # Identify project modules
        project_modules = set()
        for file_path in python_files:
            try:
                relative_path = str(file_path.relative_to(repo_path))
                module_parts = relative_path.replace('/', '.').replace('.py', '').split('.')
                if len(module_parts) > 0 and not module_parts[0].startswith('.'):
                    project_modules.add(module_parts[0])
            except ValueError:
                continue

        # Analyze files
        analysis_results = {
                "repo_name": repo_name,
                "repo_url": repo_url,
                "total_files": len(python_files),
                "project_modules": sorted(list(project_modules)),
                "modules": [],
                "summary": {
                    "total_classes": 0,
                    "total_methods": 0,
                    "total_functions": 0,
                    "key_patterns": []
                }
        }

        focus_keywords = []
        if focus_areas:
            focus_keywords = [kw.strip().lower() for kw in focus_areas.split(',')]

        for i, file_path in enumerate(python_files):
            if i % 20 == 0:
                print(f"Analyzing file {i+1}/{len(python_files)}: {file_path.name}")

            try:
                file_analysis = temp_analyzer.analyze_python_file(file_path, repo_path, project_modules)
                if file_analysis:
                    # Filter based on focus areas if specified
                    if focus_keywords:
                        file_content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                        if not any(keyword in file_content or keyword in file_analysis['module_name'].lower()
                                 for keyword in focus_keywords):
                            continue

                    analysis_results["modules"].append(file_analysis)
                    analysis_results["summary"]["total_classes"] += len(file_analysis["classes"])
                    analysis_results["summary"]["total_methods"] += sum(len(cls["methods"]) for cls in file_analysis["classes"])
                    analysis_results["summary"]["total_functions"] += len(file_analysis["functions"])

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue

        # Generate key patterns and insights
        all_class_names = []
        all_method_names = []
        all_function_names = []

        for module in analysis_results["modules"]:
            all_class_names.extend([cls["name"] for cls in module["classes"]])
            for cls in module["classes"]:
                all_method_names.extend([method["name"] for method in cls["methods"]])
            all_function_names.extend([func["name"] for func in module["functions"]])

        # Identify common patterns
        patterns = []
        if any("auth" in name.lower() for name in all_class_names + all_method_names + all_function_names):
            patterns.append("Authentication/Authorization patterns")
        if any("api" in name.lower() for name in all_class_names + all_method_names + all_function_names):
            patterns.append("API implementation patterns")
        if any("db" in name.lower() or "database" in name.lower() for name in all_class_names + all_method_names + all_function_names):
            patterns.append("Database interaction patterns")
        if any("test" in name.lower() for name in all_class_names + all_method_names + all_function_names):
            patterns.append("Testing patterns")
        if any("config" in name.lower() for name in all_class_names + all_method_names + all_function_names):
            patterns.append("Configuration management patterns")

        analysis_results["summary"]["key_patterns"] = patterns

        print(f"Temporary analysis completed for: {repo_name}")
        print(f"Found {analysis_results['summary']['total_classes']} classes, "
              f"{analysis_results['summary']['total_methods']} methods, "
              f"{analysis_results['summary']['total_functions']} functions")

        # Save analysis to knowledge_graphs temporary analysis directory
        project_root = Path.cwd()
        temp_analysis_dir = project_root / "knowledge_graphs" / "temp_analysis"
        temp_analysis_dir.mkdir(parents=True, exist_ok=True)

        # Create filename based on repo name and timestamp
        analysis_filename = f"{repo_name}_{timestamp}.json"
        analysis_file_path = temp_analysis_dir / analysis_filename

        # Save the analysis data
        analysis_data = {
            "success": True,
            "analysis_type": "temporary",
            "message": f"Successfully analyzed repository '{repo_name}' temporarily for research purposes",
            "data": analysis_results,
            "created_at": datetime.now().isoformat(),
            "usage_notes": [
                "This analysis is temporary and not stored in the knowledge graph",
                "Use this data to understand code patterns and implementation approaches",
                "Perfect for research when building similar features in your own projects",
                "Repository was automatically cleaned up after analysis"
            ]
        }

        with open(analysis_file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)

        return json.dumps({
            "success": True,
            "analysis_type": "temporary",
            "message": f"Successfully analyzed repository '{repo_name}' temporarily for research purposes",
            "analysis_file": str(analysis_file_path.relative_to(project_root)),
            "analysis_id": f"{repo_name}_{timestamp}",
            "data": analysis_results,
            "usage_notes": [
                "Analysis saved to knowledge_graphs/temp_analysis/ directory",
                "Use the analysis_id with search_temporary_analysis tool",
                "Files can be reused across different contexts and sessions",
                "Repository was automatically cleaned up after analysis"
            ]
        }, indent=2)

    except subprocess.CalledProcessError as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"Failed to clone repository: {e.stderr.decode() if e.stderr else str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"Temporary analysis failed: {str(e)}"
        }, indent=2)
    finally:
        # Clean up the temporary repository directory
        if 'temp_repo_dir' in locals() and temp_repo_dir.exists():
            print(f"Cleaning up temporary repository: {temp_repo_dir}")
            try:
                shutil.rmtree(temp_repo_dir)
                print("Cleanup completed")
            except Exception as e:
                print(f"Cleanup failed: {e}. Directory may remain at {temp_repo_dir}")

@mcp.tool()
async def search_temporary_analysis(ctx: Context, analysis_id: str, search_query: str, search_type: str = "all") -> str:
    """
    Search through temporary repository analysis data for specific patterns, classes, methods, or functions.

    This tool helps you find specific information within saved temporary analysis results.
    Perfect for research when you want to find examples of specific patterns or implementations.

    Args:
        ctx: The MCP server provided context
        analysis_id: Analysis ID from analyze_repository_temporarily tool (e.g., 'pydantic_20250109_143022')
        search_query: What to search for (e.g., 'auth', 'database', 'api', class names, method names)
        search_type: Type of search - 'all', 'classes', 'methods', 'functions', 'modules' (default: 'all')

    Returns:
        JSON string with filtered results matching the search criteria
    """
    try:
        # Find the analysis file
        project_root = Path.cwd()
        temp_analysis_dir = project_root / "knowledge_graphs" / "temp_analysis"

        # Look for the analysis file
        analysis_file = None
        if temp_analysis_dir.exists():
            for file_path in temp_analysis_dir.glob("*.json"):
                if file_path.stem == analysis_id:
                    analysis_file = file_path
                    break

        if not analysis_file or not analysis_file.exists():
            return json.dumps({
                "success": False,
                "error": f"Analysis file not found for ID: {analysis_id}. Use analyze_repository_temporarily first."
            }, indent=2)

        # Load the analysis data
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)

        if not analysis.get("success") or analysis.get("analysis_type") != "temporary":
            return json.dumps({
                "success": False,
                "error": "Invalid analysis file format."
            }, indent=2)

        repo_data = analysis["data"]
        search_lower = search_query.lower()
        results = {
            "repo_name": repo_data["repo_name"],
            "search_query": search_query,
            "search_type": search_type,
            "matches": {
                "classes": [],
                "methods": [],
                "functions": [],
                "modules": []
            },
            "summary": {
                "total_matches": 0,
                "classes_found": 0,
                "methods_found": 0,
                "functions_found": 0,
                "modules_found": 0
            }
        }

        # Search through modules
        for module in repo_data["modules"]:
            module_matches = False

            # Check if module name or path matches
            if (search_lower in module["module_name"].lower() or
                search_lower in module["file_path"].lower()):
                module_matches = True
                if search_type in ["all", "modules"]:
                    results["matches"]["modules"].append({
                        "module_name": module["module_name"],
                        "file_path": module["file_path"],
                        "classes_count": len(module["classes"]),
                        "functions_count": len(module["functions"]),
                        "reason": "Module name or path matches search query"
                    })
                    results["summary"]["modules_found"] += 1

            # Search classes
            for cls in module["classes"]:
                class_matches = False
                if search_lower in cls["name"].lower() or search_lower in cls["full_name"].lower():
                    class_matches = True
                    if search_type in ["all", "classes"]:
                        results["matches"]["classes"].append({
                            "name": cls["name"],
                            "full_name": cls["full_name"],
                            "module": module["module_name"],
                            "file_path": module["file_path"],
                            "methods_count": len(cls["methods"]),
                            "attributes_count": len(cls["attributes"]),
                            "methods": [m["name"] for m in cls["methods"]],
                            "reason": "Class name matches search query"
                        })
                        results["summary"]["classes_found"] += 1

                # Search methods within classes
                for method in cls["methods"]:
                    if (search_lower in method["name"].lower() or
                        any(search_lower in param.lower() for param in method.get("params_detailed", []))):
                        if search_type in ["all", "methods"]:
                            results["matches"]["methods"].append({
                                "name": method["name"],
                                "class_name": cls["name"],
                                "full_class_name": cls["full_name"],
                                "module": module["module_name"],
                                "file_path": module["file_path"],
                                "parameters": method.get("params_detailed", []),
                                "return_type": method.get("return_type", "Any"),
                                "reason": "Method name or parameters match search query"
                            })
                            results["summary"]["methods_found"] += 1

            # Search standalone functions
            for func in module["functions"]:
                if (search_lower in func["name"].lower() or
                    any(search_lower in param.lower() for param in func.get("params_detailed", []))):
                    if search_type in ["all", "functions"]:
                        results["matches"]["functions"].append({
                            "name": func["name"],
                            "full_name": func["full_name"],
                            "module": module["module_name"],
                            "file_path": module["file_path"],
                            "parameters": func.get("params_detailed", []),
                            "return_type": func.get("return_type", "Any"),
                            "reason": "Function name or parameters match search query"
                        })
                        results["summary"]["functions_found"] += 1

        # Calculate total matches
        results["summary"]["total_matches"] = (
            results["summary"]["classes_found"] +
            results["summary"]["methods_found"] +
            results["summary"]["functions_found"] +
            results["summary"]["modules_found"]
        )

        return json.dumps({
            "success": True,
            "message": f"Found {results['summary']['total_matches']} matches for '{search_query}' in {repo_data['repo_name']}",
            "data": results,
            "usage_notes": [
                "Use this data to understand specific implementation patterns",
                "Look at the file_path to examine the actual code",
                "Parameters and return types help understand method signatures",
                "Perfect for finding examples to guide your own implementation"
            ]
        }, indent=2)

    except json.JSONDecodeError:
        return json.dumps({
            "success": False,
            "error": "Invalid JSON in analysis_data parameter"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def list_temporary_analyses(ctx: Context) -> str:
    """
    List all available temporary repository analyses saved in the project.

    This tool shows all previously analyzed repositories that are available for searching.
    Useful to see what analyses are available before using search_temporary_analysis.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with list of available analyses and their metadata
    """
    try:
        project_root = Path.cwd()
        temp_analysis_dir = project_root / "knowledge_graphs" / "temp_analysis"

        if not temp_analysis_dir.exists():
            return json.dumps({
                "success": True,
                "analyses": [],
                "message": "No temporary analyses found. Use analyze_repository_temporarily to create some."
            }, indent=2)

        analyses = []
        for analysis_file in temp_analysis_dir.glob("*.json"):
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)

                if analysis_data.get("success") and analysis_data.get("analysis_type") == "temporary":
                    repo_data = analysis_data.get("data", {})
                    analyses.append({
                        "analysis_id": analysis_file.stem,
                        "repo_name": repo_data.get("repo_name", "unknown"),
                        "repo_url": repo_data.get("repo_url", "unknown"),
                        "created_at": analysis_data.get("created_at", "unknown"),
                        "total_files": repo_data.get("total_files", 0),
                        "total_classes": repo_data.get("summary", {}).get("total_classes", 0),
                        "total_methods": repo_data.get("summary", {}).get("total_methods", 0),
                        "total_functions": repo_data.get("summary", {}).get("total_functions", 0),
                        "key_patterns": repo_data.get("summary", {}).get("key_patterns", []),
                        "file_path": str(analysis_file.relative_to(project_root))
                    })
            except Exception as e:
                # Skip invalid files
                continue

        # Sort by creation time (newest first)
        analyses.sort(key=lambda x: x["created_at"], reverse=True)

        return json.dumps({
            "success": True,
            "analyses": analyses,
            "count": len(analyses),
            "message": f"Found {len(analyses)} temporary analyses",
            "usage_notes": [
                "Use the analysis_id with search_temporary_analysis tool",
                "Files are stored in knowledge_graphs/temp_analysis/ directory",
                "Analyses persist across sessions until manually deleted"
            ]
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list analyses: {str(e)}"
        }, indent=2)

@mcp.tool()
async def cleanup_temporary_analysis(ctx: Context, analysis_id: str = None, all_analyses: bool = False) -> str:
    """
    Remove temporary repository analysis files from the project.

    This tool helps clean up old analysis files to save disk space.

    Args:
        ctx: The MCP server provided context
        analysis_id: Specific analysis ID to remove (e.g., 'pydantic_20250109_143022')
        all_analyses: If True, removes all temporary analyses (default: False)

    Returns:
        JSON string with cleanup results
    """
    try:
        project_root = Path.cwd()
        temp_analysis_dir = project_root / "knowledge_graphs" / "temp_analysis"

        if not temp_analysis_dir.exists():
            return json.dumps({
                "success": True,
                "message": "No temporary analysis directory found - nothing to clean up"
            }, indent=2)

        removed_files = []

        if all_analyses:
            # Remove all analysis files
            for analysis_file in temp_analysis_dir.glob("*.json"):
                try:
                    analysis_file.unlink()
                    removed_files.append(analysis_file.stem)
                except Exception as e:
                    continue

            # Remove directory if empty
            try:
                if not any(temp_analysis_dir.iterdir()):
                    temp_analysis_dir.rmdir()
            except:
                pass

        elif analysis_id:
            # Remove specific analysis
            analysis_file = temp_analysis_dir / f"{analysis_id}.json"
            if analysis_file.exists():
                analysis_file.unlink()
                removed_files.append(analysis_id)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Analysis file not found: {analysis_id}"
                }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": "Must specify either analysis_id or set all_analyses=True"
            }, indent=2)

        return json.dumps({
            "success": True,
            "removed_analyses": removed_files,
            "count": len(removed_files),
            "message": f"Successfully removed {len(removed_files)} analysis file(s)"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Cleanup failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def parse_github_repository(ctx: Context, repo_url: str) -> str:
    """
    Parse a GitHub repository into the Neo4j knowledge graph.

    This tool clones a GitHub repository, analyzes its Python files, and stores
    the code structure (classes, methods, functions, imports) in Neo4j for use
    in hallucination detection. The tool:

    - Clones the repository to a temporary location
    - Analyzes Python files to extract code structure
    - Stores classes, methods, functions, and imports in Neo4j
    - Provides detailed statistics about the parsing results
    - Automatically handles module name detection for imports

    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')

    Returns:
        JSON string with parsing results, statistics, and repository information
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)

        # Get the repository extractor from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor

        if not repo_extractor:
            return json.dumps({
                "success": False,
                "error": "Repository extractor not available. Check Neo4j configuration in environment variables."
            }, indent=2)

        # Validate repository URL
        validation = validate_github_url(repo_url)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "repo_url": repo_url,
                "error": validation["error"]
            }, indent=2)

        repo_name = validation["repo_name"]

        # Parse the repository (this includes cloning, analysis, and Neo4j storage)
        print(f"Starting repository analysis for: {repo_name}")
        await repo_extractor.analyze_repository(repo_url)
        print(f"Repository analysis completed for: {repo_name}")

        # Query Neo4j for statistics about the parsed repository
        async with repo_extractor.driver.session() as session:
            # Get comprehensive repository statistics
            stats_query = """
            MATCH (r:Repository {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
            WITH r,
                 count(DISTINCT f) as files_count,
                 count(DISTINCT c) as classes_count,
                 count(DISTINCT m) as methods_count,
                 count(DISTINCT func) as functions_count,
                 count(DISTINCT a) as attributes_count

            // Get some sample module names
            OPTIONAL MATCH (r)-[:CONTAINS]->(sample_f:File)
            WITH r, files_count, classes_count, methods_count, functions_count, attributes_count,
                 collect(DISTINCT sample_f.module_name)[0..5] as sample_modules

            RETURN
                r.name as repo_name,
                files_count,
                classes_count,
                methods_count,
                functions_count,
                attributes_count,
                sample_modules
            """

            result = await session.run(stats_query, repo_name=repo_name)
            record = await result.single()

            if record:
                stats = {
                    "repository": record['repo_name'],
                    "files_processed": record['files_count'],
                    "classes_created": record['classes_count'],
                    "methods_created": record['methods_count'],
                    "functions_created": record['functions_count'],
                    "attributes_created": record['attributes_count'],
                    "sample_modules": record['sample_modules'] or []
                }
            else:
                return json.dumps({
                    "success": False,
                    "repo_url": repo_url,
                    "error": f"Repository '{repo_name}' not found in database after parsing"
                }, indent=2)

        return json.dumps({
            "success": True,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "message": f"Successfully parsed repository '{repo_name}' into knowledge graph",
            "statistics": stats,
            "ready_for_validation": True,
            "next_steps": [
                "Repository is now available for hallucination detection",
                f"Use check_ai_script_hallucinations to validate scripts against {repo_name}",
                "The knowledge graph contains classes, methods, and functions from this repository"
            ]
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"Repository parsing failed: {str(e)}"
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())