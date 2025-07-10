"""
Web crawling service for the Crawl4AI MCP server.

This service handles all web crawling operations including URL type detection,
sitemap parsing, content chunking, and various crawling strategies.
"""
import re
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

from src.config import Settings
from src.models import CrawlRequest, CrawlResult, CrawlType


class WebCrawlingService:
    """Service for handling web crawling operations."""
    
    def __init__(self, crawler: AsyncWebCrawler, settings: Settings):
        """
        Initialize the web crawling service.
        
        Args:
            crawler: AsyncWebCrawler instance
            settings: Application settings
        """
        self.crawler = crawler
        self.settings = settings
    
    def is_sitemap(self, url: str) -> bool:
        """
        Check if a URL is a sitemap.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is a sitemap, False otherwise
        """
        return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path
    
    def is_txt(self, url: str) -> bool:
        """
        Check if a URL is a text file.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is a text file, False otherwise
        """
        return url.endswith('.txt')
    
    async def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap and extract URLs.
        
        Args:
            sitemap_url: URL of the sitemap
            
        Returns:
            List of URLs found in the sitemap
        """
        print(f"📋 Fetching sitemap from: {sitemap_url}")
        try:
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
        except Exception as e:
            print(f"❌ Error fetching sitemap: {e}")
            return []
    
    def smart_chunk_markdown(self, text: str, chunk_size: int = 5000) -> List[str]:
        """
        Split text into chunks, respecting code blocks and paragraphs.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
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
            
            # If no good break point, just cut at chunk_size
            chunks.append(text[start:end].strip())
            start = end
        
        return [chunk for chunk in chunks if chunk]  # Remove empty chunks
    
    def extract_section_info(self, chunk: str) -> Dict[str, Any]:
        """
        Extract headers and stats from a chunk.
        
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
    
    async def crawl_markdown_file(self, url: str) -> List[Dict[str, Any]]:
        """
        Crawl a .txt or markdown file.
        
        Args:
            url: URL of the file
            
        Returns:
            List of dictionaries with URL and markdown content
        """
        crawl_config = CrawlerRunConfig()
        
        try:
            result = await self.crawler.arun(url=url, config=crawl_config)
            if result.success and result.markdown:
                return [{'url': url, 'markdown': result.markdown}]
            else:
                print(f"Failed to crawl {url}: {result.error_message}")
                return []
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return []
    
    async def crawl_batch(self, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Batch crawl multiple URLs in parallel.
        
        Args:
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
        
        try:
            results = await self.crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
            return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
        except Exception as e:
            print(f"Error in batch crawl: {e}")
            return []
    
    async def crawl_recursive_internal_links(
        self, 
        start_urls: List[str], 
        max_depth: int = 3, 
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recursively crawl internal links from start URLs up to a maximum depth.
        
        Args:
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
        
        try:
            for depth in range(max_depth):
                urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
                if not urls_to_crawl:
                    break
                
                results = await self.crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
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
        except Exception as e:
            print(f"Error in recursive crawl: {e}")
            return []
    
    async def process_crawl_request(self, request: CrawlRequest) -> CrawlResult:
        """
        Process a crawl request and return results.
        
        Args:
            request: Crawl request with URL and parameters
            
        Returns:
            Crawl result with success status and metadata
        """
        url = str(request.url)
        
        try:
            print(f"🔍 Detecting URL type for: {url}")
            
            # Determine crawl type and execute appropriate strategy
            if self.is_txt(url):
                print("📄 Detected: Text file")
                crawl_results = await self.crawl_markdown_file(url)
                crawl_type = CrawlType.TXT_FILE
            elif self.is_sitemap(url):
                print("🗺️ Detected: Sitemap")
                print("📋 Parsing sitemap...")
                sitemap_urls = await self.parse_sitemap(url)
                if not sitemap_urls:
                    return CrawlResult(
                        success=False,
                        url=url,
                        crawl_type=CrawlType.SITEMAP,
                        error="No URLs found in sitemap"
                    )
                print(f"✅ Found {len(sitemap_urls)} URLs in sitemap")
                print("🚀 Starting batch crawl...")
                crawl_results = await self.crawl_batch(sitemap_urls, max_concurrent=request.max_concurrent)
                crawl_type = CrawlType.SITEMAP
            else:
                print("🌐 Detected: Regular webpage - using single page crawl")
                crawl_results = await self.crawl_markdown_file(url)
                crawl_type = CrawlType.SINGLE_PAGE
            
            print(f"📊 Crawl completed. Found {len(crawl_results)} results")
            
            if not crawl_results:
                return CrawlResult(
                    success=False,
                    url=url,
                    crawl_type=crawl_type,
                    error="No content found"
                )
            
            # Calculate statistics
            total_chunks = 0
            for result in crawl_results:
                chunks = self.smart_chunk_markdown(result['markdown'], chunk_size=request.chunk_size)
                total_chunks += len(chunks)
            
            return CrawlResult(
                success=True,
                url=url,
                crawl_type=crawl_type,
                pages_crawled=len(crawl_results),
                chunks_stored=total_chunks,
                code_examples_stored=0  # Will be calculated separately if needed
            )
            
        except Exception as e:
            print(f"❌ Error processing crawl request: {e}")
            return CrawlResult(
                success=False,
                url=url,
                crawl_type=CrawlType.SINGLE_PAGE,
                error=str(e)
            )
