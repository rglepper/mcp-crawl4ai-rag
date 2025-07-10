"""
Tests for WebCrawlingService.

This module tests all web crawling functionality including URL detection,
sitemap parsing, content chunking, and various crawling strategies.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
from pathlib import Path

from src.services.web_crawling import WebCrawlingService
from src.models import CrawlRequest, CrawlResult, CrawlType
from src.config import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.use_contextual_embeddings = False
    settings.use_reranking = False
    return settings


@pytest.fixture
def mock_crawler():
    """Create mock AsyncWebCrawler for testing."""
    crawler = AsyncMock()
    return crawler


@pytest.fixture
def web_crawling_service(mock_crawler, mock_settings):
    """Create WebCrawlingService with mocked dependencies."""
    return WebCrawlingService(mock_crawler, mock_settings)


class TestUrlDetection:
    """Test URL type detection functions."""
    
    def test_is_sitemap_with_sitemap_xml(self, web_crawling_service):
        """Test sitemap detection with sitemap.xml URL."""
        assert web_crawling_service.is_sitemap("https://example.com/sitemap.xml") is True
    
    def test_is_sitemap_with_sitemap_in_path(self, web_crawling_service):
        """Test sitemap detection with sitemap in path."""
        assert web_crawling_service.is_sitemap("https://example.com/docs/sitemap/index.xml") is True
    
    def test_is_sitemap_with_regular_url(self, web_crawling_service):
        """Test sitemap detection with regular URL should return False."""
        assert web_crawling_service.is_sitemap("https://example.com/docs/page.html") is False
    
    def test_is_txt_with_txt_file(self, web_crawling_service):
        """Test text file detection with .txt URL."""
        assert web_crawling_service.is_txt("https://example.com/llms.txt") is True
    
    def test_is_txt_with_regular_url(self, web_crawling_service):
        """Test text file detection with regular URL should return False."""
        assert web_crawling_service.is_txt("https://example.com/docs/page.html") is False


class TestSitemapParsing:
    """Test sitemap parsing functionality."""
    
    @patch('src.services.web_crawling.requests.get')
    async def test_parse_sitemap_success(self, mock_get, web_crawling_service):
        """Test successful sitemap parsing."""
        # Mock successful response with XML sitemap
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'''<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>'''
        mock_get.return_value = mock_response
        
        urls = await web_crawling_service.parse_sitemap("https://example.com/sitemap.xml")
        
        assert len(urls) == 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
    
    @patch('src.services.web_crawling.requests.get')
    async def test_parse_sitemap_http_error(self, mock_get, web_crawling_service):
        """Test sitemap parsing with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        urls = await web_crawling_service.parse_sitemap("https://example.com/sitemap.xml")
        
        assert urls == []
    
    @patch('src.services.web_crawling.requests.get')
    async def test_parse_sitemap_invalid_xml(self, mock_get, web_crawling_service):
        """Test sitemap parsing with invalid XML."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<invalid xml content'
        mock_get.return_value = mock_response
        
        urls = await web_crawling_service.parse_sitemap("https://example.com/sitemap.xml")
        
        assert urls == []


class TestTextProcessing:
    """Test text chunking and processing functions."""
    
    def test_smart_chunk_markdown_small_text(self, web_crawling_service):
        """Test chunking with text smaller than chunk size."""
        text = "This is a small text that should not be chunked."
        chunks = web_crawling_service.smart_chunk_markdown(text, chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_smart_chunk_markdown_large_text(self, web_crawling_service):
        """Test chunking with text larger than chunk size."""
        # Create text larger than chunk size
        text = "This is a paragraph.\n\n" * 100  # About 2300 chars
        chunks = web_crawling_service.smart_chunk_markdown(text, chunk_size=1000)
        
        assert len(chunks) > 1
        # Each chunk should be roughly around chunk_size
        for chunk in chunks[:-1]:  # All but last chunk
            assert len(chunk) <= 1500  # Allow some flexibility
    
    def test_smart_chunk_markdown_with_code_blocks(self, web_crawling_service):
        """Test chunking respects code block boundaries."""
        text = "Some text before.\n\n```python\ndef function():\n    pass\n```\n\nSome text after." * 50
        chunks = web_crawling_service.smart_chunk_markdown(text, chunk_size=500)
        
        # Should not break code blocks
        for chunk in chunks:
            # If chunk contains ```, it should have matching pairs or be at boundary
            backtick_count = chunk.count('```')
            assert backtick_count % 2 == 0 or chunk.startswith('```') or chunk.endswith('```')
    
    def test_extract_section_info_with_headers(self, web_crawling_service):
        """Test section info extraction with headers."""
        chunk = "# Main Header\n\nSome content here.\n\n## Sub Header\n\nMore content."
        info = web_crawling_service.extract_section_info(chunk)
        
        assert "headers" in info
        assert "# Main Header" in info["headers"]
        assert "## Sub Header" in info["headers"]
        assert info["char_count"] == len(chunk)
        assert info["word_count"] > 0
    
    def test_extract_section_info_no_headers(self, web_crawling_service):
        """Test section info extraction without headers."""
        chunk = "Just some plain text without any headers."
        info = web_crawling_service.extract_section_info(chunk)
        
        assert info["headers"] == ""
        assert info["char_count"] == len(chunk)
        assert info["word_count"] == len(chunk.split())


class TestCrawlingMethods:
    """Test different crawling methods."""
    
    async def test_crawl_markdown_file_success(self, web_crawling_service, mock_crawler):
        """Test successful markdown file crawling."""
        # Mock successful crawl result
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Test Content\n\nThis is test markdown content."
        mock_crawler.arun.return_value = mock_result
        
        results = await web_crawling_service.crawl_markdown_file("https://example.com/test.txt")
        
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/test.txt"
        assert results[0]["markdown"] == mock_result.markdown
    
    async def test_crawl_markdown_file_failure(self, web_crawling_service, mock_crawler):
        """Test failed markdown file crawling."""
        # Mock failed crawl result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Failed to fetch"
        mock_crawler.arun.return_value = mock_result
        
        results = await web_crawling_service.crawl_markdown_file("https://example.com/test.txt")
        
        assert results == []
    
    async def test_crawl_batch_success(self, web_crawling_service, mock_crawler):
        """Test successful batch crawling."""
        # Mock successful batch results
        mock_results = []
        for i, url in enumerate(["https://example.com/page1", "https://example.com/page2"]):
            mock_result = Mock()
            mock_result.success = True
            mock_result.url = url
            mock_result.markdown = f"Content for page {i+1}"
            mock_results.append(mock_result)
        
        mock_crawler.arun_many.return_value = mock_results
        
        urls = ["https://example.com/page1", "https://example.com/page2"]
        results = await web_crawling_service.crawl_batch(urls)
        
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/page1"
        assert results[1]["url"] == "https://example.com/page2"
    
    async def test_crawl_batch_partial_failure(self, web_crawling_service, mock_crawler):
        """Test batch crawling with some failures."""
        # Mock mixed results (one success, one failure)
        mock_result1 = Mock()
        mock_result1.success = True
        mock_result1.url = "https://example.com/page1"
        mock_result1.markdown = "Content for page 1"
        
        mock_result2 = Mock()
        mock_result2.success = False
        mock_result2.url = "https://example.com/page2"
        mock_result2.markdown = None
        
        mock_crawler.arun_many.return_value = [mock_result1, mock_result2]
        
        urls = ["https://example.com/page1", "https://example.com/page2"]
        results = await web_crawling_service.crawl_batch(urls)
        
        # Should only return successful results
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/page1"


class TestCrawlRequestProcessing:
    """Test main crawl request processing logic."""
    
    async def test_process_crawl_request_single_page(self, web_crawling_service, mock_crawler):
        """Test processing single page crawl request."""
        # Mock successful single page crawl
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Test Page\n\nThis is test content."
        mock_crawler.arun.return_value = mock_result
        
        request = CrawlRequest(url="https://example.com/page.html")
        
        with patch.object(web_crawling_service, 'is_sitemap', return_value=False), \
             patch.object(web_crawling_service, 'is_txt', return_value=False):
            
            result = await web_crawling_service.process_crawl_request(request)
        
        assert isinstance(result, CrawlResult)
        assert result.success is True
        assert result.crawl_type == CrawlType.SINGLE_PAGE
        assert result.pages_crawled == 1
    
    async def test_process_crawl_request_sitemap(self, web_crawling_service, mock_crawler):
        """Test processing sitemap crawl request."""
        request = CrawlRequest(url="https://example.com/sitemap.xml")
        
        # Mock sitemap parsing and batch crawling
        with patch.object(web_crawling_service, 'is_sitemap', return_value=True), \
             patch.object(web_crawling_service, 'parse_sitemap', return_value=["https://example.com/page1"]), \
             patch.object(web_crawling_service, 'crawl_batch', return_value=[{"url": "https://example.com/page1", "markdown": "content"}]):
            
            result = await web_crawling_service.process_crawl_request(request)
        
        assert isinstance(result, CrawlResult)
        assert result.success is True
        assert result.crawl_type == CrawlType.SITEMAP
    
    async def test_process_crawl_request_txt_file(self, web_crawling_service, mock_crawler):
        """Test processing text file crawl request."""
        request = CrawlRequest(url="https://example.com/llms.txt")
        
        # Mock text file crawling
        with patch.object(web_crawling_service, 'is_txt', return_value=True), \
             patch.object(web_crawling_service, 'crawl_markdown_file', return_value=[{"url": "https://example.com/llms.txt", "markdown": "content"}]):
            
            result = await web_crawling_service.process_crawl_request(request)
        
        assert isinstance(result, CrawlResult)
        assert result.success is True
        assert result.crawl_type == CrawlType.TXT_FILE
    
    async def test_process_crawl_request_failure(self, web_crawling_service, mock_crawler):
        """Test processing crawl request with failure."""
        request = CrawlRequest(url="https://example.com/page.html")
        
        # Mock failed crawl
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Network error"
        mock_crawler.arun.return_value = mock_result
        
        with patch.object(web_crawling_service, 'is_sitemap', return_value=False), \
             patch.object(web_crawling_service, 'is_txt', return_value=False):
            
            result = await web_crawling_service.process_crawl_request(request)
        
        assert isinstance(result, CrawlResult)
        assert result.success is False
        assert result.error is not None
