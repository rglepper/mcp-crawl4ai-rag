"""
Directory ingestion service for the Crawl4AI MCP server.

This service handles ingesting local filesystem directories containing markdown
and text files into the knowledge base.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import Settings
from src.services.web_crawling import WebCrawlingService

logger = logging.getLogger(__name__)


class DirectoryIngestionService:
    """Service for ingesting local filesystem directories."""
    
    def __init__(self, supabase_client, settings: Settings):
        """
        Initialize the directory ingestion service.
        
        Args:
            supabase_client: Supabase client for storing content
            settings: Application settings
        """
        self.supabase_client = supabase_client
        self.settings = settings
    
    async def ingest_directory(
        self,
        directory_path: str,
        source_name: Optional[str] = None,
        file_extensions: List[str] = None,
        recursive: bool = True,
        chunk_size: int = 5000
    ) -> Dict[str, Any]:
        """
        Ingest a local directory of markdown/text files.
        
        Args:
            directory_path: Path to the directory
            source_name: Optional custom source name (default: directory name)
            file_extensions: List of file extensions to process (default: ['.md', '.txt', '.markdown'])
            recursive: Whether to recursively process subdirectories (default: True)
            chunk_size: Maximum chunk size in characters (default: 5000)
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Validate directory path
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return {
                    "success": False,
                    "error": f"Directory not found or not a directory: {directory_path}"
                }
            
            # Set default file extensions if not provided
            if file_extensions is None:
                file_extensions = ['.md', '.txt', '.markdown']
            
            # Normalize extensions (ensure they start with '.')
            normalized_extensions = []
            for ext in file_extensions:
                if not ext.startswith('.'):
                    ext = f'.{ext}'
                normalized_extensions.append(ext)
            
            # Set default source name if not provided
            if source_name is None:
                source_name = directory.name
            
            # Find all matching files
            files = []
            if recursive:
                for root, _, filenames in os.walk(directory):
                    for filename in filenames:
                        file_path = Path(root) / filename
                        if any(file_path.suffix.lower() == ext.lower() for ext in normalized_extensions):
                            files.append(file_path)
            else:
                for ext in normalized_extensions:
                    files.extend(directory.glob(f'*{ext}'))
            
            if not files:
                return {
                    "success": False,
                    "error": f"No matching files found in directory: {directory_path}"
                }
            
            logger.info(f"Found {len(files)} files to process in {directory_path}")
            
            # Process each file
            processed_files = []
            chunks_stored = 0
            
            for file_path in files:
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Process content (chunking, embedding, etc.)
                    relative_path = file_path.relative_to(directory)
                    file_url = f"file://{file_path}"
                    
                    # Use WebCrawlingService for chunking
                    web_crawling_service = WebCrawlingService(None, self.settings)
                    chunks = web_crawling_service.smart_chunk_markdown(content, chunk_size=chunk_size)
                    
                    # Store chunks in Supabase
                    for i, chunk in enumerate(chunks):
                        # Extract section info
                        section_info = web_crawling_service.extract_section_info(chunk)
                        
                        # Store in Supabase
                        self.supabase_client.table('crawled_pages').insert({
                            'url': file_url,
                            'source_id': source_name,
                            'title': str(relative_path),
                            'content': chunk,
                            'chunk_number': i + 1,
                            'total_chunks': len(chunks),
                            'headers': section_info['headers'],
                            'char_count': section_info['char_count'],
                            'word_count': section_info['word_count'],
                            'is_code_example': False
                        }).execute()
                    
                    chunks_stored += len(chunks)
                    processed_files.append(str(relative_path))
                    logger.info(f"Processed {file_path} - {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            # Update source metadata
            self.supabase_client.table('sources').upsert({
                'source_id': source_name,
                'title': source_name,
                'description': f"Local directory: {directory_path}",
                'url': f"file://{directory_path}",
                'crawl_type': 'local_directory',
                'pages_count': len(processed_files),
                'chunks_count': chunks_stored
            }).execute()
            
            return {
                "success": True,
                "source_name": source_name,
                "directory_path": directory_path,
                "files_processed": len(processed_files),
                "chunks_stored": chunks_stored,
                "processed_files": processed_files[:5] + (["..."] if len(processed_files) > 5 else [])
            }
            
        except Exception as e:
            logger.error(f"Directory ingestion failed: {e}")
            return {
                "success": False,
                "directory_path": directory_path,
                "error": f"Directory ingestion failed: {str(e)}"
            }
