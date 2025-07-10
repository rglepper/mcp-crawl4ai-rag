"""
Database service for Supabase operations.

This service encapsulates all database operations including document storage,
code example management, source tracking, and vector search functionality.
"""
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from supabase import create_client

from src.config import Settings
from src.services.embedding import create_embedding, create_embeddings_batch


class DatabaseService:
    """
    Service for managing all Supabase database operations.
    
    Handles document storage, code examples, source management, and vector search
    with proper error handling and retry logic.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the database service with Supabase client.
        
        Args:
            settings: Application settings containing Supabase credentials
            
        Raises:
            ValueError: If Supabase credentials are missing or invalid
        """
        self.settings = settings
        
        if not settings.supabase_url or not settings.supabase_service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        self.client = create_client(settings.supabase_url, settings.supabase_service_key)
    
    def add_documents_batch(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        url_to_full_document: Dict[str, str],
        batch_size: int = 20
    ) -> None:
        """
        Add documents to the crawled_pages table in batches.
        
        Args:
            urls: List of URLs
            chunk_numbers: List of chunk numbers
            contents: List of document contents
            metadatas: List of document metadata
            url_to_full_document: Dictionary mapping URLs to full document content
            batch_size: Size of each batch for insertion
        """
        if not urls:
            return
        
        # Delete existing records for these URLs
        unique_urls = list(set(urls))
        try:
            if unique_urls:
                self.client.table("crawled_pages").delete().in_("url", unique_urls).execute()
        except Exception as e:
            print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
            for url in unique_urls:
                try:
                    self.client.table("crawled_pages").delete().eq("url", url).execute()
                except Exception as inner_e:
                    print(f"Error deleting record for URL {url}: {inner_e}")
        
        # Process in batches
        total_items = len(urls)
        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            batch_contents = contents[i:batch_end]
            
            # Create embeddings for this batch
            try:
                embeddings = create_embeddings_batch(batch_contents)
            except Exception as e:
                print(f"Error creating embeddings for batch: {e}")
                continue  # Skip this batch if embeddings fail
            
            # Prepare batch data
            batch_data = []
            for j, embedding in enumerate(embeddings):
                idx = i + j
                parsed_url = urlparse(urls[idx])
                source_id = parsed_url.netloc or parsed_url.path
                
                batch_data.append({
                    'url': urls[idx],
                    'chunk_number': chunk_numbers[idx],
                    'content': contents[idx],
                    'metadata': metadatas[idx],
                    'source_id': source_id,
                    'embedding': embedding
                })
            
            # Insert batch with retry logic
            self._insert_with_retry("crawled_pages", batch_data)
    
    def search_documents(
        self,
        query: str,
        match_count: int = 10,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using vector similarity.
        
        Args:
            query: Query text
            match_count: Maximum number of results to return
            source_filter: Optional source ID to filter results
            
        Returns:
            List of matching documents
        """
        try:
            query_embedding = create_embedding(query)
            
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if source_filter:
                params['filter'] = {'source_id': source_filter}
            
            result = self.client.rpc('match_crawled_pages', params).execute()
            return result.data
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def add_code_examples_batch(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        code_examples: List[str],
        summaries: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 20
    ) -> None:
        """
        Add code examples to the code_examples table in batches.
        
        Args:
            urls: List of URLs
            chunk_numbers: List of chunk numbers
            code_examples: List of code example contents
            summaries: List of code example summaries
            metadatas: List of metadata dictionaries
            batch_size: Size of each batch for insertion
        """
        if not urls:
            return
        
        # Delete existing records for these URLs
        unique_urls = list(set(urls))
        for url in unique_urls:
            try:
                self.client.table('code_examples').delete().eq('url', url).execute()
            except Exception as e:
                print(f"Error deleting existing code examples for {url}: {e}")
        
        # Process in batches
        total_items = len(urls)
        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            batch_texts = []
            
            # Create combined texts for embedding (code + summary)
            for j in range(i, batch_end):
                combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
                batch_texts.append(combined_text)
            
            # Create embeddings for this batch
            try:
                embeddings = create_embeddings_batch(batch_texts)
            except Exception as e:
                print(f"Error creating embeddings for code examples batch: {e}")
                continue
            
            # Prepare batch data
            batch_data = []
            for j, embedding in enumerate(embeddings):
                idx = i + j
                parsed_url = urlparse(urls[idx])
                source_id = parsed_url.netloc or parsed_url.path
                
                batch_data.append({
                    'url': urls[idx],
                    'chunk_number': chunk_numbers[idx],
                    'content': code_examples[idx],
                    'summary': summaries[idx],
                    'metadata': metadatas[idx],
                    'source_id': source_id,
                    'embedding': embedding
                })
            
            # Insert batch with retry logic
            self._insert_with_retry("code_examples", batch_data)
    
    def search_code_examples(
        self,
        query: str,
        match_count: int = 10,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for code examples using vector similarity.
        
        Args:
            query: Query text
            match_count: Maximum number of results to return
            source_filter: Optional source ID to filter results
            
        Returns:
            List of matching code examples
        """
        try:
            # Enhance query for better code example matching
            enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
            query_embedding = create_embedding(enhanced_query)
            
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if source_filter:
                params['source_filter'] = source_filter
            
            result = self.client.rpc('match_code_examples', params).execute()
            return result.data
        except Exception as e:
            print(f"Error searching code examples: {e}")
            return []
    
    def update_source_info(self, source_id: str, summary: str, word_count: int) -> None:
        """
        Update or insert source information in the sources table.
        
        Args:
            source_id: The source ID (domain)
            summary: Summary of the source
            word_count: Total word count for the source
        """
        try:
            # Try to update existing source
            result = self.client.table('sources').update({
                'summary': summary,
                'total_word_count': word_count,
                'updated_at': 'now()'
            }).eq('source_id', source_id).execute()
            
            # If no rows were updated, insert new source
            if not result.data:
                self.client.table('sources').insert({
                    'source_id': source_id,
                    'summary': summary,
                    'total_word_count': word_count
                }).execute()
                print(f"Created new source: {source_id}")
            else:
                print(f"Updated source: {source_id}")
        except Exception as e:
            print(f"Error updating source info for {source_id}: {e}")
    
    def cleanup_source(self, source_id: str) -> Dict[str, Any]:
        """
        Remove a source and all its associated data.
        
        Args:
            source_id: The source ID to remove
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            # Delete from all tables
            self.client.table('crawled_pages').delete().eq('source_id', source_id).execute()
            self.client.table('code_examples').delete().eq('source_id', source_id).execute()
            self.client.table('sources').delete().eq('source_id', source_id).execute()
            
            return {
                "success": True,
                "source_id": source_id,
                "message": f"Successfully cleaned up source: {source_id}"
            }
        except Exception as e:
            return {
                "success": False,
                "source_id": source_id,
                "error": str(e)
            }
    
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """
        Get all available sources from the sources table.
        
        Returns:
            List of source information dictionaries
        """
        try:
            result = self.client.table("sources").select("*").execute()
            return result.data
        except Exception as e:
            print(f"Error getting available sources: {e}")
            return []
    
    def _insert_with_retry(self, table_name: str, batch_data: List[Dict[str, Any]]) -> None:
        """
        Insert batch data with retry logic and exponential backoff.
        
        Args:
            table_name: Name of the table to insert into
            batch_data: List of records to insert
        """
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                self.client.table(table_name).insert(batch_data).execute()
                break  # Success
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Try inserting records individually as last resort
                    self._insert_individually(table_name, batch_data)
    
    def _insert_individually(self, table_name: str, batch_data: List[Dict[str, Any]]) -> None:
        """
        Insert records individually as a fallback when batch insert fails.
        
        Args:
            table_name: Name of the table to insert into
            batch_data: List of records to insert
        """
        print("Attempting to insert records individually...")
        successful_inserts = 0
        for record in batch_data:
            try:
                self.client.table(table_name).insert(record).execute()
                successful_inserts += 1
            except Exception as e:
                print(f"Failed to insert individual record: {e}")
        
        if successful_inserts > 0:
            print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")
