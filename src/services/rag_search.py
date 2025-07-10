"""
RAG search service for vector and hybrid search operations.

This service encapsulates all search functionality including vector search,
hybrid search, reranking, and result formatting with proper error handling.
"""
from typing import List, Dict, Any, Optional

from src.config import Settings
from src.services.database import DatabaseService
from src.services.embedding import EmbeddingService


class SearchService:
    """
    Service for managing all RAG search operations.
    
    Handles vector search, hybrid search (vector + keyword), reranking,
    and result formatting with proper error handling and deduplication.
    """
    
    def __init__(
        self,
        settings: Settings,
        database_service: DatabaseService,
        embedding_service: EmbeddingService,
        reranking_model: Optional[Any] = None
    ):
        """
        Initialize the search service with required dependencies.
        
        Args:
            settings: Application settings
            database_service: Database service for search operations
            embedding_service: Embedding service for query processing
            reranking_model: Optional cross-encoder model for reranking
        """
        self.settings = settings
        self.database_service = database_service
        self.embedding_service = embedding_service
        self.reranking_model = reranking_model
    
    def search_documents(
        self,
        query: str,
        match_count: int = 10,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for documents using vector similarity with optional hybrid search and reranking.
        
        Args:
            query: Search query
            match_count: Maximum number of results to return
            source_filter: Optional source ID to filter results
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Determine search mode
            search_mode = "hybrid" if self.settings.use_hybrid_search else "vector"
            
            if self.settings.use_hybrid_search:
                # Hybrid search: combine vector and keyword search
                vector_results = self.database_service.search_documents(
                    query, match_count=match_count * 2, source_filter=source_filter
                )
                keyword_results = self._perform_keyword_search(query, match_count)
                
                # Combine and deduplicate results
                results = self._combine_and_deduplicate_results(
                    vector_results, keyword_results, max_results=match_count
                )
            else:
                # Vector search only
                results = self.database_service.search_documents(
                    query, match_count=match_count, source_filter=source_filter
                )
            
            # Apply reranking if enabled
            reranking_applied = False
            if (self.settings.use_reranking and 
                self.reranking_model and 
                results):
                results = self._rerank_results(query, results, "content")
                reranking_applied = True
            
            # Format results
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
            
            return {
                "success": True,
                "query": query,
                "source_filter": source_filter,
                "search_mode": search_mode,
                "reranking_applied": reranking_applied,
                "results": formatted_results,
                "count": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    def search_code_examples(
        self,
        query: str,
        match_count: int = 10,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for code examples using vector similarity with optional hybrid search and reranking.
        
        Args:
            query: Search query
            match_count: Maximum number of results to return
            source_filter: Optional source ID to filter results
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Determine search mode
            search_mode = "hybrid" if self.settings.use_hybrid_search else "vector"
            
            if self.settings.use_hybrid_search:
                # Hybrid search: combine vector and keyword search
                vector_results = self.database_service.search_code_examples(
                    query, match_count=match_count * 2, source_filter=source_filter
                )
                keyword_results = self._perform_keyword_search_code(query, match_count)
                
                # Combine and deduplicate results
                results = self._combine_and_deduplicate_results(
                    vector_results, keyword_results, max_results=match_count
                )
            else:
                # Vector search only
                results = self.database_service.search_code_examples(
                    query, match_count=match_count, source_filter=source_filter
                )
            
            # Apply reranking if enabled
            reranking_applied = False
            if (self.settings.use_reranking and 
                self.reranking_model and 
                results):
                results = self._rerank_results(query, results, "content")
                reranking_applied = True
            
            # Format results
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
            
            return {
                "success": True,
                "query": query,
                "source_filter": source_filter,
                "search_mode": search_mode,
                "reranking_applied": reranking_applied,
                "results": formatted_results,
                "count": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder model.
        
        Args:
            query: The search query
            results: List of search results
            content_key: The key in each result dict that contains the text content
            
        Returns:
            Reranked list of results
        """
        if not self.reranking_model or not results:
            return results
        
        try:
            # Extract content from results
            texts = [result.get(content_key, "") for result in results]
            
            # Create pairs of [query, document] for the cross-encoder
            pairs = [[query, text] for text in texts]
            
            # Get reranking scores
            scores = self.reranking_model.predict(pairs)
            
            # Add scores to results and sort by score (descending)
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked
        except Exception as e:
            print(f"Error during reranking: {e}")
            return results
    
    def _perform_keyword_search(
        self,
        query: str,
        match_count: int
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search on documents using ILIKE.
        
        Args:
            query: Search query
            match_count: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            result = (self.database_service.client
                     .from_('crawled_pages')
                     .select('id, url, content, metadata, source_id')
                     .ilike('content', f'%{query}%')
                     .limit(match_count)
                     .execute())
            
            return result.data
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def _perform_keyword_search_code(
        self,
        query: str,
        match_count: int
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search on code examples using ILIKE on both content and summary.
        
        Args:
            query: Search query
            match_count: Maximum number of results
            
        Returns:
            List of matching code examples
        """
        try:
            result = (self.database_service.client
                     .from_('code_examples')
                     .select('id, url, chunk_number, content, summary, metadata, source_id')
                     .or_(f'content.ilike.%{query}%,summary.ilike.%{query}%')
                     .limit(match_count)
                     .execute())
            
            return result.data
        except Exception as e:
            print(f"Error in code keyword search: {e}")
            return []
    
    def _combine_and_deduplicate_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results, removing duplicates.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            max_results: Maximum number of results to return
            
        Returns:
            Combined and deduplicated results
        """
        # Use URL as the deduplication key
        seen_urls = set()
        combined_results = []
        
        # Add vector results first (they have similarity scores)
        for result in vector_results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined_results.append(result)
        
        # Add keyword results that aren't duplicates
        for result in keyword_results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                # Add a default similarity score for keyword results
                result["similarity"] = 0.5
                combined_results.append(result)
        
        # Return up to max_results
        return combined_results[:max_results]
