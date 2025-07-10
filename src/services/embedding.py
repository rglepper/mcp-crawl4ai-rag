"""
Embedding service for creating vector embeddings.

This service encapsulates all embedding operations including batch processing,
contextual embeddings, and query enhancement with proper error handling.
"""
import time
import concurrent.futures
from typing import List, Tuple

import openai

from src.config import Settings


class EmbeddingService:
    """
    Service for managing all embedding operations.

    Handles OpenAI embedding creation, contextual embeddings, batch processing,
    and query enhancement with proper error handling and retry logic.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the embedding service with OpenAI configuration.

        Args:
            settings: Application settings containing OpenAI credentials

        Raises:
            ValueError: If OpenAI API key is missing
        """
        self.settings = settings

        if not settings.openai_api_key:
            raise ValueError("OpenAI API key must be set")

        # Set OpenAI API key
        openai.api_key = settings.openai_api_key

        # Configuration
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
        self.max_retries = 3
        self.initial_retry_delay = 1.0

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to create embeddings for

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []

        retry_delay = self.initial_retry_delay

        for retry in range(self.max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )

                # Extract embeddings from response
                embeddings = []
                for item in response.data:
                    embeddings.append(item.embedding)

                return embeddings
            except Exception as e:
                if retry < self.max_retries - 1:
                    print(f"Error creating embeddings (attempt {retry + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to create embeddings after {self.max_retries} attempts: {e}")
                    # Try creating embeddings individually as fallback
                    return self._create_embeddings_individually(texts)

    def _create_embeddings_individually(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings individually as a fallback when batch creation fails.

        Args:
            texts: List of texts to create embeddings for

        Returns:
            List of embeddings with fallback to zero embeddings for failures
        """
        print("Attempting to create embeddings individually...")
        embeddings = []
        successful_count = 0

        for i, text in enumerate(texts):
            try:
                individual_response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=[text]
                )
                embeddings.append(individual_response.data[0].embedding)
                successful_count += 1
            except Exception as individual_error:
                print(f"Failed to create embedding for text {i}: {individual_error}")
                # Add zero embedding as fallback
                embeddings.append([0.0] * self.embedding_dimension)

        print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
        return embeddings

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text using OpenAI's API.

        Args:
            text: Text to create an embedding for

        Returns:
            List of floats representing the embedding
        """
        try:
            embeddings = self.create_embeddings_batch([text])
            return embeddings[0] if embeddings else [0.0] * self.embedding_dimension
        except Exception as e:
            print(f"Error creating embedding: {e}")
            # Return empty embedding if there's an error
            return [0.0] * self.embedding_dimension

    def generate_contextual_embedding(self, full_document: str, chunk: str) -> Tuple[str, bool]:
        """
        Generate contextual information for a chunk within a document to improve retrieval.

        Args:
            full_document: The complete document text
            chunk: The specific chunk of text to generate context for

        Returns:
            Tuple containing:
            - The contextual text that situates the chunk within the document
            - Boolean indicating if contextual embedding was performed
        """
        if not self.settings.use_contextual_embeddings:
            return chunk, False

        try:
            # Create a prompt to generate contextual information
            prompt = f"""
            Given the following full document and a specific chunk from it, provide contextual information
            that would help with semantic search and retrieval. The context should situate the chunk within
            the broader document and highlight its key concepts.

            Full Document (first 2000 chars):
            {full_document[:2000]}

            Specific Chunk:
            {chunk}

            Please provide enhanced contextual text that includes the chunk content but with additional
            context about its role in the document:
            """

            response = openai.chat.completions.create(
                model=self.settings.model_choice,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )

            contextual_text = response.choices[0].message.content.strip()
            return contextual_text, True

        except Exception as e:
            print(f"Error generating contextual embedding: {e}")
            return chunk, False

    def process_chunks_with_context(
        self,
        chunks: List[str],
        urls: List[str],
        url_to_full_document: dict
    ) -> List[str]:
        """
        Process multiple chunks with contextual embeddings in parallel.

        Args:
            chunks: List of text chunks to process
            urls: List of URLs corresponding to each chunk
            url_to_full_document: Dictionary mapping URLs to full document content

        Returns:
            List of processed chunks (with context if enabled)
        """
        if not self.settings.use_contextual_embeddings:
            return chunks

        # Prepare arguments for parallel processing
        process_args = []
        for i, chunk in enumerate(chunks):
            url = urls[i]
            full_document = url_to_full_document.get(url, "")
            process_args.append((full_document, chunk))

        # Process in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks and collect results
            future_to_idx = {
                executor.submit(self.generate_contextual_embedding, arg[0], arg[1]): idx
                for idx, arg in enumerate(process_args)
            }

            # Process results as they complete
            results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result, success = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Error processing chunk {idx}: {e}")
                    # Use original content as fallback
                    results[idx] = chunks[idx]

            return results

    def enhance_query_for_code_search(self, query: str) -> str:
        """
        Enhance a query for better code example search.

        Args:
            query: Original search query

        Returns:
            Enhanced query with code-specific context
        """
        return f"Code example for {query}\n\nSummary: Example code showing {query}"

    def validate_embeddings(self, embeddings: List[List[float]], texts: List[str]) -> List[List[float]]:
        """
        Validate embeddings and replace invalid ones with newly created embeddings.

        Args:
            embeddings: List of embeddings to validate
            texts: Corresponding texts for creating replacement embeddings

        Returns:
            List of valid embeddings
        """
        valid_embeddings = []

        for i, embedding in enumerate(embeddings):
            # Check if embedding is valid (not None, not empty, not all zeros)
            if (embedding and
                len(embedding) > 0 and
                not all(v == 0.0 for v in embedding)):
                valid_embeddings.append(embedding)
            else:
                print(f"Warning: Invalid embedding detected at index {i}, creating new one...")
                # Create a new embedding as replacement
                replacement_embedding = self.create_embedding(texts[i])
                valid_embeddings.append(replacement_embedding)

        return valid_embeddings


# Backward compatibility functions for existing code
def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Backward compatibility function."""
    from src.config import get_settings
    settings = get_settings()
    service = EmbeddingService(settings)
    return service.create_embeddings_batch(texts)


def create_embedding(text: str) -> List[float]:
    """Backward compatibility function."""
    from src.config import get_settings
    settings = get_settings()
    service = EmbeddingService(settings)
    return service.create_embedding(text)
