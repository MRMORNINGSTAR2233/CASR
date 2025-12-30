"""
Document Embedder

Generates vector embeddings for document chunks.
"""

import asyncio
from enum import Enum
from typing import Optional

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from src.models.documents import DocumentChunk


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    COHERE = "cohere"
    GEMINI = "gemini"


class Embedder:
    """
    Document chunk embedder.
    
    Generates vector embeddings using various providers.
    """
    
    def __init__(
        self,
        provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
        model: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize the embedder.
        
        Args:
            provider: Embedding provider to use
            model: Model name (defaults based on provider)
            batch_size: Batch size for embedding requests
        """
        self.provider = provider
        self.batch_size = batch_size
        
        settings = get_settings()
        
        match provider:
            case EmbeddingProvider.OPENAI:
                self.model = model or settings.openai_embedding_model
                self._client = OpenAI(api_key=settings.openai_api_key)
                self.dimension = self._get_openai_dimension(self.model)
            
            case EmbeddingProvider.SENTENCE_TRANSFORMERS:
                self.model = model or settings.sentence_transformer_model
                self._load_sentence_transformers()
                self.dimension = self._st_model.get_sentence_embedding_dimension()
            
            case EmbeddingProvider.COHERE:
                self.model = model or "embed-english-v3.0"
                import cohere
                self._client = cohere.Client(settings.cohere_api_key)
                self.dimension = 1024
            
            case EmbeddingProvider.GEMINI:
                self.model = model or "models/text-embedding-004"
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self._genai = genai
                self.dimension = 768
    
    def _get_openai_dimension(self, model: str) -> int:
        """Get embedding dimension for OpenAI model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(model, 1536)
    
    def _load_sentence_transformers(self) -> None:
        """Lazy load sentence transformers."""
        from sentence_transformers import SentenceTransformer
        self._st_model = SentenceTransformer(self.model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI."""
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
    
    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Sentence Transformers."""
        embeddings = self._st_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _embed_cohere(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Cohere."""
        response = self._client.embed(
            model=self.model,
            texts=texts,
            input_type="search_document",
        )
        return response.embeddings
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Google Gemini."""
        embeddings = []
        for text in texts:
            result = self._genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings: list[list[float]] = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            match self.provider:
                case EmbeddingProvider.OPENAI:
                    embeddings = self._embed_openai(batch)
                case EmbeddingProvider.SENTENCE_TRANSFORMERS:
                    embeddings = self._embed_sentence_transformers(batch)
                case EmbeddingProvider.COHERE:
                    embeddings = self._embed_cohere(batch)
                case EmbeddingProvider.GEMINI:
                    embeddings = self._embed_gemini(batch)
                case _:
                    raise ValueError(f"Unknown provider: {self.provider}")
            
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]
    
    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        
        Some providers use different models/settings for queries vs documents.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        if self.provider == EmbeddingProvider.COHERE:
            # Cohere has a special input type for queries
            response = self._client.embed(
                model=self.model,
                texts=[query],
                input_type="search_query",
            )
            return response.embeddings[0]
        
        if self.provider == EmbeddingProvider.GEMINI:
            # Gemini has a special task type for queries
            result = self._genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query",
            )
            return result['embedding']
        
        return self.embed_text(query)
    
    def embed_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Generate embedding for a document chunk.
        
        Args:
            chunk: The chunk to embed
            
        Returns:
            Chunk with embedding added
        """
        content = chunk.content_for_embedding
        embedding = self.embed_text(content)
        chunk.embedding = embedding
        return chunk
    
    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Generate embeddings for multiple chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []
        
        # Get content for embedding
        texts = [chunk.content_for_embedding for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_texts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)
    
    async def aembed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Async version of embed_chunks."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_chunks, chunks)
    
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    
    @staticmethod
    def normalize_embedding(embedding: list[float]) -> list[float]:
        """Normalize an embedding vector to unit length."""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return embedding
        return (arr / norm).tolist()
