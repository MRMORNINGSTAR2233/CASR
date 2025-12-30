"""
Document Contextualizer

Adds contextual information to document chunks for improved retrieval.
Implements Anthropic's Contextual Retrieval approach.
"""

import asyncio
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from src.models.documents import Document, DocumentChunk


class Contextualizer:
    """
    Document chunk contextualizer.
    
    Uses an LLM to generate context summaries for document chunks,
    prepending relevant information about the chunk's position and
    meaning within the full document.
    """
    
    # Default prompt for context generation
    DEFAULT_PROMPT = """<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    # Prompt for domain-specific contextualization
    DOMAIN_PROMPT = """<document>
{document_content}
</document>

Document Domain: {domain}
Document Title: {title}

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context (2-3 sentences) to situate this chunk within the overall document. Include:
1. What the chunk is about
2. Its relationship to the document's main topic
3. Any key entities or concepts mentioned

Answer only with the succinct context and nothing else."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        max_document_tokens: int = 8000,
        max_context_tokens: int = 150,
    ):
        """
        Initialize the contextualizer.
        
        Args:
            provider: LLM provider ("anthropic", "openai", "gemini", or "groq")
            model: Model name (defaults based on provider)
            max_document_tokens: Max tokens for document context
            max_context_tokens: Max tokens for generated context
        """
        self.provider = provider
        self.max_document_tokens = max_document_tokens
        self.max_context_tokens = max_context_tokens
        
        settings = get_settings()
        
        if provider == "anthropic":
            self.model = model or "claude-3-haiku-20240307"
            self._client = Anthropic(api_key=settings.anthropic_api_key)
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            self._client = OpenAI(api_key=settings.openai_api_key)
        elif provider == "gemini":
            self.model = model or "gemini-2.0-flash"
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._genai = genai
            self._gemini_model = genai.GenerativeModel(self.model)
        elif provider == "groq":
            self.model = model or "llama-3.3-70b-specdec"
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _generate_context_anthropic(
        self,
        document_content: str,
        chunk_content: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Generate context using Anthropic Claude."""
        if metadata:
            prompt = self.DOMAIN_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
                domain=metadata.get("domain", "general"),
                title=metadata.get("title", "Untitled"),
            )
        else:
            prompt = self.DEFAULT_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
            )
        
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_context_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        return response.content[0].text.strip()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _generate_context_openai(
        self,
        document_content: str,
        chunk_content: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Generate context using OpenAI."""
        if metadata:
            prompt = self.DOMAIN_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
                domain=metadata.get("domain", "general"),
                title=metadata.get("title", "Untitled"),
            )
        else:
            prompt = self.DEFAULT_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
            )
        
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_context_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        return response.choices[0].message.content.strip()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _generate_context_gemini(
        self,
        document_content: str,
        chunk_content: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Generate context using Google Gemini."""
        if metadata:
            prompt = self.DOMAIN_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
                domain=metadata.get("domain", "general"),
                title=metadata.get("title", "Untitled"),
            )
        else:
            prompt = self.DEFAULT_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
            )
        
        response = self._gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": self.max_context_tokens,
                "temperature": 0.3,
            }
        )
        
        return response.text.strip()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _generate_context_groq(
        self,
        document_content: str,
        chunk_content: str,
        metadata: Optional[dict] = None
    ) -> str:
        """Generate context using Groq."""
        if metadata:
            prompt = self.DOMAIN_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
                domain=metadata.get("domain", "general"),
                title=metadata.get("title", "Untitled"),
            )
        else:
            prompt = self.DEFAULT_PROMPT.format(
                document_content=document_content,
                chunk_content=chunk_content,
            )
        
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_context_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        return response.choices[0].message.content.strip()
    
    def _truncate_document(self, content: str) -> str:
        """Truncate document to fit within token limits."""
        # Approximate token count (4 chars per token)
        max_chars = self.max_document_tokens * 4
        if len(content) > max_chars:
            return content[:max_chars] + "...[truncated]"
        return content
    
    def generate_context(
        self,
        document: Document,
        chunk: DocumentChunk
    ) -> str:
        """
        Generate context for a single chunk.
        
        Args:
            document: The parent document
            chunk: The chunk to contextualize
            
        Returns:
            Generated context string
        """
        document_content = self._truncate_document(document.content)
        
        metadata = {
            "domain": document.metadata.domain,
            "title": document.metadata.title,
        }
        
        if self.provider == "anthropic":
            return self._generate_context_anthropic(
                document_content=document_content,
                chunk_content=chunk.content,
                metadata=metadata,
            )
        elif self.provider == "openai":
            return self._generate_context_openai(
                document_content=document_content,
                chunk_content=chunk.content,
                metadata=metadata,
            )
        elif self.provider == "gemini":
            return self._generate_context_gemini(
                document_content=document_content,
                chunk_content=chunk.content,
                metadata=metadata,
            )
        elif self.provider == "groq":
            return self._generate_context_groq(
                document_content=document_content,
                chunk_content=chunk.content,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def contextualize_chunk(
        self,
        document: Document,
        chunk: DocumentChunk
    ) -> DocumentChunk:
        """
        Add context to a chunk.
        
        Args:
            document: The parent document
            chunk: The chunk to contextualize
            
        Returns:
            Updated chunk with context
        """
        context = self.generate_context(document, chunk)
        chunk.context_summary = context
        chunk.contextualized_content = f"{context}\n\n{chunk.content}"
        return chunk
    
    def contextualize_chunks(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        batch_size: int = 5
    ) -> list[DocumentChunk]:
        """
        Add context to multiple chunks.
        
        Args:
            document: The parent document
            chunks: List of chunks to contextualize
            batch_size: Number of parallel requests
            
        Returns:
            List of contextualized chunks
        """
        contextualized = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                try:
                    contextualized_chunk = self.contextualize_chunk(document, chunk)
                    contextualized.append(contextualized_chunk)
                except Exception as e:
                    # Log error but continue with original chunk
                    print(f"Error contextualizing chunk {chunk.id}: {e}")
                    chunk.context_summary = None
                    chunk.contextualized_content = chunk.content
                    contextualized.append(chunk)
        
        return contextualized
    
    async def acontextualize_chunk(
        self,
        document: Document,
        chunk: DocumentChunk
    ) -> DocumentChunk:
        """Async version of contextualize_chunk."""
        # Run in thread pool since API clients may not be async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.contextualize_chunk,
            document,
            chunk,
        )
    
    async def acontextualize_chunks(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        max_concurrent: int = 5
    ) -> list[DocumentChunk]:
        """
        Async contextualization of multiple chunks.
        
        Args:
            document: The parent document
            chunks: List of chunks to contextualize
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of contextualized chunks
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def contextualize_with_semaphore(chunk: DocumentChunk) -> DocumentChunk:
            async with semaphore:
                try:
                    return await self.acontextualize_chunk(document, chunk)
                except Exception as e:
                    print(f"Error contextualizing chunk {chunk.id}: {e}")
                    chunk.context_summary = None
                    chunk.contextualized_content = chunk.content
                    return chunk
        
        tasks = [contextualize_with_semaphore(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)


class SimpleContextualizer:
    """
    Simple rule-based contextualizer for cases where LLM is not available.
    
    Generates context based on document metadata and chunk position.
    """
    
    def __init__(self):
        """Initialize the simple contextualizer."""
        pass
    
    def generate_context(
        self,
        document: Document,
        chunk: DocumentChunk
    ) -> str:
        """
        Generate simple context based on metadata and position.
        
        Args:
            document: The parent document
            chunk: The chunk to contextualize
            
        Returns:
            Generated context string
        """
        parts = []
        
        # Document identification
        parts.append(f"From document: {document.metadata.title}")
        
        # Source and domain
        if document.metadata.source:
            parts.append(f"Source: {document.metadata.source}")
        if document.metadata.domain:
            parts.append(f"Domain: {document.metadata.domain}")
        
        # Position information
        total_length = len(document.content)
        position_pct = (chunk.start_char / total_length) * 100
        
        if position_pct < 10:
            parts.append("This is from the beginning of the document.")
        elif position_pct > 90:
            parts.append("This is from the end of the document.")
        else:
            parts.append(f"This is from approximately {position_pct:.0f}% into the document.")
        
        # Tags if available
        if document.metadata.tags:
            parts.append(f"Topics: {', '.join(document.metadata.tags[:5])}")
        
        return " ".join(parts)
    
    def contextualize_chunk(
        self,
        document: Document,
        chunk: DocumentChunk
    ) -> DocumentChunk:
        """Add simple context to a chunk."""
        context = self.generate_context(document, chunk)
        chunk.context_summary = context
        chunk.contextualized_content = f"{context}\n\n{chunk.content}"
        return chunk
    
    def contextualize_chunks(
        self,
        document: Document,
        chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Add simple context to multiple chunks."""
        return [self.contextualize_chunk(document, chunk) for chunk in chunks]
