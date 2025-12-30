"""
Document Chunker

Splits documents into chunks for embedding and retrieval.
"""

import re
from enum import Enum
from typing import Optional
from uuid import UUID

import tiktoken

from src.models.documents import Document, DocumentChunk, DocumentMetadata


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"           # Fixed token/character count
    SEMANTIC = "semantic"               # Paragraph/section-aware
    SENTENCE = "sentence"               # Sentence-based
    RECURSIVE = "recursive"             # Recursive text splitting


class Chunker:
    """
    Document chunker with multiple strategies.
    
    Splits documents into overlapping chunks suitable for embedding
    and retrieval operations.
    """
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
    
    # Paragraph separator
    PARAGRAPH_SEP = re.compile(r'\n\s*\n')
    
    # Section headers (Markdown-style)
    SECTION_HEADER = re.compile(r'^#+\s+.+$', re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        tokenizer_model: str = "cl100k_base"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: Chunking strategy to use
            tokenizer_model: Tiktoken model for tokenization
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        try:
            self._tokenizer = tiktoken.get_encoding(tokenizer_model)
        except Exception:
            # Fallback to character-based sizing
            self._tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: approximate 4 chars per token
        return len(text) // 4
    
    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """
        Chunk a document using the configured strategy.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
        """
        match self.strategy:
            case ChunkingStrategy.FIXED_SIZE:
                return self._chunk_fixed_size(document)
            case ChunkingStrategy.SEMANTIC:
                return self._chunk_semantic(document)
            case ChunkingStrategy.SENTENCE:
                return self._chunk_sentence(document)
            case ChunkingStrategy.RECURSIVE:
                return self._chunk_recursive(document)
            case _:
                return self._chunk_recursive(document)
    
    def _create_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int
    ) -> DocumentChunk:
        """Create a DocumentChunk from content."""
        return DocumentChunk(
            document_id=document.id,
            content=content.strip(),
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=document.metadata.model_copy(deep=True),
        )
    
    def _chunk_fixed_size(self, document: Document) -> list[DocumentChunk]:
        """
        Chunk using fixed token size with overlap.
        
        Simple strategy that splits text into fixed-size chunks.
        """
        text = document.content
        chunks: list[DocumentChunk] = []
        
        if self._tokenizer:
            tokens = self._tokenizer.encode(text)
            
            start = 0
            chunk_index = 0
            
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self._tokenizer.decode(chunk_tokens)
                
                # Calculate character positions (approximate)
                char_start = len(self._tokenizer.decode(tokens[:start]))
                char_end = len(self._tokenizer.decode(tokens[:end]))
                
                chunks.append(self._create_chunk(
                    document=document,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=char_start,
                    end_char=char_end,
                ))
                
                start = end - self.chunk_overlap
                if start >= end:
                    break
                chunk_index += 1
        else:
            # Character-based fallback
            char_chunk_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4
            
            start = 0
            chunk_index = 0
            
            while start < len(text):
                end = min(start + char_chunk_size, len(text))
                chunk_text = text[start:end]
                
                chunks.append(self._create_chunk(
                    document=document,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                ))
                
                start = end - char_overlap
                if start >= end:
                    break
                chunk_index += 1
        
        return chunks
    
    def _chunk_semantic(self, document: Document) -> list[DocumentChunk]:
        """
        Chunk using semantic boundaries (paragraphs, sections).
        
        Tries to keep semantic units together while respecting size limits.
        """
        text = document.content
        chunks: list[DocumentChunk] = []
        
        # Split by sections first, then paragraphs
        sections = self.SECTION_HEADER.split(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for section in sections:
            paragraphs = self.PARAGRAPH_SEP.split(section)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_tokens = self.count_tokens(para)
                current_tokens = self.count_tokens(current_chunk)
                
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    end_char = current_start + len(current_chunk)
                    chunks.append(self._create_chunk(
                        document=document,
                        content=current_chunk,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=end_char,
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + para
                    current_start = end_char - len(overlap_text)
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                document=document,
                content=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
            ))
        
        return chunks
    
    def _chunk_sentence(self, document: Document) -> list[DocumentChunk]:
        """
        Chunk at sentence boundaries.
        
        Groups sentences together up to the chunk size limit.
        """
        text = document.content
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        chunks: list[DocumentChunk] = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self.count_tokens(sentence)
            current_tokens = self.count_tokens(current_chunk)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                end_char = current_start + len(current_chunk)
                chunks.append(self._create_chunk(
                    document=document,
                    content=current_chunk,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=end_char,
                ))
                chunk_index += 1
                
                # Start new chunk
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_start = end_char - len(overlap_text)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                document=document,
                content=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
            ))
        
        return chunks
    
    def _chunk_recursive(self, document: Document) -> list[DocumentChunk]:
        """
        Recursive text splitting.
        
        Tries multiple separators in order of preference,
        falling back to smaller separators if chunks are too large.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_recursive(
            document=document,
            text=document.content,
            separators=separators,
            start_char=0,
            chunk_index_start=0,
        )
    
    def _split_recursive(
        self,
        document: Document,
        text: str,
        separators: list[str],
        start_char: int,
        chunk_index_start: int
    ) -> list[DocumentChunk]:
        """Recursively split text using separators."""
        chunks: list[DocumentChunk] = []
        
        # Find the first separator that works
        separator = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]
        
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split as last resort
            splits = list(text)
        
        current_chunk = ""
        current_start = start_char
        chunk_index = chunk_index_start
        
        for split in splits:
            if not split:
                continue
            
            test_chunk = current_chunk + separator + split if current_chunk else split
            test_tokens = self.count_tokens(test_chunk)
            
            if test_tokens > self.chunk_size:
                # Current chunk is ready
                if current_chunk:
                    # Check if we need to recursively split the current chunk
                    if self.count_tokens(current_chunk) > self.chunk_size and remaining_seps:
                        sub_chunks = self._split_recursive(
                            document=document,
                            text=current_chunk,
                            separators=remaining_seps,
                            start_char=current_start,
                            chunk_index_start=chunk_index,
                        )
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                    else:
                        end_char = current_start + len(current_chunk)
                        chunks.append(self._create_chunk(
                            document=document,
                            content=current_chunk,
                            chunk_index=chunk_index,
                            start_char=current_start,
                            end_char=end_char,
                        ))
                        chunk_index += 1
                        current_start = end_char
                
                # Handle the new split
                if self.count_tokens(split) > self.chunk_size and remaining_seps:
                    # Need to split this piece further
                    sub_chunks = self._split_recursive(
                        document=document,
                        text=split,
                        separators=remaining_seps,
                        start_char=current_start,
                        chunk_index_start=chunk_index,
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    if sub_chunks:
                        current_start = sub_chunks[-1].end_char
                    current_chunk = ""
                else:
                    current_chunk = split
            else:
                current_chunk = test_chunk
        
        # Handle remaining chunk
        if current_chunk.strip():
            if self.count_tokens(current_chunk) > self.chunk_size and remaining_seps:
                sub_chunks = self._split_recursive(
                    document=document,
                    text=current_chunk,
                    separators=remaining_seps,
                    start_char=current_start,
                    chunk_index_start=chunk_index,
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._create_chunk(
                    document=document,
                    content=current_chunk,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                ))
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from the end of a chunk."""
        if self._tokenizer:
            tokens = self._tokenizer.encode(text)
            overlap_tokens = tokens[-self.chunk_overlap:]
            return self._tokenizer.decode(overlap_tokens)
        else:
            char_overlap = self.chunk_overlap * 4
            return text[-char_overlap:] if len(text) > char_overlap else text
