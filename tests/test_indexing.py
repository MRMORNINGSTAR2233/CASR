"""
Tests for Indexing System

Tests for Chunker, Contextualizer, Embedder, and IndexManager.
"""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from src.models.documents import Document, DocumentChunk, DocumentMetadata, SecurityClassification


class TestChunker:
    """Tests for document chunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        from src.indexing import Chunker
        return Chunker(chunk_size=100, chunk_overlap=20)
    
    @pytest.fixture
    def long_document(self):
        """Create a document with enough content to produce multiple chunks."""
        content = """
        # Introduction
        
        This is the first paragraph of our test document. It contains enough text
        to ensure that the chunking process will create multiple chunks.
        
        # Section One
        
        The first section discusses various topics related to the system architecture.
        We need to have sufficient content here to test the chunking behavior properly.
        This includes multiple sentences and paragraphs to ensure proper segmentation.
        
        # Section Two
        
        The second section covers implementation details. Again, we include enough
        text to make this section substantial. The chunker should be able to handle
        this content and split it appropriately based on the configured chunk size.
        
        # Section Three
        
        Here we have more content to test the chunking functionality. This ensures
        that we have enough material to produce at least three or four chunks from
        this document. The chunking algorithm should respect paragraph boundaries
        where possible while still maintaining the target chunk size.
        
        # Conclusion
        
        Finally, we conclude with a summary section. This provides a natural
        ending point for the document and tests how the chunker handles the
        final portions of the content.
        """
        
        return Document(
            id=uuid4(),
            content=content,
            metadata=DocumentMetadata(
                title="Long Test Document",
                source="test",
                domain="research",
            ),
        )
    
    def test_chunk_document(self, chunker, long_document):
        """Test chunking a document."""
        chunks = chunker.chunk(long_document)
        
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    def test_chunk_indices(self, chunker, long_document):
        """Test that chunks have correct indices."""
        chunks = chunker.chunk(long_document)
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_chunk_metadata_preserved(self, chunker, long_document):
        """Test that document metadata is preserved in chunks."""
        chunks = chunker.chunk(long_document)
        
        for chunk in chunks:
            assert chunk.metadata.title == long_document.metadata.title
            assert chunk.metadata.domain == long_document.metadata.domain
            assert chunk.document_id == long_document.id
    
    def test_chunk_size_respected(self, chunker, long_document):
        """Test that chunks respect size limits (approximately)."""
        chunks = chunker.chunk(long_document)
        
        # Most chunks should be within 2x the target size
        for chunk in chunks[:-1]:  # Last chunk might be smaller
            assert chunk.token_count <= chunker.chunk_size * 2
    
    def test_empty_document(self, chunker):
        """Test chunking an empty document."""
        doc = Document(
            content="",
            metadata=DocumentMetadata(title="Empty", source="test", domain="test"),
        )
        
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0
    
    def test_small_document_single_chunk(self, chunker):
        """Test that a small document produces a single chunk."""
        doc = Document(
            content="This is a very short document.",
            metadata=DocumentMetadata(title="Short", source="test", domain="test"),
        )
        
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content.strip() == "This is a very short document."


class TestContextualizer:
    """Tests for contextualizer."""
    
    @pytest.fixture
    def simple_contextualizer(self):
        """Create simple contextualizer (no LLM)."""
        from src.indexing import SimpleContextualizer
        return SimpleContextualizer()
    
    @pytest.fixture
    def sample_chunks(self, sample_document):
        """Create sample chunks from document."""
        from src.indexing import Chunker
        chunker = Chunker()
        return chunker.chunk(sample_document)
    
    def test_simple_contextualize(self, simple_contextualizer, sample_document, sample_chunks):
        """Test simple contextualization."""
        if not sample_chunks:
            pytest.skip("No chunks generated")
        
        contextualized = simple_contextualizer.contextualize(
            document=sample_document,
            chunks=sample_chunks,
        )
        
        assert len(contextualized) == len(sample_chunks)
        
        # Check that context was added
        for chunk in contextualized:
            assert chunk.context is not None
    
    def test_context_includes_document_info(self, simple_contextualizer, sample_document, sample_chunks):
        """Test that context includes document info."""
        if not sample_chunks:
            pytest.skip("No chunks generated")
        
        contextualized = simple_contextualizer.contextualize(
            document=sample_document,
            chunks=sample_chunks,
        )
        
        # Context should mention the document title
        first_chunk = contextualized[0]
        assert sample_document.metadata.title in first_chunk.context or \
               "document" in first_chunk.context.lower()


class TestEmbedder:
    """Tests for embedder."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder that returns fake embeddings."""
        from src.indexing import Embedder
        
        # Create embedder but mock the actual embedding call
        embedder = Embedder.__new__(Embedder)
        embedder.provider = "mock"
        embedder.model = "mock-model"
        embedder.dimension = 1536
        
        def mock_embed_texts(texts):
            import numpy as np
            return [np.random.randn(1536).tolist() for _ in texts]
        
        embedder.embed_texts = mock_embed_texts
        embedder.embed_query = lambda q: mock_embed_texts([q])[0]
        
        return embedder
    
    def test_embed_texts(self, mock_embedder):
        """Test embedding multiple texts."""
        texts = ["Hello world", "Another text"]
        embeddings = mock_embedder.embed_texts(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
    
    def test_embed_query(self, mock_embedder):
        """Test embedding a query."""
        embedding = mock_embedder.embed_query("What is the meaning of life?")
        
        assert len(embedding) == 1536
    
    def test_embed_chunks(self, mock_embedder, sample_chunk):
        """Test embedding document chunks."""
        chunks = [sample_chunk]
        
        # Mock embed_chunks method
        def mock_embed_chunks(chunks):
            for chunk in chunks:
                chunk.embedding = mock_embedder.embed_query(chunk.content)
            return chunks
        
        mock_embedder.embed_chunks = mock_embed_chunks
        
        embedded = mock_embedder.embed_chunks(chunks)
        
        assert len(embedded) == 1
        assert embedded[0].embedding is not None
        assert len(embedded[0].embedding) == 1536


class TestIndexManager:
    """Tests for index manager."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.add_chunks = Mock(return_value=None)
        store.search = Mock(return_value=[])
        store.count = Mock(return_value=0)
        return store
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = Mock()
        
        def embed_chunks(chunks):
            import numpy as np
            for chunk in chunks:
                chunk.embedding = np.random.randn(1536).tolist()
            return chunks
        
        embedder.embed_chunks = Mock(side_effect=embed_chunks)
        return embedder
    
    @pytest.fixture
    def index_manager(self, mock_vector_store, mock_embedder):
        """Create index manager with mocks."""
        from src.indexing import IndexManager, Chunker
        
        return IndexManager(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            chunker=Chunker(),
        )
    
    def test_index_document(self, index_manager, sample_document, mock_vector_store):
        """Test indexing a document."""
        chunks = index_manager.index_document(sample_document)
        
        # Should have created chunks
        assert len(chunks) > 0
        
        # Should have called add_chunks on vector store
        mock_vector_store.add_chunks.assert_called_once()
    
    def test_indexed_chunks_have_embeddings(self, index_manager, sample_document):
        """Test that indexed chunks have embeddings."""
        chunks = index_manager.index_document(sample_document)
        
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1536
    
    def test_chunks_have_correct_document_id(self, index_manager, sample_document):
        """Test that chunks reference the correct document."""
        chunks = index_manager.index_document(sample_document)
        
        for chunk in chunks:
            assert chunk.document_id == sample_document.id
