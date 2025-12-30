#!/usr/bin/env python3
"""
Index Documents Script

Command-line tool for indexing documents into the CASR system.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from src.indexing import Chunker, Contextualizer, Embedder, IndexManager
from src.models.documents import Document, DocumentMetadata, SecurityClassification
from src.storage import ChromaVectorStore, PineconeVectorStore, WeaviateVectorStore


def get_vector_store(store_type: str, **kwargs):
    """Create vector store based on type."""
    match store_type:
        case "chroma":
            return ChromaVectorStore(
                collection_name=kwargs.get("collection", "casr_documents"),
                persist_directory=kwargs.get("persist_dir", "./.chroma"),
            )
        case "pinecone":
            return PineconeVectorStore(
                index_name=kwargs.get("index", "casr-documents"),
            )
        case "weaviate":
            return WeaviateVectorStore(
                class_name=kwargs.get("class_name", "CASRDocument"),
                url=kwargs.get("url", "http://localhost:8080"),
            )
        case _:
            raise ValueError(f"Unknown store type: {store_type}")


def read_document(path: Path) -> tuple[str, str]:
    """Read document content and detect format."""
    suffix = path.suffix.lower()
    
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8"), "text"
    
    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text = "\n".join(page.extract_text() for page in reader.pages)
            return text, "pdf"
        except ImportError:
            print("Warning: pypdf not installed, skipping PDF")
            return "", "pdf"
    
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("content", json.dumps(data)), "json"
        return json.dumps(data), "json"
    
    # Try as text
    try:
        return path.read_text(encoding="utf-8"), "unknown"
    except UnicodeDecodeError:
        print(f"Warning: Cannot read {path} as text")
        return "", "binary"


def index_file(
    path: Path,
    index_manager: IndexManager,
    domain: str = "general",
    classification: SecurityClassification = SecurityClassification.PUBLIC,
    owner: str = "system",
    verbose: bool = False,
) -> Optional[Document]:
    """Index a single file."""
    content, doc_type = read_document(path)
    
    if not content:
        if verbose:
            print(f"  Skipping empty file: {path}")
        return None
    
    document = Document(
        content=content,
        metadata=DocumentMetadata(
            title=path.stem,
            source=f"file:{path.name}",
            domain=domain,
            classification=classification,
            tags=[doc_type, path.suffix.lstrip(".")],
            owner_id=owner,
        ),
    )
    
    chunks = index_manager.index_document(document)
    
    if verbose:
        print(f"  Indexed: {path.name} ({len(chunks)} chunks)")
    
    return document


def index_directory(
    directory: Path,
    index_manager: IndexManager,
    recursive: bool = True,
    extensions: Optional[list[str]] = None,
    domain: str = "general",
    classification: SecurityClassification = SecurityClassification.PUBLIC,
    verbose: bool = False,
) -> list[Document]:
    """Index all documents in a directory."""
    extensions = extensions or [".txt", ".md", ".pdf", ".json"]
    documents = []
    
    pattern = "**/*" if recursive else "*"
    
    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            doc = index_file(
                path,
                index_manager,
                domain=domain,
                classification=classification,
                verbose=verbose,
            )
            if doc:
                documents.append(doc)
    
    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Index documents into the CASR system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a single file
  python index_documents.py file.txt

  # Index a directory
  python index_documents.py ./docs --recursive

  # Index with specific settings
  python index_documents.py ./data --domain research --classification confidential

  # Use Pinecone instead of ChromaDB
  python index_documents.py ./docs --store pinecone --index my-index
        """,
    )
    
    parser.add_argument(
        "path",
        type=Path,
        help="Path to file or directory to index",
    )
    
    parser.add_argument(
        "--store",
        choices=["chroma", "pinecone", "weaviate"],
        default="chroma",
        help="Vector store to use (default: chroma)",
    )
    
    parser.add_argument(
        "--collection",
        default="casr_documents",
        help="Collection/index name",
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./.chroma",
        help="Persistence directory for ChromaDB",
    )
    
    parser.add_argument(
        "--domain",
        default="general",
        help="Domain for documents",
    )
    
    parser.add_argument(
        "--classification",
        choices=["public", "internal", "confidential", "secret", "top_secret"],
        default="public",
        help="Security classification",
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively index directories",
    )
    
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt", ".md", ".pdf", ".json"],
        help="File extensions to include",
    )
    
    parser.add_argument(
        "--chunking-strategy",
        choices=["fixed_size", "semantic", "sentence", "recursive"],
        default="recursive",
        help="Chunking strategy",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens",
    )
    
    parser.add_argument(
        "--contextualize",
        action="store_true",
        help="Add contextual information to chunks",
    )
    
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "sentence-transformers", "cohere"],
        default="openai",
        help="Embedding provider",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Validate path
    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)
    
    # Create components
    print(f"Initializing with {args.store} vector store...")
    
    vector_store = get_vector_store(
        args.store,
        collection=args.collection,
        persist_dir=args.persist_dir,
    )
    
    chunker = Chunker(
        strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
    )
    
    embedder = Embedder(provider=args.embedding_provider)
    
    contextualizer = None
    if args.contextualize:
        contextualizer = Contextualizer()
    
    index_manager = IndexManager(
        vector_store=vector_store,
        chunker=chunker,
        embedder=embedder,
        contextualizer=contextualizer,
    )
    
    # Parse classification
    classification = SecurityClassification(args.classification)
    
    # Index documents
    print(f"Indexing documents from: {args.path}")
    
    if args.path.is_file():
        doc = index_file(
            args.path,
            index_manager,
            domain=args.domain,
            classification=classification,
            verbose=args.verbose,
        )
        documents = [doc] if doc else []
    else:
        documents = index_directory(
            args.path,
            index_manager,
            recursive=args.recursive,
            extensions=args.extensions,
            domain=args.domain,
            classification=classification,
            verbose=args.verbose,
        )
    
    # Summary
    total_chunks = sum(doc.chunk_count for doc in documents)
    print(f"\nIndexing complete!")
    print(f"  Documents indexed: {len(documents)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Vector store: {args.store}")
    print(f"  Collection: {args.collection}")


if __name__ == "__main__":
    main()
