#!/usr/bin/env python3
"""
Query CLI

Interactive command-line interface for querying the CASR system.
"""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from src.indexing import Embedder
from src.models.documents import SecurityClassification
from src.models.queries import QueryContext, SearchQuery
from src.models.users import User, UserRole
from src.retrieval import SecureRetriever
from src.security.policies import PolicyEngine
from src.storage import ChromaVectorStore


def create_retriever(store_type: str = "chroma", collection: str = "casr_documents"):
    """Create a retriever with the specified configuration."""
    match store_type:
        case "chroma":
            vector_store = ChromaVectorStore(collection_name=collection)
        case "pinecone":
            from src.storage import PineconeVectorStore
            vector_store = PineconeVectorStore(index_name=collection)
        case "weaviate":
            from src.storage import WeaviateVectorStore
            vector_store = WeaviateVectorStore(class_name=collection)
        case _:
            raise ValueError(f"Unknown store type: {store_type}")
    
    return SecureRetriever(
        vector_store=vector_store,
        embedder=Embedder(),
        policy_engine=PolicyEngine(),
    )


def create_user(
    role: str = "analyst",
    department: str = None,
    domains: list[str] = None,
) -> User:
    """Create a user for queries."""
    try:
        user_role = UserRole(role)
    except ValueError:
        user_role = UserRole.USER
    
    return User(
        id=uuid4(),
        username="cli-user",
        email="cli@example.com",
        role=user_role,
        department=department,
        allowed_domains=domains or [],
        is_active=True,
    )


def format_result(chunk, score: float, index: int) -> str:
    """Format a search result for display."""
    lines = [
        f"\n{'='*60}",
        f"Result {index + 1} | Score: {score:.4f}",
        f"{'='*60}",
        f"Title: {chunk.metadata.title}",
        f"Domain: {chunk.metadata.domain} | Classification: {chunk.metadata.classification.value}",
        f"Source: {chunk.metadata.source}",
        f"{'-'*60}",
    ]
    
    if chunk.context:
        lines.append(f"Context: {chunk.context}")
        lines.append(f"{'-'*60}")
    
    # Truncate content for display
    content = chunk.content
    if len(content) > 500:
        content = content[:500] + "..."
    lines.append(content)
    
    return "\n".join(lines)


def interactive_mode(retriever: SecureRetriever, user: User):
    """Run interactive query mode."""
    print("\n" + "="*60)
    print("CASR Query Interface")
    print("="*60)
    print(f"User: {user.username} | Role: {user.role.value}")
    print("Type 'quit' or 'exit' to leave")
    print("Type 'help' for commands")
    print("="*60 + "\n")
    
    while True:
        try:
            query_text = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query_text:
            continue
        
        if query_text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        if query_text.lower() == "help":
            print("\nCommands:")
            print("  quit, exit, q  - Exit the program")
            print("  help           - Show this help")
            print("  status         - Show current settings")
            print("\nOtherwise, just type your query.\n")
            continue
        
        if query_text.lower() == "status":
            print(f"\nUser: {user.username}")
            print(f"Role: {user.role.value}")
            print(f"Department: {user.department}")
            print(f"Domains: {user.allowed_domains}")
            print()
            continue
        
        # Perform search
        try:
            results = retriever.simple_retrieve(
                query_text=query_text,
                user=user,
                max_results=5,
            )
            
            if not results:
                print("\nNo results found.\n")
            else:
                print(f"\nFound {len(results)} results:")
                for i, (chunk, score) in enumerate(results):
                    print(format_result(chunk, score, i))
                print()
        
        except Exception as e:
            print(f"\nError: {e}\n")


def single_query(
    retriever: SecureRetriever,
    user: User,
    query: str,
    max_results: int = 5,
    output_format: str = "text",
):
    """Execute a single query."""
    results = retriever.simple_retrieve(
        query_text=query,
        user=user,
        max_results=max_results,
    )
    
    if output_format == "json":
        output = {
            "query": query,
            "results": [
                {
                    "score": score,
                    "chunk_id": chunk.id,
                    "document_id": str(chunk.document_id),
                    "title": chunk.metadata.title,
                    "domain": chunk.metadata.domain,
                    "classification": chunk.metadata.classification.value,
                    "content": chunk.content,
                    "context": chunk.context,
                }
                for chunk, score in results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print("No results found.")
        else:
            for i, (chunk, score) in enumerate(results):
                print(format_result(chunk, score, i))


def main():
    parser = argparse.ArgumentParser(
        description="Query the CASR system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query text (omit for interactive mode)",
    )
    
    parser.add_argument(
        "--store",
        choices=["chroma", "pinecone", "weaviate"],
        default="chroma",
        help="Vector store to use",
    )
    
    parser.add_argument(
        "--collection",
        default="casr_documents",
        help="Collection/index name",
    )
    
    parser.add_argument(
        "--role",
        choices=["guest", "user", "analyst", "manager", "executive", "admin"],
        default="analyst",
        help="User role for access control",
    )
    
    parser.add_argument(
        "--department",
        help="User department",
    )
    
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Allowed domains",
    )
    
    parser.add_argument(
        "-n", "--max-results",
        type=int,
        default=5,
        help="Maximum number of results",
    )
    
    parser.add_argument(
        "-o", "--output",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    
    args = parser.parse_args()
    
    # Create retriever and user
    retriever = create_retriever(args.store, args.collection)
    user = create_user(args.role, args.department, args.domains)
    
    if args.query:
        # Single query mode
        single_query(
            retriever,
            user,
            args.query,
            args.max_results,
            args.output,
        )
    else:
        # Interactive mode
        interactive_mode(retriever, user)


if __name__ == "__main__":
    main()
