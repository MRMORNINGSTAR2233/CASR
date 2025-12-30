#!/usr/bin/env python3
"""
CASR Setup Script

Sets up the CASR environment and verifies configuration.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check required dependencies."""
    required = [
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("chromadb", "ChromaDB"),
        ("tiktoken", "Tiktoken"),
        ("structlog", "Structlog"),
    ]
    
    optional = [
        ("cohere", "Cohere"),
        ("google.generativeai", "Google Gemini"),
        ("groq", "Groq"),
        ("pinecone", "Pinecone"),
        ("weaviate", "Weaviate"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    all_ok = True
    
    print("\nRequired dependencies:")
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} - not installed")
            all_ok = False
    
    print("\nOptional dependencies:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} - not installed (optional)")
    
    return all_ok


def check_env_file():
    """Check for .env file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("\n✓ .env file found")
        return True
    
    if env_example.exists():
        print("\n○ .env file not found, but .env.example exists")
        print("  Run: cp .env.example .env")
        return False
    
    print("\n❌ No .env or .env.example file found")
    return False


def check_api_keys():
    """Check for API keys."""
    print("\nAPI Keys:")
    
    keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "COHERE_API_KEY": "Cohere",
        "GEMINI_API_KEY": "Google Gemini",
        "GROQ_API_KEY": "Groq",
        "PINECONE_API_KEY": "Pinecone",
    }
    
    for env_var, name in keys.items():
        value = os.environ.get(env_var, "")
        if value and len(value) > 10:
            print(f"  ✓ {name} - configured")
        else:
            print(f"  ○ {name} - not configured")


def check_directories():
    """Check and create necessary directories."""
    print("\nDirectories:")
    
    dirs = [
        Path(".chroma"),
        Path("logs"),
        Path("data"),
    ]
    
    for dir_path in dirs:
        if dir_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  + {dir_path}/ (created)")


def test_imports():
    """Test that core modules can be imported."""
    print("\nModule imports:")
    
    modules = [
        ("config", "Configuration"),
        ("src.models", "Data Models"),
        ("src.security", "Security Layer"),
        ("src.indexing", "Indexing System"),
        ("src.storage", "Storage Layer"),
        ("src.retrieval", "Retrieval System"),
        ("src.api", "API Layer"),
    ]
    
    all_ok = True
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False
    
    return all_ok


def create_sample_data():
    """Create sample data for testing."""
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample document
    sample_doc = data_dir / "sample_document.txt"
    if not sample_doc.exists():
        sample_doc.write_text("""# Sample Document

This is a sample document for testing the CASR system.

## Overview

Context-Aware Secure Retrieval (CASR) is a RAG system designed for enterprise use cases.
It features:

- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Contextual retrieval with LLM-enhanced chunks
- Multi-vector store support
- Hybrid search capabilities

## Getting Started

1. Configure your environment variables
2. Index your documents
3. Start the API server
4. Query the system

## Security

All queries are filtered based on user permissions. The system supports
multiple security classification levels from PUBLIC to TOP_SECRET.
""")
        print("  + Created sample document")
    else:
        print("  ✓ Sample data exists")


def main():
    print("=" * 50)
    print("CASR Setup Check")
    print("=" * 50)
    
    # Run checks
    py_ok = check_python_version()
    deps_ok = check_dependencies()
    env_ok = check_env_file()
    check_api_keys()
    check_directories()
    imports_ok = test_imports()
    create_sample_data()
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    if py_ok and deps_ok and imports_ok:
        print("✓ Core setup is complete!")
        if not env_ok:
            print("\nNote: Configure .env file for full functionality")
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and add your API keys")
        print("  2. Index some documents:")
        print("     python scripts/index_documents.py data/sample")
        print("  3. Start the API server:")
        print("     uvicorn src.api.app:app --reload")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
