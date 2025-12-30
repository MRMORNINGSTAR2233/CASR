# CASR: Context-Aware Secure Retrieval

A production-ready Retrieval-Augmented Generation (RAG) system designed for enterprise use cases with a focus on security, contextual retrieval, and fine-grained access control.

## 🌟 Key Features

### Security-First Architecture
- **Role-Based Access Control (RBAC)**: 7-level role hierarchy from Guest to System
- **Attribute-Based Access Control (ABAC)**: Flexible policy-based access with conditions
- **Security Classifications**: 5 levels from PUBLIC to TOP_SECRET
- **Comprehensive Audit Logging**: Full audit trail for compliance

### Advanced Retrieval
- **Contextual Retrieval**: LLM-enhanced chunk contextualization (Anthropic's approach)
- **Hybrid Search**: Combined vector + keyword search
- **Multi-Stage Reranking**: Cohere and cross-encoder support
- **Query Analysis**: Intent detection, entity extraction, query reformulation

### Enterprise Integration
- **Multi-Vector Store Support**: ChromaDB, Pinecone, Weaviate
- **Multiple Embedding Providers**: OpenAI, Sentence Transformers, Cohere
- **Multiple LLM Providers**: OpenAI, Anthropic Claude
- **RESTful API**: FastAPI with OpenAPI documentation

## 📁 Project Structure

```
CASR/
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py            # Pydantic Settings
│
├── src/
│   ├── models/                # Data models
│   │   ├── documents.py       # Document, DocumentChunk, Metadata
│   │   ├── users.py           # User, UserRole, UserSession
│   │   ├── queries.py         # SearchQuery, QueryContext, Results
│   │   └── policies.py        # AccessPolicy, PolicyCondition
│   │
│   ├── security/              # Security layer
│   │   ├── rbac.py           # Role-based access control
│   │   ├── abac.py           # Attribute-based access control
│   │   ├── policies.py       # Policy engine & store
│   │   └── audit.py          # Audit logging
│   │
│   ├── indexing/              # Document indexing
│   │   ├── chunker.py        # Document chunking strategies
│   │   ├── contextualizer.py # LLM-based contextualization
│   │   ├── embedder.py       # Multi-provider embeddings
│   │   └── index_manager.py  # Indexing orchestration
│   │
│   ├── storage/               # Vector storage
│   │   ├── base.py           # Abstract VectorStore
│   │   ├── chroma_store.py   # ChromaDB implementation
│   │   ├── pinecone_store.py # Pinecone implementation
│   │   └── weaviate_store.py # Weaviate implementation
│   │
│   ├── retrieval/             # Query & retrieval
│   │   ├── analyzer.py       # Query analysis
│   │   ├── retriever.py      # Secure retrieval
│   │   └── reranker.py       # Result reranking
│   │
│   └── api/                   # REST API
│       ├── app.py            # FastAPI application
│       ├── dependencies.py   # Dependency injection
│       ├── middleware.py     # Request logging, rate limiting
│       └── routes/           # API endpoints
│           ├── health.py
│           ├── users.py
│           ├── documents.py
│           └── queries.py
│
├── scripts/                   # Utility scripts
│   ├── setup.py              # Environment setup
│   ├── index_documents.py    # Document indexing CLI
│   └── query.py              # Query CLI
│
├── tests/                     # Test suite
│   ├── conftest.py           # Fixtures
│   ├── test_models.py
│   ├── test_security.py
│   ├── test_indexing.py
│   └── test_api.py
│
├── pyproject.toml            # Project configuration
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API keys for LLM providers (OpenAI, Anthropic, Cohere)
- Vector store (ChromaDB included, or Pinecone/Weaviate)

### Installation

```bash
# Clone the repository
git clone https://github.com/MRMORNINGSTAR2233/CASR.git
cd CASR

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
# Or: pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys

# Run setup check
python scripts/setup.py
```

### Configuration

Edit `.env` with your settings:

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional (for enhanced features)
COHERE_API_KEY=...
PINECONE_API_KEY=...
WEAVIATE_URL=http://localhost:8080

# Vector store selection
VECTOR_STORE=chroma  # or: pinecone, weaviate
```

### Index Documents

```bash
# Index a single file
python scripts/index_documents.py document.txt

# Index a directory
python scripts/index_documents.py ./docs --recursive

# With custom settings
python scripts/index_documents.py ./data \
    --domain research \
    --classification confidential \
    --chunking-strategy recursive \
    --contextualize
```

### Start the API Server

```bash
# Development mode
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.app:app --workers 4 --host 0.0.0.0 --port 8000
```

### Query via CLI

```bash
# Interactive mode
python scripts/query.py

# Single query
python scripts/query.py "What is the system architecture?"

# With specific role
python scripts/query.py --role admin "Confidential data"
```

## 📖 API Reference

### Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/v1/users/login \
    -H "Content-Type: application/json" \
    -d '{"username": "admin", "password": "admin123"}'

# Response
{
    "access_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 86400
}
```

### Search Documents

```bash
# Full search with analysis
curl -X POST http://localhost:8000/api/v1/queries/search \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "How does the security system work?",
        "max_results": 10,
        "use_reranking": true,
        "use_hybrid": true
    }'

# Quick search
curl -X POST http://localhost:8000/api/v1/queries/quick \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"query": "security architecture", "limit": 5}'
```

### Upload Documents

```bash
# Create document
curl -X POST http://localhost:8000/api/v1/documents \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "title": "Security Guidelines",
        "content": "...",
        "domain": "security",
        "classification": "confidential"
    }'

# Upload file
curl -X POST http://localhost:8000/api/v1/documents/upload \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@document.pdf" \
    -F "domain=research" \
    -F "classification=internal"
```

## 🔐 Security Model

### Role Hierarchy

| Role | Permissions | Max Classification |
|------|-------------|-------------------|
| Guest | Read public | PUBLIC |
| User | Read, search | INTERNAL |
| Analyst | + analyze, export | CONFIDENTIAL |
| Manager | + write, share | CONFIDENTIAL |
| Executive | + approve | SECRET |
| Admin | All | TOP_SECRET |
| System | All (internal) | TOP_SECRET |

### ABAC Policies

```python
from src.models.policies import AccessPolicy, PolicyEffect, PolicyAction, PolicyCondition

policy = AccessPolicy(
    name="Research Department Access",
    effect=PolicyEffect.ALLOW,
    subjects={"role": ["analyst", "manager"]},
    resources={"domain": ["research"]},
    actions=[PolicyAction.READ, PolicyAction.SEARCH],
    conditions=[
        PolicyCondition(
            attribute="department",
            operator=ConditionOperator.EQUALS,
            value="Research"
        ),
        PolicyCondition(
            attribute="time",
            operator=ConditionOperator.BETWEEN,
            value=["09:00", "18:00"]
        )
    ]
)
```

## 🔬 Technical Details

### Contextual Retrieval

CASR implements Anthropic's Contextual Retrieval approach:

1. **Chunking**: Documents are split using recursive text splitter
2. **Contextualization**: Each chunk is sent to an LLM with the full document context
3. **Context Prepending**: Generated context is prepended to the chunk
4. **Embedding**: Enhanced chunks are embedded for better retrieval

```python
# The system automatically generates context like:
# "This section discusses the authentication flow in the CASR system,
#  specifically how JWT tokens are validated and user sessions managed."
```

### Query Analysis

The query analyzer extracts:

- **Intent**: factual, analytical, procedural, definitional, exploratory, troubleshooting, summarization
- **Entities**: People, organizations, products, concepts
- **Keywords**: Important terms for retrieval
- **Domain**: Relevant business domain
- **Ambiguity**: Detection and disambiguation options
- **Security**: Minimum clearance needed

### Hybrid Search

For vector stores that support it (Weaviate), CASR uses native hybrid search combining:

- **Vector Search**: Semantic similarity via embeddings
- **Keyword Search**: BM25/TF-IDF for exact matching
- **Fusion**: Configurable alpha parameter (0=keyword, 1=vector)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_security.py -v

# Run specific test
pytest tests/test_security.py::TestRBACEngine::test_check_permission -v
```

## 🐳 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t casr .
docker run -p 8000:8000 --env-file .env casr
```

## 📊 Performance Considerations

### Embedding Caching
- Embeddings are cached in the vector store
- Re-indexing only affects changed documents

### Batch Processing
- Documents are processed in configurable batch sizes
- Async support for concurrent operations

### Rate Limiting
- Configurable per-minute rate limits
- Per-client tracking (IP or token-based)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Inspiration for contextualization approach
- [CRAG Paper](https://arxiv.org/abs/2401.15884) - Corrective RAG concepts
- [LangChain](https://python.langchain.com/) - RAG building blocks
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**CASR** - Secure, Context-Aware Retrieval for the Enterprise
