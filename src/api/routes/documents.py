"""
Document Routes

Document upload, indexing, and management endpoints.
"""

import hashlib
from datetime import datetime
from typing import Annotated, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_current_user,
    get_index_manager,
    get_policy_engine,
    get_vector_store,
    require_role,
)
from src.indexing import IndexManager
from src.models.documents import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    SecurityClassification,
)
from src.models.policies import PolicyAction
from src.models.users import User, UserRole
from src.security.policies import PolicyEngine
from src.storage.base import VectorStore

router = APIRouter()


# Request/Response Models
class DocumentCreateRequest(BaseModel):
    """Document creation request."""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: str = "manual"
    domain: str = "general"
    classification: SecurityClassification = SecurityClassification.PUBLIC
    tags: list[str] = Field(default_factory=list)
    custom_metadata: dict = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document response model."""
    id: UUID
    title: str
    source: str
    domain: str
    classification: str
    chunk_count: int
    created_at: datetime
    tags: list[str]


class DocumentDetailResponse(DocumentResponse):
    """Detailed document response."""
    content_preview: str
    chunks: list[dict]


class ChunkResponse(BaseModel):
    """Chunk response model."""
    id: str
    content: str
    chunk_index: int
    has_context: bool
    token_count: int


class IndexingResult(BaseModel):
    """Result of document indexing."""
    document_id: UUID
    chunks_created: int
    embeddings_generated: int
    indexing_time_ms: float


# Document storage (replace with real database)
_documents_db: dict[str, Document] = {}


@router.post("", response_model=IndexingResult, status_code=status.HTTP_201_CREATED)
async def create_document(
    request: DocumentCreateRequest,
    user: Annotated[User, Depends(get_current_user)],
    index_manager: IndexManager = Depends(get_index_manager),
) -> IndexingResult:
    """
    Create and index a new document.
    
    The document will be chunked, contextualized, embedded, and stored.
    """
    import time
    start_time = time.time()
    
    # Create document
    doc_id = uuid4()
    
    document = Document(
        id=doc_id,
        content=request.content,
        metadata=DocumentMetadata(
            title=request.title,
            source=request.source,
            domain=request.domain,
            classification=request.classification,
            tags=request.tags,
            custom=request.custom_metadata,
            owner_id=str(user.id),
            allowed_roles=[user.role.value],
        ),
    )
    
    # Index document
    chunks = index_manager.index_document(document)
    
    # Store document reference
    _documents_db[str(doc_id)] = document
    
    indexing_time = (time.time() - start_time) * 1000
    
    return IndexingResult(
        document_id=doc_id,
        chunks_created=len(chunks),
        embeddings_generated=len(chunks),
        indexing_time_ms=indexing_time,
    )


@router.post("/upload", response_model=IndexingResult, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    domain: str = Form("general"),
    classification: str = Form("public"),
    tags: str = Form(""),
    user: User = Depends(get_current_user),
    index_manager: IndexManager = Depends(get_index_manager),
) -> IndexingResult:
    """
    Upload and index a document file.
    
    Supports: .txt, .md, .pdf (text extraction), .docx (coming soon)
    """
    import time
    start_time = time.time()
    
    # Read file content
    content = await file.read()
    
    # Determine file type and extract text
    filename = file.filename or "unknown"
    
    if filename.endswith(".txt") or filename.endswith(".md"):
        text_content = content.decode("utf-8")
    elif filename.endswith(".pdf"):
        # Placeholder for PDF extraction
        # In production, use pypdf or pdfplumber
        try:
            import pypdf
            from io import BytesIO
            reader = pypdf.PdfReader(BytesIO(content))
            text_content = "\n".join(
                page.extract_text() for page in reader.pages
            )
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF support requires pypdf package",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {filename}",
        )
    
    # Parse classification
    try:
        sec_class = SecurityClassification(classification)
    except ValueError:
        sec_class = SecurityClassification.PUBLIC
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    
    # Create document
    doc_id = uuid4()
    document = Document(
        id=doc_id,
        content=text_content,
        metadata=DocumentMetadata(
            title=title or filename,
            source=f"upload:{filename}",
            domain=domain,
            classification=sec_class,
            tags=tag_list,
            owner_id=str(user.id),
            allowed_roles=[user.role.value],
        ),
    )
    
    # Index document
    chunks = index_manager.index_document(document)
    
    # Store document reference
    _documents_db[str(doc_id)] = document
    
    indexing_time = (time.time() - start_time) * 1000
    
    return IndexingResult(
        document_id=doc_id,
        chunks_created=len(chunks),
        embeddings_generated=len(chunks),
        indexing_time_ms=indexing_time,
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    user: Annotated[User, Depends(get_current_user)],
    policy_engine: PolicyEngine = Depends(get_policy_engine),
    domain: Optional[str] = Query(None),
    classification: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> list[DocumentResponse]:
    """
    List documents accessible to the current user.
    """
    documents = []
    
    for doc in list(_documents_db.values())[skip:skip + limit]:
        # Check access
        allowed, _ = policy_engine.evaluate_access(
            user=user,
            resource=doc.metadata,
            action=PolicyAction.READ,
        )
        
        if not allowed:
            continue
        
        # Apply filters
        if domain and doc.metadata.domain != domain:
            continue
        if classification:
            try:
                if doc.metadata.classification != SecurityClassification(classification):
                    continue
            except ValueError:
                pass
        
        documents.append(
            DocumentResponse(
                id=doc.id,
                title=doc.metadata.title,
                source=doc.metadata.source,
                domain=doc.metadata.domain,
                classification=doc.metadata.classification.value,
                chunk_count=doc.chunk_count,
                created_at=doc.metadata.created_at,
                tags=doc.metadata.tags,
            )
        )
    
    return documents


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    policy_engine: PolicyEngine = Depends(get_policy_engine),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentDetailResponse:
    """
    Get document details by ID.
    """
    doc = _documents_db.get(str(document_id))
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Check access
    allowed, _ = policy_engine.evaluate_access(
        user=user,
        resource=doc.metadata,
        action=PolicyAction.READ,
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    
    # Get chunks from vector store
    chunks = vector_store.get_chunks_by_document(document_id)
    chunk_data = [
        {
            "id": chunk.id,
            "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]
    
    return DocumentDetailResponse(
        id=doc.id,
        title=doc.metadata.title,
        source=doc.metadata.source,
        domain=doc.metadata.domain,
        classification=doc.metadata.classification.value,
        chunk_count=len(chunks),
        created_at=doc.metadata.created_at,
        tags=doc.metadata.tags,
        content_preview=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
        chunks=chunk_data,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    user: Annotated[User, Depends(require_role(UserRole.MANAGER))],
    policy_engine: PolicyEngine = Depends(get_policy_engine),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Delete a document and its chunks (manager+ only).
    """
    doc = _documents_db.get(str(document_id))
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Check access
    allowed, _ = policy_engine.evaluate_access(
        user=user,
        resource=doc.metadata,
        action=PolicyAction.DELETE,
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    
    # Delete from vector store
    vector_store.delete_by_document_id(document_id)
    
    # Delete from document storage
    del _documents_db[str(document_id)]


@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(
    document_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    policy_engine: PolicyEngine = Depends(get_policy_engine),
    vector_store: VectorStore = Depends(get_vector_store),
) -> list[ChunkResponse]:
    """
    Get all chunks for a document.
    """
    doc = _documents_db.get(str(document_id))
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Check access
    allowed, _ = policy_engine.evaluate_access(
        user=user,
        resource=doc.metadata,
        action=PolicyAction.READ,
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    
    chunks = vector_store.get_chunks_by_document(document_id)
    
    return [
        ChunkResponse(
            id=chunk.id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            has_context=chunk.context is not None,
            token_count=chunk.token_count,
        )
        for chunk in sorted(chunks, key=lambda c: c.chunk_index)
    ]


@router.post("/batch", response_model=list[IndexingResult], status_code=status.HTTP_201_CREATED)
async def batch_create_documents(
    documents: list[DocumentCreateRequest],
    user: Annotated[User, Depends(require_role(UserRole.MANAGER))],
    index_manager: IndexManager = Depends(get_index_manager),
) -> list[IndexingResult]:
    """
    Create and index multiple documents in batch (manager+ only).
    """
    import time
    
    results = []
    
    for request in documents:
        start_time = time.time()
        
        doc_id = uuid4()
        document = Document(
            id=doc_id,
            content=request.content,
            metadata=DocumentMetadata(
                title=request.title,
                source=request.source,
                domain=request.domain,
                classification=request.classification,
                tags=request.tags,
                custom=request.custom_metadata,
                owner_id=str(user.id),
                allowed_roles=[user.role.value],
            ),
        )
        
        chunks = index_manager.index_document(document)
        _documents_db[str(doc_id)] = document
        
        indexing_time = (time.time() - start_time) * 1000
        
        results.append(
            IndexingResult(
                document_id=doc_id,
                chunks_created=len(chunks),
                embeddings_generated=len(chunks),
                indexing_time_ms=indexing_time,
            )
        )
    
    return results
