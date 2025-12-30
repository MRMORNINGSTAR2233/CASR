"""
CASR Settings Configuration

Centralized configuration using Pydantic Settings with environment variable support.
"""

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    COHERE = "cohere"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Supports .env file loading and provides sensible defaults for development.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==========================================================================
    # LLM Provider Configuration
    # ==========================================================================
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for embeddings and LLM"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for query analysis"
    )
    
    cohere_api_key: Optional[str] = Field(
        default=None,
        description="Cohere API key for reranking"
    )
    
    # ==========================================================================
    # Vector Store Configuration
    # ==========================================================================
    
    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.CHROMA,
        description="Vector store backend to use"
    )
    
    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory"
    )
    
    # Pinecone
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key"
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment"
    )
    pinecone_index_name: str = Field(
        default="casr-index",
        description="Pinecone index name"
    )
    
    # Weaviate
    weaviate_url: str = Field(
        default="http://localhost:8080",
        description="Weaviate server URL"
    )
    weaviate_api_key: Optional[str] = Field(
        default=None,
        description="Weaviate API key"
    )
    
    # ==========================================================================
    # Embedding Configuration
    # ==========================================================================
    
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="Embedding provider to use"
    )
    
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformers model name"
    )
    
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension"
    )
    
    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    
    jwt_secret_key: str = Field(
        default="change-me-in-production-use-strong-secret",
        description="JWT signing secret key"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        description="JWT access token expiration in minutes"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key for service authentication"
    )
    
    # ==========================================================================
    # Application Configuration
    # ==========================================================================
    
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    
    api_port: int = Field(
        default=8000,
        description="API server port"
    )
    
    api_debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # ==========================================================================
    # Retrieval Configuration
    # ==========================================================================
    
    retrieval_top_k: int = Field(
        default=20,
        description="Number of chunks to retrieve initially"
    )
    
    rerank_top_k: int = Field(
        default=5,
        description="Number of chunks after reranking"
    )
    
    chunk_size: int = Field(
        default=512,
        description="Document chunk size in tokens"
    )
    
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks in tokens"
    )
    
    enable_reranking: bool = Field(
        default=True,
        description="Enable reranking step"
    )
    
    enable_contextual_embeddings: bool = Field(
        default=True,
        description="Enable contextual embeddings"
    )
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
