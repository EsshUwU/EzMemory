"""
Data schemas for memory records.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from ..utils.time import get_current_timestamp, timestamp_to_string


class MemoryMetadata(BaseModel):
    """Metadata for a memory record."""
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    source: Optional[str] = Field(None, description="Source of the memory")
    confidence: float = Field(1.0, description="Confidence score (0-1)")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class Memory(BaseModel):
    """A memory record."""
    id: Optional[str] = Field(None, description="Unique identifier")
    content: str = Field(..., description="Memory content text")
    embedding: Optional[list[float]] = Field(None, description="Embedding vector")
    created_at: datetime = Field(default_factory=get_current_timestamp, description="Creation timestamp")
    accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
    access_count: int = Field(0, description="Number of times accessed")
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: timestamp_to_string(v)
        }
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to vector store payload."""
        return {
            "content": self.content,
            "created_at": timestamp_to_string(self.created_at),
            "accessed_at": timestamp_to_string(self.accessed_at) if self.accessed_at else None,
            "access_count": self.access_count,
            "metadata": self.metadata.model_dump(),
        }
    
    @classmethod
    def from_payload(cls, memory_id: str, payload: Dict[str, Any]) -> "Memory":
        """Create Memory from vector store payload."""
        from ..utils.time import string_to_timestamp
        
        return cls(
            id=memory_id,
            content=payload["content"],
            created_at=string_to_timestamp(payload["created_at"]),
            accessed_at=string_to_timestamp(payload["accessed_at"]) if payload.get("accessed_at") else None,
            access_count=payload.get("access_count", 0),
            metadata=MemoryMetadata(**payload.get("metadata", {})),
        )


class SearchResult(BaseModel):
    """A search result."""
    memory: Memory
    score: float = Field(..., description="Similarity score")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.memory.id,
            "content": self.memory.content,
            "score": self.score,
            "created_at": timestamp_to_string(self.memory.created_at),
            "metadata": self.memory.metadata.model_dump(),
        }
