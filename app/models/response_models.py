from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChunkMetadata(BaseModel):
    document_id: str
    section_title: str
    section_path: List[str]
    currency: List[str]
    amounts: List[float]
    Dates: List[str]
    document_type: str

class ChunkAnchors(BaseModel):
    page_range: List[int]
    line_refs: List[int]

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    anchors: ChunkAnchors

class DocumentResponse(BaseModel):
    document_id: str
    status: str
    chunks: List[DocumentChunk]