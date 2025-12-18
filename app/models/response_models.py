from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Sub-models for Extractions ---

class EntityExtraction(BaseModel):
    id: str
    value: str
    reliability: float
    section_id: str
    block_id: str
    page: int
    source: str = "regex"

class Extractions(BaseModel):
    emails: List[EntityExtraction] = []
    phone_numbers: List[EntityExtraction] = []
    dates: List[EntityExtraction] = []
    currency: List[EntityExtraction] = []

# --- Sub-models for Content Structure ---

class ContentBlock(BaseModel):
    id: str
    type: str = "paragraph"
    text: str
    page: int

class ExtractionRefs(BaseModel):
    emails: List[str] = []
    phone_numbers: List[str] = []
    dates: List[str] = []
    currency: List[str] = []

class Section(BaseModel):
    id: str
    title: str
    level: int = 1
    page_start: int
    page_end: int
    content: List[ContentBlock]
    extraction_refs: ExtractionRefs

# --- Sub-models for Metadata and Removed Content ---

class DocMetadata(BaseModel):
    source_type: str
    page_count: int
    language: str
    processed_at: str
    line_count: int
    content_line_count: int
    title_count: int
    header_footer_count: int

class RemovedItem(BaseModel):
    text: str
    pages: List[int]
    frequency: int

class RemovedContent(BaseModel):
    headers: List[RemovedItem]
    footers: List[RemovedItem]

# --- Main Response Model ---

class DocumentResponse(BaseModel):
    filename: str
    status: str
    version: str = "1.0"
    document_id: str
    metadata: DocMetadata
    sections: List[Section]
    extractions: Extractions
    removed: RemovedContent
    markdown: Optional[str] = None