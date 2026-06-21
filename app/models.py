from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    text: str
    heading: str | None = None
    page_number: int | None = None
    section_type: str = "paragraph"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    source_path: Path
    folder_name: str
    product_name: str | None = None
    sections: list[DocumentSection]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    source_path: str
    folder_name: str
    page_number: int | None = None
    product_name: str | None = None
    heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedField(BaseModel):
    value: Any = None
    confidence: float = 0.0
    source_chunk: str | None = None
    source_page: int | None = None
    source_file: str | None = None
    evidence: str | None = None
    validation_reason: str | None = None


class ProductExtraction(BaseModel):
    product_id: str
    source_file: str
    source_path: str
    fields: dict[str, ExtractedField]
    needs_review: bool = False


class ProcessingStatus(BaseModel):
    state: str = "idle"
    files_discovered: int = 0
    files_processed: int = 0
    products_extracted: int = 0
    failures: list[str] = Field(default_factory=list)
