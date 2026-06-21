from __future__ import annotations

import uuid

from app.models import DocumentChunk, ParsedDocument


class ChunkingEngine:
    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for section_index, section in enumerate(document.sections):
            prefix = ""
            if section.heading:
                prefix = f"Heading: {section.heading}\n"
            if section.section_type == "table":
                prefix += "Table context:\n"
            text = (prefix + section.text).strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                chunk_id = self._chunk_id(str(document.source_path), section_index, start, chunk_text)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source_file=document.source_path.name,
                        source_path=str(document.source_path),
                        folder_name=document.folder_name,
                        page_number=section.page_number,
                        product_name=document.product_name,
                        heading=section.heading,
                        metadata={
                            "section_index": section_index,
                            "section_type": section.section_type,
                            **section.metadata,
                        },
                    )
                )
                if end == len(text):
                    break
                start = max(0, end - self.overlap)
        return chunks

    @staticmethod
    def _chunk_id(source_path: str, section_index: int, start: int, text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_path}:{section_index}:{start}:{text}"))
