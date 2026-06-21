from __future__ import annotations

import logging
from pathlib import Path

from app.models import DocumentSection, ParsedDocument
from parsers.base import BaseParser

logger = logging.getLogger(__name__)


class PdfParser(BaseParser):
    def parse(self, path: Path) -> ParsedDocument:
        sections = self._parse_with_pypdf(path)
        return ParsedDocument(
            source_path=path,
            folder_name=path.parent.name,
            product_name=path.stem,
            sections=sections,
        )

    def _parse_with_pypdf(self, path: Path) -> list[DocumentSection]:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            sections: list[DocumentSection] = []
            for index, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    sections.append(DocumentSection(text=text, page_number=index, section_type="pdf_page"))
            return sections
        except Exception as exc:
            logger.warning("PDF parsing failed for %s: %s", path, exc)
            return []
