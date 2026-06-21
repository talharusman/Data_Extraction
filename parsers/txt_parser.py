from __future__ import annotations

from pathlib import Path

from app.models import DocumentSection, ParsedDocument
from parsers.base import BaseParser


class TxtParser(BaseParser):
    def parse(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return ParsedDocument(
            source_path=path,
            folder_name=path.parent.name,
            product_name=path.stem,
            sections=[DocumentSection(text=text, section_type="text")],
        )
