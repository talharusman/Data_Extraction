from __future__ import annotations

from pathlib import Path

from parsers.base import BaseParser
from parsers.docx_parser import DocxParser
from parsers.json_parser import JsonParser
from parsers.pdf_parser import PdfParser
from parsers.txt_parser import TxtParser


class ParserFactory:
    def __init__(self) -> None:
        self.parsers: dict[str, BaseParser] = {
            ".json": JsonParser(),
            ".pdf": PdfParser(),
            ".docx": DocxParser(),
            ".txt": TxtParser(),
        }

    def get_parser(self, path: Path) -> BaseParser:
        suffix = path.suffix.lower()
        if suffix not in self.parsers:
            raise ValueError(f"Unsupported file type: {suffix}")
        return self.parsers[suffix]
