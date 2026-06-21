from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models import DocumentSection, ParsedDocument
from parsers.base import BaseParser


class JsonParser(BaseParser):
    def parse(self, path: Path) -> ParsedDocument:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            data = json.load(handle)
        sections = [
            DocumentSection(
                text="\n".join(self._flatten(data)),
                heading="JSON content",
                section_type="json",
                metadata={"json_root_type": type(data).__name__},
            )
        ]
        return ParsedDocument(
            source_path=path,
            folder_name=path.parent.name,
            product_name=self._product_name(data) or path.stem,
            sections=sections,
        )

    def _flatten(self, value: Any, prefix: str = "") -> list[str]:
        lines: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                lines.extend(self._flatten(item, next_prefix))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                lines.extend(self._flatten(item, f"{prefix}[{index}]"))
        else:
            lines.append(f"{prefix}: {value}")
        return lines

    @staticmethod
    def _product_name(data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        for key in ("PRODUCT_NAME", "product_name", "name", "title"):
            if key in data and data[key]:
                return str(data[key])
        return None
