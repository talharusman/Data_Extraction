from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from agents.groq_llm import GroqLLM
from app.models import DocumentChunk, ExtractedField

logger = logging.getLogger(__name__)

FIELD_BATCH_SIZE = 6


class ExtractionAgent:
    def __init__(self, llm: GroqLLM, prompt_path: str = "prompts/extraction_prompt.txt") -> None:
        self.llm = llm
        self.template = Path(prompt_path).read_text(encoding="utf-8")

    def extract(self, fields: list[str], chunks: list[tuple[DocumentChunk, float]]) -> dict[str, ExtractedField]:
        result: dict[str, ExtractedField] = {}
        for batch_start in range(0, len(fields), FIELD_BATCH_SIZE):
            batch = fields[batch_start : batch_start + FIELD_BATCH_SIZE]
            result.update(self._extract_batch(batch, chunks))
        return result

    def _extract_batch(self, fields: list[str], chunks: list[tuple[DocumentChunk, float]]) -> dict[str, ExtractedField]:
        prompt = (
            self.template
            .replace("{{FIELDS}}", "\n".join(fields))
            .replace("{{CONTEXT}}", self._context(chunks))
        )
        raw = self.llm.generate_json(prompt)
        result: dict[str, ExtractedField] = {}

        for field in fields:
            value = raw.get(field)
            if not isinstance(value, dict):
                result[field] = ExtractedField()
                continue
            try:
                result[field] = ExtractedField(**value)
            except ValidationError as exc:
                logger.warning("Malformed extraction for %s: %s", field, exc)
                result[field] = ExtractedField(
                    value=value.get("value"),
                    confidence=float(value.get("confidence", 0.0) or 0.0),
                    source_chunk=value.get("source_chunk"),
                    source_page=value.get("source_page"),
                    source_file=value.get("source_file"),
                    evidence=value.get("evidence"),
                )
        return result

    @staticmethod
    def _context(chunks: list[tuple[DocumentChunk, float]]) -> str:
        parts = []
        for chunk, score in chunks[:5]:
            text = chunk.text.strip()
            if len(text) > 800:
                text = text[:800].rstrip() + "..."
            parts.append(
                f"CHUNK_ID: {chunk.chunk_id}\n"
                f"SCORE: {score:.4f}\n"
                f"SOURCE_FILE: {chunk.source_file}\n"
                f"PAGE: {chunk.page_number}\n"
                f"TEXT:\n{text}"
            )
        return "\n\n---\n\n".join(parts)
