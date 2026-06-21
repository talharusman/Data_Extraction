from __future__ import annotations

from pathlib import Path

from agents.groq_llm import GroqLLM
from app.models import DocumentChunk, ExtractedField


class ExtractionAgent:
    def __init__(self, llm: GroqLLM, prompt_path: str = "prompts/extraction_prompt.txt") -> None:
        self.llm = llm
        self.template = Path(prompt_path).read_text(encoding="utf-8")

    def extract(self, fields: list[str], chunks: list[tuple[DocumentChunk, float]]) -> dict[str, ExtractedField]:
        prompt = (
            self.template
            .replace("{{FIELDS}}", "\n".join(fields))
            .replace("{{CONTEXT}}", self._context(chunks))
        )
        raw = self.llm.generate_json(prompt)
        result: dict[str, ExtractedField] = {}
        for field in fields:
            value = raw.get(field) or {}
            result[field] = ExtractedField(**value)
        return result

    @staticmethod
    def _context(chunks: list[tuple[DocumentChunk, float]]) -> str:
        parts = []
        for chunk, score in chunks[:3]:
            text = chunk.text.strip()
            if len(text) > 600:
                text = text[:600].rstrip() + "..."
            parts.append(
                f"CHUNK_ID: {chunk.chunk_id}\n"
                f"SCORE: {score:.4f}\n"
                f"SOURCE_FILE: {chunk.source_file}\n"
                f"PAGE: {chunk.page_number}\n"
                f"TEXT:\n{text}"
            )
        return "\n\n---\n\n".join(parts)
