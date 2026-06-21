from __future__ import annotations

import json
from pathlib import Path

from agents.extraction_agent import ExtractionAgent
from agents.groq_llm import GroqLLM
from app.models import DocumentChunk, ExtractedField


class ValidationAgent:
    def __init__(self, llm: GroqLLM, prompt_path: str = "prompts/validation_prompt.txt") -> None:
        self.llm = llm
        self.template = Path(prompt_path).read_text(encoding="utf-8")

    def validate(
        self,
        extraction: dict[str, ExtractedField],
        chunks: list[tuple[DocumentChunk, float]],
    ) -> dict[str, ExtractedField]:
        prompt = (
            self.template
            .replace("{{EXTRACTION}}", json.dumps({k: v.model_dump() for k, v in extraction.items()}, ensure_ascii=False))
            .replace("{{CONTEXT}}", ExtractionAgent._context(chunks))
        )
        raw = self.llm.generate_json(prompt)
        validated: dict[str, ExtractedField] = {}
        for field, extracted in extraction.items():
            verdict = raw.get(field) or {}
            accepted = bool(verdict.get("accepted", False))
            adjusted = extracted.model_copy(deep=True)
            if not accepted:
                adjusted.value = verdict.get("corrected_value")
                adjusted.confidence = min(adjusted.confidence, 0.35)
            adjusted.validation_reason = verdict.get("reason")
            adjustment = float(verdict.get("confidence_adjustment", 0.0) or 0.0)
            adjusted.confidence = max(0.0, min(1.0, adjusted.confidence + adjustment))
            validated[field] = adjusted
        return validated
