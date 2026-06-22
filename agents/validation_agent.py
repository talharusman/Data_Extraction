from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.extraction_agent import ExtractionAgent, FIELD_BATCH_SIZE
from agents.local_llm import LocalGGUFLLM
from app.models import DocumentChunk, ExtractedField

logger = logging.getLogger(__name__)


class ValidationAgent:
    def __init__(self, llm: LocalGGUFLLM, prompt_path: str = "prompts/validation_prompt.txt") -> None:
        self.llm = llm
        self.template = Path(prompt_path).read_text(encoding="utf-8")

    def validate(
        self,
        extraction: dict[str, ExtractedField],
        chunks: list[tuple[DocumentChunk, float]],
    ) -> dict[str, ExtractedField]:
        if not extraction:
            return extraction

        populated = {
            field: extracted
            for field, extracted in extraction.items()
            if extracted.value not in (None, "")
        }
        if not populated:
            return {field: extracted.model_copy(deep=True) for field, extracted in extraction.items()}

        validated = {field: extracted.model_copy(deep=True) for field, extracted in extraction.items()}
        field_names = list(populated.keys())
        for batch_start in range(0, len(field_names), FIELD_BATCH_SIZE):
            batch_fields = field_names[batch_start : batch_start + FIELD_BATCH_SIZE]
            batch = {field: populated[field] for field in batch_fields}
            batch_verdicts = self._validate_batch(batch, chunks)
            for field, extracted in batch.items():
                verdict = batch_verdicts.get(field)
                adjusted = extracted.model_copy(deep=True)
                if not isinstance(verdict, dict):
                    validated[field] = adjusted
                    continue

                accepted = verdict.get("accepted")
                if accepted is None:
                    accepted = True
                accepted = bool(accepted)

                if not accepted:
                    corrected = verdict.get("corrected_value")
                    if corrected is not None:
                        adjusted.value = corrected
                    adjusted.confidence = min(adjusted.confidence, 0.35)

                adjusted.validation_reason = verdict.get("reason")
                adjustment = float(verdict.get("confidence_adjustment", 0.0) or 0.0)
                adjusted.confidence = max(0.0, min(1.0, adjusted.confidence + adjustment))
                validated[field] = adjusted

        return validated

    def _validate_batch(
        self,
        extraction: dict[str, ExtractedField],
        chunks: list[tuple[DocumentChunk, float]],
    ) -> dict[str, dict]:
        prompt = (
            self.template
            .replace("{{EXTRACTION}}", json.dumps({k: v.model_dump() for k, v in extraction.items()}, ensure_ascii=False))
            .replace("{{CONTEXT}}", ExtractionAgent._context(chunks))
        )
        raw = self.llm.generate_json(prompt)
        if not raw:
            logger.warning("Validation LLM returned empty output; keeping extracted values unchanged.")
            return {}
        return {field: raw[field] for field in extraction if isinstance(raw.get(field), dict)}
