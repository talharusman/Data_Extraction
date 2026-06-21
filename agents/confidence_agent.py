from __future__ import annotations

from app.models import ExtractedField


class ConfidenceScoringAgent:
    def score(self, fields: dict[str, ExtractedField], retrieval_scores: dict[str, float]) -> dict[str, ExtractedField]:
        scored: dict[str, ExtractedField] = {}
        for name, field in fields.items():
            updated = field.model_copy(deep=True)
            if updated.value is None or updated.value == "": 
                updated.confidence = 0.0
            else:
                retrieval_score = retrieval_scores.get(updated.source_chunk or "", 0.0)
                evidence_bonus = 0.1 if updated.evidence else 0.0
                validator_penalty = 0.15 if updated.validation_reason and "reject" in updated.validation_reason.lower() else 0.0
                updated.confidence = max(
                    0.0,
                    min(1.0, (updated.confidence * 0.65) + (retrieval_score * 0.25) + evidence_bonus - validator_penalty),
                )
            scored[name] = updated
        return scored
