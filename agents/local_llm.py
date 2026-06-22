from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from raw model output."""
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass

    return None


class LocalGGUFLLM:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        chat_format: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, os.cpu_count() or 1)
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_format = chat_format
        self._warned_missing = False

        if not self.model_path.exists():
            raise RuntimeError(
                f"GGUF model file does not exist: {self.model_path}. "
                "Set LLM_MODEL_PATH to a valid Qwen2.5 GGUF file."
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Missing dependency: llama-cpp-python. Install it with pip before running local GGUF inference."
            ) from exc

        logger.info("Loading local GGUF model from %s", self.model_path)
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            chat_format=self.chat_format,
            verbose=False,
        )

    def generate_json(self, prompt: str) -> dict[str, Any]:
        strategies: list[dict[str, str]] = [
            {
                "system": "You are a JSON API. Respond with a single valid JSON object only. No markdown, no commentary.",
            },
            {
                "system": "Return one valid JSON object only. Use double quotes for all keys and strings.",
            },
        ]

        last_error: str | None = None
        for strategy in strategies:
            messages = [
                {"role": "system", "content": strategy["system"]},
                {"role": "user", "content": prompt},
            ]
            try:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=1.0,
                )
                content = response["choices"][0]["message"]["content"]
                parsed = _extract_json_from_text(content)
                if parsed is not None:
                    return parsed
                last_error = f"Unparseable JSON content: {content[:300]}"
                logger.warning("Local GGUF model returned unparseable JSON.")
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                last_error = str(exc)
                logger.warning("Local GGUF inference error: %s", exc)

        if last_error and not self._warned_missing:
            logger.warning("Local GGUF request failed; returning empty result. Last error: %s", last_error[:500])
            self._warned_missing = True
        return {}
