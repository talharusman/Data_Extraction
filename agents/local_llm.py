from __future__ import annotations

import json
import logging
import re
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


class TransformersLLM:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        use_4bit: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.use_4bit = use_4bit
        self._warned_missing = False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Missing dependency: transformers. Install the Colab requirements before running inference."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Missing dependency: bitsandbytes. Install it to load Qwen3-14B in 4-bit on Colab."
                ) from exc
        else:
            load_kwargs["torch_dtype"] = torch.float16

        logger.info("Loading Transformers model from %s", self.model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **load_kwargs)
        self._model.eval()
        self._input_device = next(self._model.parameters()).device
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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
                inputs = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self._input_device)
                with self._torch.no_grad():
                    output_ids = self._model.generate(
                        inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=self.temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                    )
                content = self._tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
                parsed = _extract_json_from_text(content)
                if parsed is not None:
                    return parsed
                last_error = f"Unparseable JSON content: {content[:300]}"
                logger.warning("Transformers model returned unparseable JSON.")
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                last_error = str(exc)
                logger.warning("Transformers inference error: %s", exc)

        if last_error and not self._warned_missing:
            logger.warning("Transformers request failed; returning empty result. Last error: %s", last_error[:500])
            self._warned_missing = True
        return {}


LocalGGUFLLM = TransformersLLM
