from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from raw LLM output."""
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


class GroqLLM:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._disabled = False
        self._warned_disabled = False
        self._min_request_interval = 0.75
        self._last_request_at = 0.0

    def generate_json(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            return self._empty_response("Groq API key is missing.")
        if self._disabled:
            return self._empty_response("Groq API is disabled due to authentication failure.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.base_url}/chat/completions"

        strategies: list[dict[str, Any]] = [
            {
                "system": "You are a JSON API. Respond with a single valid JSON object only. No markdown, no commentary.",
                "response_format": {"type": "json_object"},
            },
            {
                "system": "Return one valid JSON object only. Use double quotes for all keys and strings.",
                "response_format": None,
            },
        ]

        last_error: str | None = None
        for strategy in strategies:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": strategy["system"]},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }
            if strategy["response_format"]:
                payload["response_format"] = strategy["response_format"]

            for attempt in range(self.max_retries + 1):
                self._pace_requests()
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        response = client.post(endpoint, headers=headers, json=payload)

                        if response.status_code in (401, 403):
                            self._disabled = True
                            logger.error("Groq authentication failed (%s); disabling API calls.", response.status_code)
                            return {}

                        if response.status_code == 429:
                            retry_after = response.headers.get("retry-after")
                            sleep_for = float(retry_after) if retry_after else min(2**attempt, 8)
                            logger.warning(
                                "Groq rate limit hit; waiting %s seconds (attempt %s/%s).",
                                sleep_for,
                                attempt + 1,
                                self.max_retries + 1,
                            )
                            time.sleep(sleep_for)
                            continue

                        if response.status_code == 400:
                            error_body = response.text[:2000]
                            last_error = error_body
                            recovered = self._recover_from_error_body(error_body)
                            if recovered is not None:
                                return recovered
                            logger.warning(
                                "Groq JSON validation failed (attempt %s/%s): %s",
                                attempt + 1,
                                self.max_retries + 1,
                                error_body[:500],
                            )
                            if attempt < self.max_retries:
                                time.sleep(min(2**attempt, 4))
                                continue
                            break

                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        parsed = _extract_json_from_text(content)
                        if parsed is not None:
                            return parsed
                        last_error = f"Unparseable JSON content: {content[:300]}"
                        logger.warning("Groq returned unparseable JSON (attempt %s/%s).", attempt + 1, self.max_retries + 1)

                except httpx.HTTPStatusError as exc:
                    last_error = str(exc)
                    logger.warning("Groq HTTP error (attempt %s/%s): %s", attempt + 1, self.max_retries + 1, exc)
                except httpx.RequestError as exc:
                    last_error = str(exc)
                    logger.warning("Groq network error (attempt %s/%s): %s", attempt + 1, self.max_retries + 1, exc)

                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))

        if last_error:
            logger.warning("Groq request exhausted retries for this call; returning empty result. Last error: %s", last_error[:500])
        return {}

    def _pace_requests(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _recover_from_error_body(self, error_body: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(error_body)
        except ValueError:
            return None

        error = payload.get("error", {})
        failed_generation = error.get("failed_generation")
        if isinstance(failed_generation, str) and failed_generation.strip():
            parsed = _extract_json_from_text(failed_generation)
            if parsed is not None:
                logger.info("Recovered JSON from Groq failed_generation payload.")
                return parsed
        return None

    def _empty_response(self, reason: str) -> dict[str, Any]:
        if not self._warned_disabled:
            logger.warning("%s Returning empty extraction output for this call.", reason)
            self._warned_disabled = True
        return {}
