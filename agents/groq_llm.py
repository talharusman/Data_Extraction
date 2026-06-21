from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GroqLLM:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._offline = False
        self._warned_offline = False

    def generate_json(self, prompt: str) -> dict[str, Any]:
        if not self.api_key or self._offline:
            if not self._warned_offline:
                logger.warning("Groq API is unavailable; returning empty extraction output.")
                self._warned_offline = True
            self._offline = True
            return {}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        endpoint = f"{self.base_url}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(endpoint, headers=headers, json=payload)
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        sleep_for = float(retry_after) if retry_after else min(2**attempt, 8)
                        logger.warning(
                            "Groq rate limit hit; waiting %s seconds before retrying (attempt %s/%s).",
                            sleep_for,
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        time.sleep(sleep_for)
                        continue
                    if response.status_code >= 400:
                        logger.warning(
                            "Groq request failed with %s: %s",
                            response.status_code,
                            response.text[:1000],
                        )
                    response.raise_for_status()
                    data = response.json()
                    break
            except Exception as exc:
                last_exc = exc
                logger.warning("Groq request failed; returning empty extraction output: %s", exc)
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                self._offline = True
                return {}
        else:
            self._offline = True
            if last_exc is not None:
                logger.warning("Groq request failed; returning empty extraction output: %s", last_exc)
            return {}

        content = data["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except ValueError as exc:
            logger.warning("Groq returned invalid JSON; returning empty extraction output: %s", exc)
            return {}
