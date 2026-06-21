from __future__ import annotations

import hashlib
import math
import re
import logging

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class EmbeddingGenerator:
    fallback_dimension = 384

    def __init__(self, model_name: str, batch_size: int = 16, backend: str = "hashing") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.backend = backend.lower().strip()
        self._model = None

    @property
    def model(self):
        if self.backend != "sentence-transformers":
            return None
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.warning(
                    "Falling back to offline hashing embeddings because %r could not be loaded: %s",
                    self.model_name,
                    exc,
                )
                self.backend = "hashing"
        return self._model

    @property
    def dimension(self) -> int:
        if self.backend != "sentence-transformers":
            return self.fallback_dimension
        model = self.model
        if model is None:
            return self.fallback_dimension
        return int(model.get_sentence_embedding_dimension())

    def _fallback_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.fallback_dimension
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "little") % self.fallback_dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + min(len(token), 12) / 12.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.backend != "sentence-transformers":
            return [self._fallback_embed(text) for text in texts]
        model = self.model
        if model is None:
            return [self._fallback_embed(text) for text in texts]
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
