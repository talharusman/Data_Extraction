from __future__ import annotations

from app.models import DocumentChunk
from embeddings.generator import EmbeddingGenerator
from vectorstore.qdrant_store import QdrantVectorStore


class SemanticRetriever:
    def __init__(self, embedder: EmbeddingGenerator, store: QdrantVectorStore, top_k: int) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, source_file: str | None = None) -> list[tuple[DocumentChunk, float]]:
        vector = self.embedder.embed_query(query)
        filters = {"source_file": source_file} if source_file else None
        hits = self.store.search(vector, self.top_k, filters=filters)
        results: list[tuple[DocumentChunk, float]] = []
        for hit in hits:
            results.append((DocumentChunk(**hit["payload"]), float(hit["score"])))
        return results
