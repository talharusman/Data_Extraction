from __future__ import annotations

from app.models import DocumentChunk
from embeddings.generator import EmbeddingGenerator
from vectorstore.qdrant_store import QdrantVectorStore


class SemanticRetriever:
    def __init__(self, embedder: EmbeddingGenerator, store: QdrantVectorStore, top_k: int) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        source_file: str | None = None,
        source_path: str | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        vector = self.embedder.embed_query(query)
        filters: dict[str, str] | None = None
        if source_path:
            filters = {"source_path": source_path}
        elif source_file:
            filters = {"source_file": source_file}
        hits = self.store.search(vector, self.top_k, filters=filters)
        results: list[tuple[DocumentChunk, float]] = []
        for hit in hits:
            results.append((DocumentChunk(**hit["payload"]), float(hit["score"])))
        return results
