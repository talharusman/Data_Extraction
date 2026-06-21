from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from app.models import DocumentChunk

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        api_key: str | None,
        storage_path: str,
        collection_name: str,
        vector_size: int,
        recreate_collection: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.vector_size = vector_size
        self.client = self._build_client(url, api_key, storage_path)
        self._ensure_collection(vector_size, recreate_collection)

    def _build_client(self, url: str, api_key: str | None, storage_path: str) -> QdrantClient:
        try:
            client = QdrantClient(url=url, api_key=api_key)
            client.get_collections()
            return client
        except Exception:
            logger.warning(
                "Qdrant server at %s is unavailable; using in-memory embedded storage instead.",
                url,
            )
            return QdrantClient(location=":memory:")

    def _ensure_collection(self, vector_size: int, recreate_collection: bool) -> None:
        try:
            if recreate_collection:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
                return
            existing = [collection.name for collection in self.client.get_collections().collections]
            if self.collection_name not in existing:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
        except ResponseHandlingException as exc:
            raise RuntimeError("Qdrant could not be initialized.") from exc

    def clear_collection(self) -> None:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )

    def delete_by_source_path(self, source_path: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source_path", match=models.MatchValue(value=source_path))]
                )
            ),
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.model_dump())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        query_filter = None
        if filters:
            must = [models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key, value in filters.items()]
            query_filter = models.Filter(must=must)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [{"score": hit.score, "payload": hit.payload} for hit in hits]
