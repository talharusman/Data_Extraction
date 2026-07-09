"""
embeddings_index.py
====================

Per-document embedding cache + FAISS vector index + BM25 keyword index.

Lifecycle (enforced by the orchestrator, 03_extract_fields_hybrid.py):
    idx = DocumentIndex(embedder)
    idx.build(chunks)              # embeds once, builds FAISS + BM25 once
    ... run hybrid_search() for every field group (reuses the same index) ...
    idx.clear()                    # <-- REQUIRED after finishing one product:
                                    #     drops FAISS index, BM25 index, chunk
                                    #     store, and embedding cache from memory
                                    #     before moving to the next document.

Embedder resolution order (first available wins), so this works even in
environments without internet access to download a sentence-transformers
model:
    1. sentence-transformers (semantic embeddings) -- best quality
    2. scikit-learn TF-IDF (lexical embeddings)     -- no model download
    3. pure-python hashing vectorizer                -- zero dependencies
"""
from __future__ import annotations

import gc
import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(vecs.astype("float32"))


class TfidfEmbedder:
    """Fallback embedder requiring only scikit-learn (no model download)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer_cls = TfidfVectorizer
        self.vectorizer = None
        self.dim = None

    def encode(self, texts: list[str]) -> np.ndarray:
        # Fit fresh per document (per-doc vocabulary is fine: the index is
        # scoped to one document and cleared afterward anyway).
        self.vectorizer = self._vectorizer_cls(max_features=4096)
        mat = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        self.dim = mat.shape[1]
        return _l2_normalize(mat)

    def encode_query(self, text: str) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfidfEmbedder.encode() must be called on the corpus first")
        vec = self.vectorizer.transform([text]).toarray().astype("float32")
        return _l2_normalize(vec)[0]


class HashingEmbedder:
    """Zero-dependency fallback: bag-of-words hashed into a fixed-size vector."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _hash_vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[A-Za-z0-9]+", text.lower()):
            vec[hash(tok) % self.dim] += 1.0
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.stack([self._hash_vec(t) for t in texts])
        return _l2_normalize(mat)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def make_default_embedder():
    """
    Resolve the best available embedder. Controlled via env var
    EMBEDDING_MODEL_NAME (default: sentence-transformers/all-MiniLM-L6-v2).
    """
    model_name = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2").strip()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as e:
        print(f"  [embeddings] sentence-transformers unavailable ({e}); falling back to TF-IDF.")
    try:
        return TfidfEmbedder()
    except Exception as e:
        print(f"  [embeddings] scikit-learn unavailable ({e}); falling back to hashing embedder.")
    return HashingEmbedder()


# ---------------------------------------------------------------------------
# DocumentIndex: FAISS + BM25, scoped to one document
# ---------------------------------------------------------------------------

class DocumentIndex:
    """
    Holds the embedding cache, FAISS index, and BM25 index for exactly ONE
    document's chunks. Build once per document, query many times (once per
    field group), then call clear() before moving to the next document.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder or make_default_embedder()
        self.chunks: list[dict] = []
        self.chunk_embeddings: np.ndarray | None = None  # cached, built once
        self.faiss_index = None
        self.bm25 = None
        self._bm25_corpus_tokens: list[list[str]] = []
        self._built = False

    # -- build (once per document) ---------------------------------------
    def build(self, chunks: list[dict]) -> None:
        if self._built:
            raise RuntimeError(
                "DocumentIndex.build() called twice without clear() in between -- "
                "each document must get a fresh index."
            )
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        if not texts:
            self._built = True
            return

        # Embeddings: generated ONCE here, cached in self.chunk_embeddings,
        # reused by every subsequent hybrid_search() call for this document.
        self.chunk_embeddings = self.embedder.encode(texts)

        self._build_faiss(self.chunk_embeddings)
        self._build_bm25(texts)
        self._built = True

    def _build_faiss(self, embeddings: np.ndarray) -> None:
        import faiss
        dim = embeddings.shape[1]
        # Inner product on L2-normalized vectors == cosine similarity.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.faiss_index = index

    def _build_bm25(self, texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi
        self._bm25_corpus_tokens = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self._bm25_corpus_tokens)

    # -- search ------------------------------------------------------------
    def search_faiss(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Returns [(chunk_index, cosine_score), ...] sorted best-first."""
        if self.faiss_index is None or not self.chunks:
            return []
        if hasattr(self.embedder, "encode_query"):
            qvec = self.embedder.encode_query(query)
        else:
            qvec = self.embedder.encode([query])[0]
        qvec = np.asarray([qvec], dtype="float32")
        k = min(top_k, len(self.chunks))
        scores, idxs = self.faiss_index.search(qvec, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.bm25 is None or not self.chunks:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def hybrid_search(self, query: str, top_k: int = 4, alpha: float = 0.5) -> list[dict]:
        """
        1. FAISS search  2. BM25 search  3. merge  4. dedupe
        5. rank by combined score  6. return top-K chunk dicts (with 'score').

        alpha weights FAISS (semantic) vs BM25 (keyword): combined =
        alpha*faiss_norm + (1-alpha)*bm25_norm. Both components are
        min-max normalized to [0,1] before combining so neither metric's
        raw scale dominates.
        """
        if not self.chunks:
            return []

        pool_k = max(top_k * 3, top_k)
        faiss_hits = dict(self.search_faiss(query, pool_k))
        bm25_hits = dict(self.search_bm25(query, pool_k))

        faiss_norm = _minmax_normalize(faiss_hits)
        bm25_norm = _minmax_normalize(bm25_hits)

        combined: dict[int, float] = {}
        for idx in set(faiss_norm) | set(bm25_norm):
            combined[idx] = alpha * faiss_norm.get(idx, 0.0) + (1 - alpha) * bm25_norm.get(idx, 0.0)

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            chunk = dict(self.chunks[idx])
            chunk["score"] = score
            results.append(chunk)
        return results

    # -- cleanup -------------------------------------------------------------
    def clear(self) -> None:
        """
        Drops all per-document state: FAISS index, BM25 index, chunk store,
        and cached embeddings. MUST be called after finishing extraction for
        one product file, before building the index for the next one.
        """
        self.chunks = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.bm25 = None
        self._bm25_corpus_tokens = []
        self._built = False
        gc.collect()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
