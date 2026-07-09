"""
retrieval.py
============

Thin layer tying field_groups.FIELD_GROUPS (dev-time queries) to
embeddings_index.DocumentIndex (runtime hybrid search). No query text is
written or chosen at runtime -- each group's query was generated once in
field_groups.py from field names/descriptions.
"""
from __future__ import annotations

from .embeddings_index import DocumentIndex


def retrieve_for_group(doc_index: DocumentIndex, group: dict, top_k: int = 4, alpha: float = 0.5) -> list[dict]:
    """
    Runs hybrid (FAISS + BM25) retrieval for one field group using that
    group's pre-generated query, returning the top-K chunk dicts.
    """
    return doc_index.hybrid_search(group["query"], top_k=top_k, alpha=alpha)


def retrieve_for_all_groups(doc_index: DocumentIndex, groups: list[dict], top_k: int = 4, alpha: float = 0.5) -> dict[str, list[dict]]:
    """Convenience: {group_key: [chunk, ...]} for every group in one call."""
    return {g["key"]: retrieve_for_group(doc_index, g, top_k=top_k, alpha=alpha) for g in groups}
