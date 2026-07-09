"""
token_chunking.py
==================

Replaces the character-based chunk_text() in 02_extract_fields.py with
TOKEN-based chunking, per spec:
    - 700-1000 tokens per chunk (default target 850)
    - 100-200 token overlap (default 150)
    - metadata: chunk_id, page_number, text, token_count

Token counting prefers the model's own HF tokenizer (passed in from
02_extract_fields.py's already-loaded tokenizer) so token budgeting is
consistent with what the LLM actually sees. If no tokenizer is supplied
(e.g. for offline unit tests), falls back to tiktoken, and finally to a
~4-chars-per-token heuristic so the module still works with zero extra
dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: int
    page_number: int
    text: str
    token_count: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "text": self.text,
            "token_count": self.token_count,
        }


class _TokenCounter:
    """Wraps whichever tokenizer is available behind count()/encode()/decode()."""

    def __init__(self, hf_tokenizer=None):
        self.hf_tokenizer = hf_tokenizer
        self._tiktoken_enc = None
        if hf_tokenizer is None:
            try:
                import tiktoken
                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tiktoken_enc = None

    def encode(self, text: str) -> list[int]:
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.encode(text, add_special_tokens=False)
        if self._tiktoken_enc is not None:
            return self._tiktoken_enc.encode(text)
        # Heuristic fallback: ~4 chars/token, encode as pseudo-ids by index.
        approx_tokens = max(1, len(text) // 4)
        return list(range(approx_tokens))

    def decode(self, ids: list[int], original_text: str = "") -> str:
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.decode(ids)
        if self._tiktoken_enc is not None:
            return self._tiktoken_enc.decode(ids)
        # Heuristic fallback has no real ids -> caller should slice by chars
        # instead (see chunk_document below, which never calls decode() in
        # heuristic mode).
        raise RuntimeError("decode() unavailable in heuristic fallback mode")

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def is_heuristic(self) -> bool:
        return self.hf_tokenizer is None and self._tiktoken_enc is None


def _split_into_pages(text: str, page_texts: list[str] | None) -> list[tuple[int, str]]:
    """Returns [(page_number, page_text), ...]. Falls back to one page."""
    if page_texts:
        return [(i + 1, t) for i, t in enumerate(page_texts) if t and t.strip()]
    return [(1, text)]


def _chunk_page_by_tokens(
    page_text: str,
    counter: _TokenCounter,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int]]:
    """Token-window chunking of a single page's text. Returns [(text, token_count), ...]."""
    if counter.is_heuristic:
        # ~4 chars/token heuristic, operate directly on characters so we
        # never need a real decode().
        target_chars = target_tokens * 4
        overlap_chars = overlap_tokens * 4
        pieces = []
        start = 0
        length = len(page_text)
        if length <= target_chars:
            return [(page_text.strip(), counter.count(page_text))] if page_text.strip() else []
        while start < length:
            end = min(start + target_chars, length)
            if end < length:
                boundary = page_text.rfind(". ", max(start, end - 200), end)
                if boundary > start:
                    end = boundary + 1
            piece = page_text[start:end].strip()
            if piece:
                pieces.append((piece, counter.count(piece)))
            if end >= length:
                break
            start = max(end - overlap_chars, start + 1)
        return pieces

    ids = counter.encode(page_text)
    if len(ids) <= target_tokens:
        return [(page_text.strip(), len(ids))] if page_text.strip() else []

    pieces = []
    start = 0
    n = len(ids)
    while start < n:
        end = min(start + target_tokens, n)
        window_ids = ids[start:end]
        piece_text = counter.decode(window_ids).strip()
        if piece_text:
            pieces.append((piece_text, len(window_ids)))
        if end >= n:
            break
        start = max(end - overlap_tokens, start + 1)
    return pieces


def chunk_document(
    text: str,
    hf_tokenizer=None,
    target_tokens: int = 850,
    overlap_tokens: int = 150,
    page_texts: list[str] | None = None,
) -> list[dict]:
    """
    Token-based chunking with metadata.

    target_tokens: 700-1000 recommended (default 850).
    overlap_tokens: 100-200 recommended (default 150).
    page_texts: optional list of per-page text (e.g. from pdfplumber, one
        entry per PDF page). When provided, chunks never cross a page
        boundary, so page_number is exact. When omitted, the whole
        document is treated as page 1 (page_number is then a best-effort
        constant, matching what plain .txt/.docx sources can offer).

    Returns a list of dicts: {chunk_id, page_number, text, token_count}.
    """
    target_tokens = max(700, min(1000, target_tokens))
    overlap_tokens = max(100, min(200, overlap_tokens))

    counter = _TokenCounter(hf_tokenizer)
    pages = _split_into_pages(text, page_texts)

    chunks: list[Chunk] = []
    chunk_id = 1
    for page_number, page_text in pages:
        for piece_text, token_count in _chunk_page_by_tokens(
            page_text, counter, target_tokens, overlap_tokens
        ):
            chunks.append(Chunk(chunk_id, page_number, piece_text, token_count))
            chunk_id += 1

    return [c.to_dict() for c in chunks]
