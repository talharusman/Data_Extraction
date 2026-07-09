"""
hybrid_rag
==========

Additive Hybrid-RAG extraction layer for the Banking Product extraction
pipeline. This package does NOT replace 02_extract_fields.py — it plugs
into it. See 03_extract_fields_hybrid.py at the project root for the
orchestrator that wires this package into the existing model loader,
normalization, and validation code.

Modules:
    field_groups        -- ONE-TIME, dev-authored field grouping (56 cols
                            grouped into 9 semantic groups) + auto-generated
                            retrieval queries. No LLM involved.
    base_prompt          -- Reusable, group-agnostic system prompt.
    prompt_builder       -- Builds a compact per-group prompt from
                            base_prompt + field_groups (Python string
                            templating only — never an LLM call).
    token_chunking       -- Token-based chunking (700-1000 tokens,
                            100-200 overlap) with chunk_id/page/token_count.
    embeddings_index     -- Embedding generation/caching + FAISS + BM25
                            index construction, scoped to ONE document
                            (DocumentIndex.clear() wipes it after use).
    retrieval            -- Hybrid FAISS+BM25 retrieval per field group.
    group_extraction     -- Runs the LLM once per field group against the
                            retrieved chunks; returns value/confidence/
                            evidence/page/chunk_id per field.
    merge_strategy       -- Confidence/evidence/table/majority/recency
                            based merge (replaces "first non-null wins").
"""
