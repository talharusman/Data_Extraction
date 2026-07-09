# Hybrid RAG Upgrade — Setup

## Files
Drop these alongside your existing `02_extract_fields.py` and `pipeline_config.py`:
```
hybrid_rag/
  __init__.py
  field_groups.py       one-time schema grouping (9 groups, 55 fields)
  base_prompt.py         shared reusable system prompt
  prompt_builder.py      builds per-group prompts (no LLM involved)
  token_chunking.py      700-1000 token chunks, 100-200 overlap
  embeddings_index.py    embeddings + FAISS + BM25, per-document, clearable
  retrieval.py           hybrid FAISS+BM25 search per field group
  group_extraction.py    one LLM call per group, value/confidence/evidence
  merge_strategy.py      confidence > evidence > table > majority > recency
03_extract_fields_hybrid.py   new entrypoint — run this instead of 02
```
`02_extract_fields.py` is untouched. `03_extract_fields_hybrid.py` imports it
directly (model loading, file reading, `normalize_record`, the validation
prompt, resumability) and only replaces the retrieval/extraction core.

## Install
```
pip install faiss-cpu rank_bm25 scikit-learn sentence-transformers --break-system-packages
```
`sentence-transformers` is optional — if it's missing or can't reach the
internet, the pipeline automatically falls back to TF-IDF (scikit-learn),
and finally to a zero-dependency hashing embedder. FAISS + BM25 keyword
search still run either way.

## Run
```
python 03_extract_fields_hybrid.py
```
Same interactive input prompt (single file / folder / recursive folder) and
the same `OUT_JSONL` output file/schema as `02_extract_fields.py`.

## New .env options
```
RAG_TARGET_TOKENS=850          # 700-1000
RAG_OVERLAP_TOKENS=150         # 100-200
RAG_TOP_K=4                    # chunks retrieved per field group
RAG_ALPHA=0.5                  # 1.0 = pure semantic, 0.0 = pure keyword
RAG_ENSEMBLE_PASSES=1          # set >1 for genuine majority-vote merging
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EXTRACTION_DEBUG_MODE=false    # true writes OUT_JSONL.debug.jsonl with
                                # confidence/evidence/chunk_id/page per field
```

## What changed vs. the old pipeline
| Step | Before | After |
|---|---|---|
| Chunking | character windows | token windows (HF tokenizer-accurate), page-aware for PDFs |
| Retrieval | none — every chunk sent in full | FAISS + BM25 hybrid, top-K per field group |
| Prompting | one 56-field prompt per chunk | 9 focused group prompts, shared base prompt text deduped |
| LLM calls/doc | ~1 per char-chunk × 56 fields each | ~9 (one per group) × ensemble passes, + 1 final validation |
| Merge | first non-null wins | confidence → evidence → table → majority → recency |
| Grouping | n/a | computed once in `field_groups.py`, never at runtime |
| Cleanup | n/a | `DocumentIndex.clear()` wipes FAISS/BM25/embeddings after every product |

## Verified in this environment
`field_groups`, `token_chunking`, `embeddings_index`/`retrieval` (TF-IDF
fallback path), `group_extraction` (mocked LLM), and `merge_strategy`
(majority-agreement and high-confidence-outlier-override cases) were all
unit-tested directly and behave as designed. The full `03_extract_fields_hybrid.py`
orchestrator is syntax-verified but could not be run end-to-end here since
it needs `torch`/`transformers`/your local Qwen model/`pipeline_config.py`,
none of which are present in this sandbox — run it in your Colab/GPU
environment first on one sample document before a full batch.
