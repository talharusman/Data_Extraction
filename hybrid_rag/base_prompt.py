"""
base_prompt.py
===============

REVISED per "Existing Extraction Prompt Migration": this file no longer
contains a paraphrased copy of your extraction rules -- those now come
verbatim from EXTRACTION_SYSTEM_PROMPT.txt via prompt_source.py and are
spliced directly into each group's prompt by prompt_builder.py.

What remains here is ONLY the RAG-specific output envelope: the
instruction to wrap each field's value in {value, confidence, evidence,
chunk_id, page}. This is a new OUTPUT-FORMAT requirement needed so
merge_strategy.py can do confidence/evidence-based merging across
retrieved chunks -- it is not a business/extraction rule, and it does not
override, restate, or conflict with anything in your original prompt.
"""

RAG_OUTPUT_CONTRACT = """OUTPUT FORMAT FOR THIS HYBRID-RAG PIPELINE (applies on top of all rules above):
You are only given a RETRIEVED SUBSET of the document (the top-matching chunks
for this field group), not the whole document. Use ONLY that retrieved text as
your source of truth for these fields.

For each field above, instead of returning the bare value, return an object:
    {
      "value": <the extracted value, following every rule above exactly,
                or null if the retrieved text does not contain it>,
      "confidence": <float 0.0-1.0, your certainty that "value" is correct
                      and explicitly present in the retrieved text>,
      "evidence": <the exact short sentence/phrase FROM THE RETRIEVED TEXT
                    that supports "value", or null if value is null. This
                    MUST be a real substring of the retrieved text below --
                    never invent evidence text.>,
      "chunk_id": <the chunk_id (integer) of the retrieved chunk containing
                    the evidence, or null>,
      "page": <the page_number of that chunk, or null if unknown>
    }
Confidence guidance: 0.9-1.0 stated verbatim/unambiguously; 0.6-0.85 stated
but needs light normalization (unit conversion, picking entry age out of a
range); 0.3-0.55 implied by context but not explicitly labeled; value=null
implies confidence 0.0.

If a field's rule above says to default to "N/A" when absent, still return
value=null here (not the string "N/A") -- the calling code performs that
substitution after merging candidates from multiple chunks/groups."""
