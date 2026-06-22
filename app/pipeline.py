from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.confidence_agent import ConfidenceScoringAgent
from agents.extraction_agent import ExtractionAgent
from agents.groq_llm import GroqLLM
from agents.validation_agent import ValidationAgent
from app.chunking import ChunkingEngine
from app.config import AppConfig
from app.document_reader import DocumentReader
from app.excel_exporter import ExcelExportAgent
from app.models import ProcessingStatus, ProductExtraction, ExtractedField
from embeddings.generator import EmbeddingGenerator
from parsers.factory import ParserFactory
from retrievers.semantic_retriever import SemanticRetriever
from vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class BankingExtractionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.status = ProcessingStatus()
        self.reader = DocumentReader(config.get("documents", "supported_file_types"))
        self.parser_factory = ParserFactory()
        self.chunker = ChunkingEngine(config.get("chunking", "chunk_size"), config.get("chunking", "overlap"))
        self.embedder = EmbeddingGenerator(
            config.get("embeddings", "model_name"),
            config.get("embeddings", "batch_size"),
            config.get("embeddings", "backend", default="hashing"),
        )
        self.store = QdrantVectorStore(
            url=config.runtime.qdrant_url,
            api_key=config.runtime.qdrant_api_key,
            storage_path=config.runtime.qdrant_path,
            collection_name=config.get("qdrant", "collection_name"),
            vector_size=self.embedder.dimension,
            recreate_collection=config.get("qdrant", "recreate_collection", default=False),
        )
        self.retriever = SemanticRetriever(self.embedder, self.store, config.get("retrieval", "top_k"))
        llm = GroqLLM(
            api_key=config.runtime.groq_api_key,
            base_url=config.runtime.groq_base_url,
            model=config.runtime.groq_model,
        )
        self.extractor = ExtractionAgent(llm)
        self.validator = ValidationAgent(llm)
        self.confidence = ConfidenceScoringAgent()
        self.exporter = ExcelExportAgent(
            columns=config.columns,
            products_path=config.get("outputs", "products_xlsx"),
            review_path=config.get("outputs", "review_xlsx"),
            review_threshold=float(config.get("confidence", "threshold")),
        )

    def process_folder(self, root_folder: str | Path) -> list[ProductExtraction]:
        files = self.reader.scan(root_folder)
        self.status = ProcessingStatus(state="running", files_discovered=len(files))
        self.store.clear_collection()
        products = []
        for path in files:
            try:
                products.append(self.process_file(path, export=False))
                self.status.files_processed += 1
            except Exception as exc:
                logger.exception("Failed to process %s", path)
                self.status.failures.append(f"{path}: {exc}")
        self.status.products_extracted = len(products)
        self.exporter.export(products)
        self._write_trace(products)
        self.status.state = "completed"
        return products

    def process_file(self, path: str | Path, export: bool = True) -> ProductExtraction:
        path = Path(path).resolve()
        logger.info("Processing file: %s", path)
        parser = self.parser_factory.get_parser(path)
        parsed = parser.parse(path)
        chunks = self.chunker.chunk(parsed)
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        source_path = str(path)
        try:
            self.store.delete_by_source_path(source_path)
        except Exception:
            logger.debug("Could not delete prior chunks for %s; continuing with upsert.", source_path)
        self.store.upsert_chunks(chunks, vectors)

        fields = {}
        retrieval_scores: dict[str, float] = {}
        for group_name, group_fields in self.config.extraction_groups.items():
            query = f"Banking product fields for {group_name}: {', '.join(group_fields)}"
            retrieved = self.retriever.retrieve(query, source_path=source_path)
            if not retrieved:
                logger.warning("No chunks retrieved for %s group '%s'; skipping LLM calls.", path.name, group_name)
                for field_name in group_fields:
                    fields.setdefault(field_name, ExtractedField())
                continue
            retrieval_scores.update({chunk.chunk_id: score for chunk, score in retrieved})
            extracted = self.extractor.extract(group_fields, retrieved)
            validated = self.validator.validate(extracted, retrieved)
            scored = self.confidence.score(validated, retrieval_scores)
            fields.update(scored)

        threshold = float(self.config.get("confidence", "threshold"))
        product = ProductExtraction(
            product_id=path.stem,
            source_file=path.name,
            source_path=str(path),
            fields=fields,
            needs_review=any(field.value is not None and field.confidence < threshold for field in fields.values()),
        )
        product = self.exporter.normalize_product(product)
        if export:
            self.exporter.export([product])
            self._write_trace([product])
        return product

    def _write_trace(self, products: list[ProductExtraction]) -> None:
        path = Path(self.config.get("outputs", "trace_jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for product in products:
                handle.write(json.dumps(product.model_dump(), default=str, ensure_ascii=False) + "\n")
