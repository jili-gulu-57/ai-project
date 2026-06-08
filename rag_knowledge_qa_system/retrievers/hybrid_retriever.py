"""Hybrid retrieval over Chroma vector search and BM25 keyword recall."""

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

import config_data as config
from utils.hash_utils import chunk_hash


@dataclass
class RetrievalConfig:
    vector_top_k: int = config.vector_top_k
    bm25_top_k: int = config.bm25_top_k
    final_top_k: int = config.final_top_k
    similarity_threshold: float | None = config.similarity_threshold
    bm25_min_score: float = config.bm25_min_score
    bm25_min_coverage: float = config.bm25_min_coverage


def document_key(document: Document) -> str:
    source = document.metadata.get("source")
    chunk_id = document.metadata.get("chunk_id")
    if source and chunk_id:
        return f"{source}:{chunk_id}"
    return chunk_hash(document.page_content)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text with Chinese bi/tri-grams."""
    lowered = text.lower()
    latin_terms = re.findall(r"[a-z0-9_]+", lowered)
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]+", lowered)
    stop_chars = set("的是了在和与及或有什么请问一下")
    cjk_terms = []
    for sequence in cjk_sequences:
        cleaned = "".join(char for char in sequence if char not in stop_chars)
        if not cleaned:
            continue
        if len(cleaned) == 1:
            cjk_terms.append(cleaned)
            continue
        cjk_terms.extend(cleaned[index : index + 2] for index in range(len(cleaned) - 1))
        if len(cleaned) >= 3:
            cjk_terms.extend(cleaned[index : index + 3] for index in range(len(cleaned) - 2))
    return latin_terms + cjk_terms


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = set(tokenize_for_bm25(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize_for_bm25(text))
    return len(query_terms & text_terms) / len(query_terms)


class HybridRetriever:
    """Merge vector recall and BM25 recall while preserving retrieval diagnostics."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def _ensure_document_metadata(self, document: Document) -> Document:
        metadata = dict(document.metadata or {})
        if not metadata.get("source"):
            metadata["source"] = "unknown"
        if not metadata.get("file_type"):
            metadata["file_type"] = "unknown"
        if not metadata.get("chunk_id"):
            metadata["chunk_id"] = f"legacy-{chunk_hash(document.page_content)}"
        document.metadata = metadata
        return document

    def _load_all_documents(self) -> list[Document]:
        try:
            result = self.vector_store.get(include=["documents", "metadatas"])
        except Exception:
            return []

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        loaded = []
        for text, metadata in zip(documents, metadatas):
            if text:
                loaded.append(
                    self._ensure_document_metadata(
                        Document(page_content=text, metadata=metadata or {})
                    )
                )
        return loaded

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        min_coverage: float,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        documents = self._load_all_documents()
        if not documents:
            return []

        try:
            from rank_bm25 import BM25L

            query_tokens = tokenize_for_bm25(query)
            corpus = [tokenize_for_bm25(document.page_content) for document in documents]
            scores = BM25L(corpus).get_scores(query_tokens)
            max_score = max(scores, default=0.0)
            scored_results = []
            for document, raw_score in zip(documents, scores):
                normalized_score = float(raw_score / max_score) if max_score > 0 else 0.0
                coverage = keyword_overlap_score(query, document.page_content)
                if normalized_score < min_score or coverage < min_coverage:
                    continue
                scored_results.append((normalized_score, coverage, float(raw_score), document))
            scored_results.sort(key=lambda item: (item[0], item[1]), reverse=True)
        except Exception:
            scored_results = self._keyword_fallback(
                query,
                documents,
                top_k,
                min_coverage,
            )

        return [
            {
                "doc": document,
                "key": document_key(document),
                "bm25_rank": rank,
                "bm25_hit": True,
                "bm25_score": normalized_score,
                "bm25_raw_score": raw_score,
                "keyword_coverage": coverage,
            }
            for rank, (normalized_score, coverage, raw_score, document) in enumerate(
                scored_results[:top_k],
                start=1,
            )
        ]

    def _keyword_fallback(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
        min_coverage: float,
    ) -> list[tuple[float, float, float, Document]]:
        scored = []
        for document in documents:
            coverage = keyword_overlap_score(query, document.page_content)
            if coverage >= min_coverage:
                scored.append((coverage, coverage, coverage, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        except Exception:
            results = []

        normalized_results = []
        for rank, (document, score) in enumerate(results, start=1):
            document = self._ensure_document_metadata(document)
            normalized_results.append(
                {
                    "doc": document,
                    "key": document_key(document),
                    "vector_rank": rank,
                    "vector_score": float(score),
                    "vector_hit": True,
                }
            )
        return normalized_results

    def retrieve(self, query: str, retrieval_config: RetrievalConfig | None = None) -> dict[str, Any]:
        retrieval_config = retrieval_config or RetrievalConfig()
        vector_results = self._vector_search(query, retrieval_config.vector_top_k)
        bm25_results = self._bm25_search(
            query,
            retrieval_config.bm25_top_k,
            retrieval_config.bm25_min_score,
            retrieval_config.bm25_min_coverage,
        )

        merged: dict[str, dict[str, Any]] = {}
        for item in [*vector_results, *bm25_results]:
            key = item["key"]
            if key not in merged:
                merged[key] = {
                    "doc": item["doc"],
                    "key": key,
                    "vector_hit": False,
                    "bm25_hit": False,
                    "vector_rank": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "bm25_raw_score": None,
                    "keyword_coverage": 0.0,
                    "vector_score": None,
                    "filtered_by_similarity": False,
                }
            merged[key].update({name: value for name, value in item.items() if name != "doc"})

        candidates = list(merged.values())
        threshold = retrieval_config.similarity_threshold
        for item in candidates:
            score = item.get("vector_score")
            passes_vector = score is not None and (threshold is None or score >= threshold)
            passes_bm25 = (
                item.get("bm25_hit")
                and (item.get("bm25_score") or 0.0) >= retrieval_config.bm25_min_score
                and (item.get("keyword_coverage") or 0.0) >= retrieval_config.bm25_min_coverage
            )
            item["filtered_by_similarity"] = not (passes_vector or passes_bm25)

        filtered = [item for item in candidates if not item["filtered_by_similarity"]]
        return {
            "query": query,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "merged_results": candidates,
            "filtered_results": filtered,
        }
