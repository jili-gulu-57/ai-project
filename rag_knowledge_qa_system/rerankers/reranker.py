"""Lightweight reranker for hybrid RAG candidates.

The implementation is model-free by default so the project remains runnable in
simple environments. It combines vector relevance, BM25 hits, query coverage and
a small length penalty. A real rerank model can replace this class later without
changing the rest of the pipeline.
"""

import re
from dataclasses import dataclass
from typing import Any

import config_data as config


@dataclass
class RerankConfig:
    final_top_k: int = config.final_top_k
    min_score: float = config.rerank_min_score
    relative_threshold: float = config.rerank_relative_threshold


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    return latin + cjk


def _keyword_coverage(query: str, text: str) -> float:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(_tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


def _length_penalty(text: str) -> float:
    length = len(text)
    if 120 <= length <= 1800:
        return 0.0
    if length < 120:
        return min(0.15, (120 - length) / 120 * 0.15)
    return min(0.2, (length - 1800) / 1800 * 0.2)


class LightweightReranker:
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        rerank_config: RerankConfig | None = None,
    ) -> list[dict[str, Any]]:
        rerank_config = rerank_config or RerankConfig()
        ranked = []
        for original_rank, item in enumerate(candidates, start=1):
            document = item["doc"]
            vector_score = item.get("vector_score")
            vector_component = max(0.0, min(1.0, float(vector_score))) if vector_score is not None else 0.0
            bm25_component = float(item.get("bm25_score") or 0.0)
            coverage = max(
                float(item.get("keyword_coverage") or 0.0),
                _keyword_coverage(query, document.page_content),
            )
            penalty = _length_penalty(document.page_content)
            rerank_score = (
                0.55 * vector_component
                + 0.25 * bm25_component
                + 0.3 * coverage
                - penalty
            )
            enriched = dict(item)
            enriched["original_rank"] = original_rank
            enriched["rerank_score"] = round(max(0.0, rerank_score), 6)
            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                item["rerank_score"],
                item.get("vector_score") or 0.0,
                -1 * (item.get("bm25_rank") or 999),
            ),
            reverse=True,
        )
        for index, item in enumerate(ranked, start=1):
            item["rerank_rank"] = index

        if not ranked or ranked[0]["rerank_score"] < rerank_config.min_score:
            return []

        cutoff = max(
            rerank_config.min_score,
            ranked[0]["rerank_score"] * rerank_config.relative_threshold,
        )
        return [
            item
            for item in ranked
            if item["rerank_score"] >= cutoff
        ][: rerank_config.final_top_k]
