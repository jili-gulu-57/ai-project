"""Small retrieval evaluation for the bundled school-rule knowledge base."""

import argparse
import json
from pathlib import Path


def hit_at_k(ranked_sources, expected_source):
    return int(expected_source in ranked_sources)


def reciprocal_rank(ranked_sources, expected_source):
    for rank, source in enumerate(ranked_sources, start=1):
        if source == expected_source:
            return 1.0 / rank
    return 0.0


def evaluate_cases(search_with_scores, cases):
    details = []
    for case in cases:
        results = search_with_scores(case["question"])
        sources = [document.metadata.get("source", "未知来源") for document, _ in results]
        details.append(
            {
                "question": case["question"],
                "expected_source": case["expected_source"],
                "retrieved_sources": sources,
                "hit_at_k": hit_at_k(sources, case["expected_source"]),
                "reciprocal_rank": reciprocal_rank(sources, case["expected_source"]),
            }
        )

    count = len(details) or 1
    return {
        "case_count": len(details),
        "hit_at_k": sum(item["hit_at_k"] for item in details) / count,
        "mrr": sum(item["reciprocal_rank"] for item in details) / count,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation_cases.json")
    parser.add_argument("--knowledge-dir", default="学校规则测试文档")
    args = parser.parse_args()

    from knowledge_base import KnowledgeBaseService
    from vector_stores import VectorStoreService

    base_dir = Path(__file__).resolve().parent
    cases = json.loads((base_dir / args.cases).read_text(encoding="utf-8"))
    knowledge_dir = base_dir / args.knowledge_dir

    kb_service = KnowledgeBaseService()
    for file_path in sorted(knowledge_dir.glob("*.txt")):
        kb_service.upload_by_str(file_path.read_text(encoding="utf-8"), file_path.name)

    vector_store = VectorStoreService(kb_service.embedding)
    metrics = evaluate_cases(vector_store.search_with_scores, cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
