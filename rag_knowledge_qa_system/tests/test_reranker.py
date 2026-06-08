import unittest

from rerankers.reranker import LightweightReranker


class Document:
    def __init__(self, text, chunk_id):
        self.page_content = text
        self.metadata = {"source": "faq.txt", "chunk_id": chunk_id}


class RerankerTest(unittest.TestCase):
    def test_rerank_prefers_relevant_vector_and_keyword_hit(self):
        candidates = [
            {
                "doc": Document("完全无关的内容", "c0"),
                "vector_score": 0.1,
                "bm25_hit": False,
                "bm25_rank": None,
            },
            {
                "doc": Document("图书馆 开放 时间 借书 服务", "c1"),
                "vector_score": 0.8,
                "bm25_hit": True,
                "bm25_rank": 1,
                "bm25_score": 1.0,
                "keyword_coverage": 0.8,
            },
        ]

        ranked = LightweightReranker().rerank("图书馆开放时间", candidates)
        self.assertEqual(ranked[0]["doc"].metadata["chunk_id"], "c1")
        self.assertEqual(len(ranked), 1)


if __name__ == "__main__":
    unittest.main()
