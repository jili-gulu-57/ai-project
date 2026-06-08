import tempfile
import unittest
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag import _source_label
from rerankers.reranker import LightweightReranker
from retrievers.hybrid_retriever import HybridRetriever, RetrievalConfig


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_relevance_scores(self, query, k):
        scores = [0.1, 0.9]
        return [
            (document, scores[index] if index < len(scores) else 0.5)
            for index, document in enumerate(self.docs[:k])
        ]

    def get(self, include=None):
        return {
            "documents": [doc.page_content for doc in self.docs],
            "metadatas": [doc.metadata for doc in self.docs],
        }


class FakeEmbeddings(Embeddings):
    def _embed(self, text):
        vector = [0.0] * 8
        for char in text:
            vector[ord(char) % len(vector)] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class RuntimePipelineTest(unittest.TestCase):
    def test_chroma_persistence_hybrid_filter_rerank_and_citation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            persist_dir = Path(temp_dir) / "chroma"
            embedding = FakeEmbeddings()
            docs = [
                Document(
                    page_content="食堂开放时间为周一到周五 7:00 到 20:00，支持早餐午餐和晚餐。",
                    metadata={"source": "食堂规则.txt", "file_type": "txt", "chunk_id": "c1"},
                ),
                Document(
                    page_content="图书馆借书需要携带校园卡，每次最多借阅五本。",
                    metadata={"source": "图书馆规则.txt", "file_type": "txt", "chunk_id": "c2", "page": 3},
                ),
            ]

            store = Chroma(
                collection_name="runtime_pipeline_test",
                embedding_function=embedding,
                persist_directory=str(persist_dir),
            )
            store.add_documents(docs)

            reopened = Chroma(
                collection_name="runtime_pipeline_test",
                embedding_function=embedding,
                persist_directory=str(persist_dir),
            )
            self.assertEqual(reopened.get()["documents"], [doc.page_content for doc in docs])

            retriever = HybridRetriever(reopened)
            trace = retriever.retrieve(
                "食堂 开放 时间",
                RetrievalConfig(vector_top_k=2, bm25_top_k=2, final_top_k=1, similarity_threshold=0.0),
            )
            self.assertTrue(trace["vector_results"])
            self.assertTrue(trace["bm25_results"])
            self.assertTrue(all(not item["filtered_by_similarity"] for item in trace["filtered_results"]))

            ranked = LightweightReranker().rerank(
                "食堂 开放 时间",
                trace["filtered_results"],
            )
            self.assertEqual(ranked[0]["rerank_rank"], 1)
            self.assertIn("source", ranked[0]["doc"].metadata)

            label = _source_label(docs[1].metadata)
            self.assertIn("图书馆规则.txt", label)
            self.assertIn("chunk_id", label)
            self.assertIn("3", label)

    def test_similarity_threshold_filters_low_vector_scores(self):
        docs = [
            Document(
                page_content="无关内容",
                metadata={"source": "a.txt", "chunk_id": "a"},
            ),
            Document(
                page_content="相关内容",
                metadata={"source": "b.txt", "chunk_id": "b"},
            ),
        ]

        trace = HybridRetriever(FakeVectorStore(docs)).retrieve(
            "相关",
            RetrievalConfig(vector_top_k=2, bm25_top_k=0, final_top_k=1, similarity_threshold=0.5),
        )
        self.assertTrue(trace["merged_results"][0]["filtered_by_similarity"])
        self.assertFalse(trace["merged_results"][1]["filtered_by_similarity"])
        self.assertEqual([item["key"] for item in trace["filtered_results"]], ["b.txt:b"])

    def test_bm25_handles_chinese_query_without_spaces(self):
        docs = [
            Document(
                page_content="食堂开放时间为每天早上七点。",
                metadata={"source": "food.txt", "chunk_id": "food"},
            ),
            Document(
                page_content="图书馆借书需要校园卡。",
                metadata={"source": "library.txt", "chunk_id": "library"},
            ),
        ]

        trace = HybridRetriever(FakeVectorStore(docs)).retrieve(
            "食堂开放时间",
            RetrievalConfig(vector_top_k=0, bm25_top_k=1, final_top_k=1, similarity_threshold=0.0),
        )
        self.assertEqual(trace["bm25_results"][0]["key"], "food.txt:food")

    def test_bm25_does_not_fill_results_with_unrelated_documents(self):
        docs = [
            Document(
                page_content="A线校车路线为东门、图书馆、实验楼，每半小时一班。",
                metadata={"source": "bus.xlsx", "chunk_id": "bus"},
            ),
            Document(
                page_content="实验室预约需要提前一天，安全测试达到八十五分。",
                metadata={"source": "lab.docx", "chunk_id": "lab"},
            ),
            Document(
                page_content="奖学金申请需要提交成绩单和申请表。",
                metadata={"source": "award.pdf", "chunk_id": "award"},
            ),
            Document(
                page_content="食堂早餐供应时间为七点到九点。",
                metadata={"source": "food.txt", "chunk_id": "food"},
            ),
        ]

        trace = HybridRetriever(FakeVectorStore(docs)).retrieve(
            "A线校车的路线是什么？",
            RetrievalConfig(
                vector_top_k=0,
                bm25_top_k=6,
                final_top_k=4,
                similarity_threshold=0.35,
                bm25_min_score=0.15,
                bm25_min_coverage=0.2,
            ),
        )
        self.assertEqual(
            [item["key"] for item in trace["bm25_results"]],
            ["bus.xlsx:bus"],
        )

    def test_missing_chunk_id_gets_stable_legacy_id(self):
        docs = [
            Document(
                page_content="校园生活服务说明。",
                metadata={"source": "legacy.txt", "chunk_id": None},
            ),
        ]
        trace = HybridRetriever(FakeVectorStore(docs)).retrieve(
            "校园生活服务",
            RetrievalConfig(vector_top_k=1, bm25_top_k=0, similarity_threshold=0.0),
        )
        chunk_id = trace["vector_results"][0]["doc"].metadata["chunk_id"]
        self.assertTrue(chunk_id.startswith("legacy-"))


if __name__ == "__main__":
    unittest.main()
