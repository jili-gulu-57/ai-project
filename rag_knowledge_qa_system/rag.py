import os
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config_data as config
from file_history_store import get_history
from rerankers.reranker import LightweightReranker, RerankConfig
from retrievers.hybrid_retriever import HybridRetriever, RetrievalConfig
from vector_stores import VectorStoreService


FALLBACK_ANSWER = "当前知识库中没有找到足够相关的信息，建议补充相关文档或转人工处理。"


def _source_label(metadata: dict) -> str:
    parts = [str(metadata.get("source", "未知来源"))]
    if metadata.get("page"):
        parts.append(f"第 {metadata['page']} 页")
    if metadata.get("sheet_name"):
        parts.append(f"表：{metadata['sheet_name']}")
    if metadata.get("chunk_id"):
        parts.append(f"chunk_id：{metadata['chunk_id']}")
    return "，".join(parts)


def _summarize_document(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


class RagAnswerChain:
    """Small compatibility wrapper for the previous `rag_service.chain` usage."""

    def __init__(self, service: "RagService"):
        self.service = service

    def stream(self, value: dict, runtime_config: dict | None = None):
        session_id = "user_001"
        if runtime_config:
            session_id = runtime_config.get("configurable", {}).get("session_id", session_id)
        yield from self.service.stream_answer(value["input"], session_id=session_id)

    def invoke(self, value: dict, runtime_config: dict | None = None):
        return "".join(self.stream(value, runtime_config))


class RagService:
    """Enterprise knowledge-base QA service with rewrite, hybrid retrieval and rerank."""

    def __init__(self):
        embedding = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=os.getenv(config.openai_api_key_env),
            base_url=config.openai_base_url,
            dimensions=config.embedding_dimensions,
            chunk_size=10,
            check_embedding_ctx_length=False,
        )
        self.vector_store_service = VectorStoreService(embedding=embedding)
        self.hybrid_retriever = HybridRetriever(self.vector_store_service.vector_store)
        self.reranker = LightweightReranker()
        self.chat_model = ChatOpenAI(
            model=config.chat_model,
            api_key=os.getenv(config.openai_api_key_env),
            base_url=config.openai_base_url,
            temperature=0.1,
        )
        self.query_rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是企业知识库检索助手。请结合历史对话，把用户最新问题改写成一个完整、明确、适合检索的中文问题。"
                    "只输出改写后的问题，不要解释。",
                ),
                MessagesPlaceholder("history"),
                ("user", "{input}"),
            ]
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是基于 RAG 的企业知识库智能客服。只能基于检索到的知识库内容回答，不能编造。"
                    "如果知识库没有依据，必须说明不知道，并建议补充文档或转人工。",
                ),
                (
                    "system",
                    "回答要求：\n"
                    "1. 先给出直接结论；\n"
                    "2. 必要时分点说明；\n"
                    "3. 不要使用知识库之外的事实扩展答案；\n"
                    "4. 不要自行输出引用来源，系统会在答案后统一追加。",
                ),
                ("system", "知识库内容：\n{context}"),
                MessagesPlaceholder("history"),
                ("user", "{input}"),
            ]
        )
        self.chain = RagAnswerChain(self)
        self.last_trace: dict[str, Any] = {}

    def rewrite_query(self, question: str, session_id: str = "user_001") -> str:
        history = get_history(session_id).messages
        if not history:
            return question
        rewrite_chain = self.query_rewrite_prompt | self.chat_model | StrOutputParser()
        return rewrite_chain.invoke({"input": question, "history": history[-8:]})

    def retrieve_documents(self, query: str):
        return [item["doc"] for item in self.retrieve_pipeline(query)["final_results"]]

    def retrieve_documents_with_scores(self, query: str):
        results = self.vector_store_service.search_with_scores(query)
        return results

    def retrieve_pipeline(
        self,
        rewritten_query: str,
        retrieval_config: RetrievalConfig | None = None,
        rerank_config: RerankConfig | None = None,
    ) -> dict[str, Any]:
        retrieval_config = retrieval_config or RetrievalConfig()
        rerank_config = rerank_config or RerankConfig(final_top_k=retrieval_config.final_top_k)
        trace = self.hybrid_retriever.retrieve(rewritten_query, retrieval_config)
        final_results = self.reranker.rerank(
            rewritten_query,
            trace["filtered_results"],
            rerank_config=rerank_config,
        )
        trace["rerank_results"] = final_results
        trace["final_results"] = final_results
        return trace

    def build_trace(
        self,
        question: str,
        session_id: str = "user_001",
        retrieval_config: RetrievalConfig | None = None,
        rerank_config: RerankConfig | None = None,
    ) -> dict[str, Any]:
        rewritten_query = self.rewrite_query(question, session_id=session_id)
        trace = self.retrieve_pipeline(rewritten_query, retrieval_config, rerank_config)
        trace["original_question"] = question
        trace["rewritten_query"] = rewritten_query
        self.last_trace = trace
        return trace

    def _format_documents_for_prompt(self, items: Iterable[dict[str, Any]]) -> str:
        formatted = []
        for index, item in enumerate(items, start=1):
            doc = item["doc"]
            metadata = doc.metadata
            formatted.append(
                f"[资料{index}] {_source_label(metadata)}\n"
                f"向量相似度：{item.get('vector_score')}\n"
                f"rerank_score：{item.get('rerank_score')}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(formatted)

    def _format_citations(self, items: Iterable[dict[str, Any]]) -> str:
        labels = []
        seen = set()
        for item in items:
            label = _source_label(item["doc"].metadata)
            if label not in seen:
                labels.append(f"- {label}")
                seen.add(label)
        return "引用来源：\n" + "\n".join(labels)

    def _should_fallback(self, final_results: list[dict[str, Any]], rerank_config: RerankConfig) -> bool:
        if not final_results:
            return True
        top = final_results[0]
        if top.get("rerank_score", 0.0) < rerank_config.min_score:
            return True
        return False

    def stream_answer(
        self,
        question: str,
        session_id: str = "user_001",
        retrieval_config: RetrievalConfig | None = None,
        rerank_config: RerankConfig | None = None,
    ):
        retrieval_config = retrieval_config or RetrievalConfig()
        rerank_config = rerank_config or RerankConfig(final_top_k=retrieval_config.final_top_k)
        trace = self.build_trace(question, session_id, retrieval_config, rerank_config)
        final_results = trace["final_results"]
        history = get_history(session_id).messages[-8:]

        if self._should_fallback(final_results, rerank_config):
            answer = FALLBACK_ANSWER
            yield answer
            self.save_chat_turn(session_id, question, answer)
            return

        context = self._format_documents_for_prompt(final_results)
        messages = self.prompt_template.format_messages(
            input=question,
            history=history,
            context=context,
        )

        chunks = []
        for chunk in self.chat_model.stream(messages):
            if chunk.content:
                chunks.append(chunk.content)
                yield chunk.content

        citations = "\n\n" + self._format_citations(final_results)
        chunks.append(citations)
        yield citations
        self.save_chat_turn(session_id, question, "".join(chunks))

    def save_chat_turn(self, session_id: str, question: str, answer: str):
        history = get_history(session_id)
        history.add_messages([HumanMessage(content=question), AIMessage(content=answer)])

    def trace_for_display(self, trace: dict[str, Any] | None = None) -> dict[str, Any]:
        trace = trace or self.last_trace
        display = dict(trace)
        for key in ("vector_results", "bm25_results", "merged_results", "filtered_results", "rerank_results", "final_results"):
            display[key] = [
                {
                    **{name: value for name, value in item.items() if name != "doc"},
                    "source": item["doc"].metadata.get("source", "未知来源"),
                    "page": item["doc"].metadata.get("page"),
                    "sheet_name": item["doc"].metadata.get("sheet_name"),
                    "chunk_id": item["doc"].metadata.get("chunk_id"),
                    "summary": _summarize_document(item["doc"].page_content),
                }
                for item in trace.get(key, [])
            ]
        return display


if __name__ == "__main__":
    session_config = {"configurable": {"session_id": "user_001"}}
    result = RagService().chain.invoke({"input": "什么时候可以去图书馆借书"}, session_config)
    print(result)
