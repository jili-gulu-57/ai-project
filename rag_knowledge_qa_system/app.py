import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)

from knowledge_base import KnowledgeBaseService  # noqa: E402
from rag import RagService  # noqa: E402
from rerankers.reranker import RerankConfig  # noqa: E402
from retrievers.hybrid_retriever import RetrievalConfig  # noqa: E402
from splitters.text_splitter import SplitterConfig  # noqa: E402


SESSION_DIR = APP_DIR / "integrated_sessions"
CHAT_HISTORY_DIR = APP_DIR / "chat_history"
SERVICE_CACHE_VERSION = "2026-06-08-indexed-files-v1"


def generate_session_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def init_state():
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": "你好，我是企业知识库智能客服。你可以先上传企业文档，再开始多轮提问。",
            }
        ],
        "current_session": generate_session_id(),
        "assistant_name": "知识库智能客服",
        "assistant_style": "专业、简洁、可靠",
        "mode": "知识库问答",
        "splitter_type": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "separator": "\n\n",
        "vector_top_k": 6,
        "bm25_top_k": 6,
        "final_top_k": 4,
        "similarity_threshold": 0.35,
        "bm25_min_coverage": 0.2,
        "rerank_min_score": 0.15,
        "rerank_relative_threshold": 0.6,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_session():
    SESSION_DIR.mkdir(exist_ok=True)
    data = {
        "current_session": st.session_state.current_session,
        "assistant_name": st.session_state.assistant_name,
        "assistant_style": st.session_state.assistant_style,
        "mode": st.session_state.mode,
        "messages": st.session_state.messages,
    }
    path = SESSION_DIR / f"{st.session_state.current_session}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session_list():
    if not SESSION_DIR.exists():
        return []
    return sorted([path.stem for path in SESSION_DIR.glob("*.json")], reverse=True)


def load_session(session_id: str):
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        st.warning("会话文件不存在。")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    st.session_state.current_session = data.get("current_session", session_id)
    st.session_state.assistant_name = data.get("assistant_name", "知识库智能客服")
    st.session_state.assistant_style = data.get("assistant_style", "专业、简洁、可靠")
    st.session_state.mode = data.get("mode", "知识库问答")
    st.session_state.messages = data.get("messages", [])


def delete_session(session_id: str):
    session_file = SESSION_DIR / f"{session_id}.json"
    history_file = CHAT_HISTORY_DIR / session_id
    if session_file.exists():
        session_file.unlink()
    if history_file.exists():
        history_file.unlink()
    if st.session_state.current_session == session_id:
        st.session_state.current_session = generate_session_id()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "已新建会话。你可以继续围绕知识库进行多轮提问。",
            }
        ]


@st.cache_resource
def get_rag_service(cache_version: str):
    del cache_version
    return RagService()


@st.cache_resource
def get_kb_service(cache_version: str):
    del cache_version
    return KnowledgeBaseService()


def to_langchain_messages(messages):
    converted = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
    return converted


def build_splitter_config() -> SplitterConfig:
    chunk_overlap = min(st.session_state.chunk_overlap, max(0, st.session_state.chunk_size - 1))
    return SplitterConfig(
        splitter_type=st.session_state.splitter_type,
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=chunk_overlap,
        separator=st.session_state.separator,
    )


def build_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        vector_top_k=st.session_state.vector_top_k,
        bm25_top_k=st.session_state.bm25_top_k,
        final_top_k=st.session_state.final_top_k,
        similarity_threshold=st.session_state.similarity_threshold,
        bm25_min_coverage=st.session_state.bm25_min_coverage,
    )


def build_rerank_config() -> RerankConfig:
    return RerankConfig(
        final_top_k=st.session_state.final_top_k,
        min_score=st.session_state.rerank_min_score,
        relative_threshold=st.session_state.rerank_relative_threshold,
    )


def stream_normal_chat(rag_service: RagService, prompt: str):
    system_prompt = (
        f"你叫{st.session_state.assistant_name}，风格是{st.session_state.assistant_style}。"
        "你正在和用户进行多轮聊天。回答要自然、清晰；如果问题需要企业资料，提醒用户切换到知识库问答模式。"
    )
    history = to_langchain_messages(st.session_state.messages[:-1][-12:])
    messages = [SystemMessage(content=system_prompt), *history, HumanMessage(content=prompt)]

    for chunk in rag_service.chat_model.stream(messages):
        if chunk.content:
            yield chunk.content


def stream_rag_answer(rag_service: RagService, prompt: str):
    yield from rag_service.stream_answer(
        prompt,
        session_id=st.session_state.current_session,
        retrieval_config=build_retrieval_config(),
        rerank_config=build_rerank_config(),
    )


def _render_item(item: dict, index: int):
    title = (
        f"{index}. {item.get('source', '未知来源')} | "
        f"chunk_id: {item.get('chunk_id', '-')}"
    )
    with st.expander(title, expanded=False):
        cols = st.columns(5)
        cols[0].metric("向量分数", "-" if item.get("vector_score") is None else f"{item['vector_score']:.3f}")
        cols[1].metric("BM25 分数", "-" if item.get("bm25_score") is None else f"{item['bm25_score']:.3f}")
        cols[2].metric("关键词覆盖", f"{float(item.get('keyword_coverage') or 0.0):.2f}")
        cols[3].metric("Rerank 分", "-" if item.get("rerank_score") is None else f"{item['rerank_score']:.3f}")
        cols[4].metric("是否过滤", "是" if item.get("filtered_by_similarity") else "否")
        meta = []
        if item.get("page"):
            meta.append(f"页码：{item['page']}")
        if item.get("sheet_name"):
            meta.append(f"表名：{item['sheet_name']}")
        if item.get("original_rank"):
            meta.append(f"rerank 前排名：{item['original_rank']}")
        if item.get("rerank_rank"):
            meta.append(f"rerank 后排名：{item['rerank_rank']}")
        if meta:
            st.caption(" | ".join(meta))
        st.write(item.get("summary", ""))


def render_retrieval_details(rag_service: RagService):
    trace = rag_service.trace_for_display()
    if not trace:
        return
    with st.expander("本轮检索详情", expanded=False):
        st.markdown(f"**原始问题：** {trace.get('original_question', '-')}")
        st.markdown(f"**Query Rewrite：** {trace.get('rewritten_query', '-')}")

        tabs = st.tabs([
            "向量召回",
            "BM25 召回",
            "合并去重",
            "相似度过滤",
            "Rerank 最终结果",
        ])
        tab_keys = [
            "vector_results",
            "bm25_results",
            "merged_results",
            "filtered_results",
            "rerank_results",
        ]
        for tab, key in zip(tabs, tab_keys):
            with tab:
                items = trace.get(key, [])
                if not items:
                    st.info("暂无结果。")
                for index, item in enumerate(items, start=1):
                    _render_item(item, index)


def render_sidebar():
    with st.sidebar:
        st.header("控制台")

        if st.button("新建会话", use_container_width=True):
            save_session()
            st.session_state.current_session = generate_session_id()
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "新会话已创建。可以上传资料或直接提问。",
                }
            ]
            st.rerun()

        st.caption(f"当前会话：{st.session_state.current_session}")

        st.subheader("对话模式")
        st.session_state.mode = st.radio(
            "选择回答方式",
            ["知识库问答", "普通聊天"],
            index=0 if st.session_state.mode == "知识库问答" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.subheader("助手配置")
        st.session_state.assistant_name = st.text_input("助手名称", value=st.session_state.assistant_name)
        st.session_state.assistant_style = st.text_input("回答风格", value=st.session_state.assistant_style)

        st.subheader("知识库写入")
        uploaded_file = st.file_uploader(
            "上传企业文档",
            type=["pdf", "md", "markdown", "docx", "xlsx", "xls", "txt"],
            accept_multiple_files=False,
        )
        st.session_state.splitter_type = st.selectbox(
            "文本切分策略",
            ["recursive", "fixed", "separator", "markdown_header", "semantic_optional"],
            index=["recursive", "fixed", "separator", "markdown_header", "semantic_optional"].index(st.session_state.splitter_type),
        )
        st.session_state.chunk_size = st.number_input("chunk_size", min_value=100, max_value=4000, value=st.session_state.chunk_size, step=100)
        max_overlap = max(0, st.session_state.chunk_size - 1)
        st.session_state.chunk_overlap = st.number_input(
            "chunk_overlap",
            min_value=0,
            max_value=max_overlap,
            value=min(st.session_state.chunk_overlap, max_overlap),
            step=50,
        )
        st.session_state.separator = st.text_input("separator", value=st.session_state.separator)
        force_reindex = st.checkbox("强制重新写入")

        if uploaded_file and st.button("写入知识库", type="primary", use_container_width=True):
            with st.spinner("正在加载、切分、向量化并写入 Chroma..."):
                result = get_kb_service(SERVICE_CACHE_VERSION).upload_by_file(
                    uploaded_file,
                    force=force_reindex,
                    splitter_config=build_splitter_config(),
                )
                get_rag_service.clear()
            st.success(result)

        indexed_files = get_kb_service(SERVICE_CACHE_VERSION).list_indexed_files()
        with st.expander(f"已入库文件（{len(indexed_files)}）", expanded=False):
            if not indexed_files:
                st.caption("当前 Chroma 知识库中还没有文件。")
            for item in indexed_files:
                st.markdown(f"**{item['source']}**")
                details = [
                    f"类型：{item['file_type']}",
                    f"chunks：{item['chunk_count']}",
                ]
                if item["page_count"]:
                    details.append(f"页数：{item['page_count']}")
                if item["sheet_count"]:
                    details.append(f"工作表：{item['sheet_count']}")
                st.caption(" | ".join(details))

        st.subheader("检索参数")
        st.session_state.vector_top_k = st.slider("vector_top_k", 1, 20, st.session_state.vector_top_k)
        st.session_state.bm25_top_k = st.slider("bm25_top_k", 1, 20, st.session_state.bm25_top_k)
        st.session_state.final_top_k = st.slider("final_top_k", 1, 10, st.session_state.final_top_k)
        st.session_state.similarity_threshold = st.slider("similarity_threshold", 0.0, 1.0, st.session_state.similarity_threshold, 0.01)
        st.session_state.bm25_min_coverage = st.slider("bm25_min_coverage", 0.0, 1.0, st.session_state.bm25_min_coverage, 0.01)
        st.session_state.rerank_min_score = st.slider("rerank_min_score", 0.0, 1.0, st.session_state.rerank_min_score, 0.01)
        st.session_state.rerank_relative_threshold = st.slider(
            "rerank_relative_threshold",
            0.0,
            1.0,
            st.session_state.rerank_relative_threshold,
            0.05,
        )

        st.subheader("历史会话")
        for session_id in load_session_list():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(session_id, key=f"load_{session_id}", use_container_width=True):
                    load_session(session_id)
                    st.rerun()
            with col2:
                if st.button("删", key=f"delete_{session_id}", use_container_width=True):
                    delete_session(session_id)
                    st.rerun()


def main():
    st.set_page_config(
        page_title="企业知识库智能客服系统",
        page_icon="AI",
        layout="wide",
    )
    init_state()
    render_sidebar()

    st.title("基于 RAG 的企业知识库智能客服系统")
    st.caption("文档加载 -> 文本切分 -> Embedding -> Chroma -> 混合检索 -> 相似度过滤 -> Rerank -> Prompt -> LLM")

    try:
        rag_service = get_rag_service(SERVICE_CACHE_VERSION)
    except Exception as exc:
        st.error(f"初始化模型或向量库失败：{exc}")
        st.stop()

    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    prompt = st.chat_input("请输入问题")
    if not prompt:
        return

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chunks = []
    try:
        stream = stream_rag_answer(rag_service, prompt) if st.session_state.mode == "知识库问答" else stream_normal_chat(rag_service, prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(chunks.append(chunk) or chunk for chunk in stream)

        final_answer = response if isinstance(response, str) else "".join(chunks)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        save_session()

        if st.session_state.mode == "知识库问答":
            render_retrieval_details(rag_service)
    except Exception as exc:
        st.error(f"回答失败：{exc}")
        st.session_state.messages.append({"role": "assistant", "content": f"回答失败：{exc}"})
        save_session()


if __name__ == "__main__":
    main()
