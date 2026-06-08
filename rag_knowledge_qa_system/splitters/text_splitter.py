"""Configurable text splitting strategies for enterprise knowledge chunks."""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import config_data as config
from utils.hash_utils import chunk_hash


@dataclass
class SplitterConfig:
    splitter_type: str = config.splitter_type
    chunk_size: int = config.chunk_size
    chunk_overlap: int = config.chunk_overlap
    separator: str = config.separator


def _recursive_splitter(splitter_config: SplitterConfig):
    return RecursiveCharacterTextSplitter(
        chunk_size=splitter_config.chunk_size,
        chunk_overlap=splitter_config.chunk_overlap,
        separators=config.separators,
        length_function=len,
    )


def _build_splitter(splitter_config: SplitterConfig):
    if splitter_config.splitter_type == "fixed":
        return RecursiveCharacterTextSplitter(
            separators=[""],
            chunk_size=splitter_config.chunk_size,
            chunk_overlap=splitter_config.chunk_overlap,
        )
    if splitter_config.splitter_type == "separator":
        return CharacterTextSplitter(
            separator=splitter_config.separator or "\n\n",
            chunk_size=splitter_config.chunk_size,
            chunk_overlap=splitter_config.chunk_overlap,
        )
    return _recursive_splitter(splitter_config)


def _split_markdown_headers(documents: list[Document], splitter_config: SplitterConfig) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ]
    )
    recursive = _recursive_splitter(splitter_config)
    chunks = []
    for document in documents:
        header_docs = header_splitter.split_text(document.page_content)
        for header_doc in header_docs:
            header_doc.metadata = {**document.metadata, **header_doc.metadata}
        chunks.extend(recursive.split_documents(header_docs))
    return chunks


def _split_semantic_optional(
    documents: list[Document],
    splitter_config: SplitterConfig,
    embedding=None,
) -> list[Document]:
    """Use SemanticChunker when available; otherwise fall back to recursive splitting."""
    if embedding is None:
        return _recursive_splitter(splitter_config).split_documents(documents)
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        return _recursive_splitter(splitter_config).split_documents(documents)

    try:
        semantic_splitter = SemanticChunker(embedding)
        return semantic_splitter.split_documents(documents)
    except Exception:
        return _recursive_splitter(splitter_config).split_documents(documents)


def split_documents(
    documents: list[Document],
    splitter_config: SplitterConfig | None = None,
    embedding=None,
) -> list[Document]:
    splitter_config = splitter_config or SplitterConfig()
    splitter_type = splitter_config.splitter_type

    if splitter_type == "markdown_header":
        chunks = _split_markdown_headers(documents, splitter_config)
    elif splitter_type == "semantic_optional":
        chunks = _split_semantic_optional(documents, splitter_config, embedding=embedding)
    else:
        chunks = _build_splitter(splitter_config).split_documents(documents)

    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = metadata.get("chunk_id") or f"{index:05d}-{chunk_hash(chunk.page_content)}"
        metadata["splitter_type"] = splitter_type
        metadata.setdefault("file_type", "txt")
        metadata.setdefault("source", "unknown")
        chunk.metadata = metadata
    return chunks
