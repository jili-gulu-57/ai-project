import datetime
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import config_data as config
from loaders.document_loader import load_document, load_uploaded_file, merge_document_text
from splitters.text_splitter import SplitterConfig, split_documents
from utils.file_utils import normalize_file_type
from utils.hash_utils import md5_bytes, md5_text


def check_md5(md5_str: str):
    """Check whether a file/content hash has already been indexed."""
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()
        return False

    for line in open(config.md5_path, "r", encoding="utf-8").readlines():
        if line.strip() == md5_str:
            return True
    return False


def save_md5(md5_str):
    """Persist a file/content hash after successful vector-store writes."""
    with open(config.md5_path, "a", encoding="utf-8") as file:
        file.write(md5_str + "\n")


def get_string_md5(input_str, encoding="utf-8"):
    return md5_text(input_str, encoding=encoding)


def _clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


class KnowledgeBaseService:
    def __init__(self):
        self.embedding = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=os.getenv(config.openai_api_key_env),
            base_url=config.openai_base_url,
            dimensions=config.embedding_dimensions,
            chunk_size=10,
            check_embedding_ctx_length=False,
        )
        os.makedirs(config.persist_directory, exist_ok=True)
        os.makedirs(config.upload_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def _has_vectors(self, md5_hex: str, filename: str) -> bool:
        """md5.txt is only a write record; Chroma must contain vectors too."""
        for where in ({"file_md5": md5_hex}, {"source": filename}):
            try:
                result = self.chroma.get(where=where, limit=1)
                if result.get("ids"):
                    return True
            except Exception:
                pass
        return False

    def _delete_old_vectors(self, md5_hex: str, filename: str):
        """Remove old chunks on forced reindex to avoid duplicate recall."""
        for where in ({"file_md5": md5_hex}, {"source": filename}):
            try:
                self.chroma.delete(where=where)
            except Exception:
                pass

    def _write_documents(
        self,
        documents: list[Document],
        filename: str,
        md5_hex: str,
        splitter_config: SplitterConfig | None,
        force: bool,
    ) -> str:
        md5_exists = check_md5(md5_hex)
        vectors_exist = self._has_vectors(md5_hex, filename)

        if md5_exists and vectors_exist and not force:
            return "数据已存在，向量库中也能检索到，无需重复添加。"

        if force:
            self._delete_old_vectors(md5_hex, filename)

        chunks = split_documents(documents, splitter_config=splitter_config, embedding=self.embedding)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for index, chunk in enumerate(chunks):
            metadata = {
                **chunk.metadata,
                "source": chunk.metadata.get("source", filename),
                "file_type": chunk.metadata.get("file_type", normalize_file_type(filename)),
                "file_md5": md5_hex,
                "create_time": now,
                "operator": "Qiii",
                "chunk_index": index,
            }
            chunk.metadata = _clean_metadata(metadata)

        if not chunks:
            return "未从文件中抽取到可写入的文本内容。"

        self.chroma.add_documents(chunks)

        if not md5_exists:
            save_md5(md5_hex)
            return f"文件上传成功，已写入知识库，共生成 {len(chunks)} 个文本块。"

        if not vectors_exist:
            return f"文件记录已存在，但向量库缺失，已自动重新写入 {len(chunks)} 个文本块。"

        return f"文件已强制重新写入知识库，共生成 {len(chunks)} 个文本块。"

    def upload_by_str(
        self,
        data: str,
        filename,
        force: bool = False,
        splitter_config: SplitterConfig | None = None,
    ):
        """Vectorize raw text and write it into Chroma. Kept for existing callers."""
        md5_hex = get_string_md5(data)
        file_type = normalize_file_type(str(filename))
        document = Document(
            page_content=data,
            metadata={
                "source": str(filename),
                "file_type": file_type,
            },
        )
        return self._write_documents([document], str(filename), md5_hex, splitter_config, force)

    def upload_by_path(
        self,
        file_path: str | Path,
        force: bool = False,
        splitter_config: SplitterConfig | None = None,
    ) -> str:
        path = Path(file_path)
        documents = load_document(path)
        md5_hex = md5_bytes(path.read_bytes())
        return self._write_documents(documents, path.name, md5_hex, splitter_config, force)

    def upload_by_file(
        self,
        uploaded_file,
        force: bool = False,
        splitter_config: SplitterConfig | None = None,
    ) -> str:
        documents = load_uploaded_file(uploaded_file, config.upload_directory)
        md5_hex = md5_bytes(uploaded_file.getvalue())
        return self._write_documents(documents, uploaded_file.name, md5_hex, splitter_config, force)

    def preview_uploaded_file(self, uploaded_file) -> str:
        documents = load_uploaded_file(uploaded_file, config.upload_directory)
        return merge_document_text(documents)

    def list_indexed_files(self) -> list[dict]:
        """List files that actually have chunks stored in the Chroma collection."""
        try:
            result = self.chroma.get(include=["metadatas"])
        except Exception:
            return []

        files: dict[str, dict] = {}
        for metadata in result.get("metadatas") or []:
            metadata = metadata or {}
            source = str(metadata.get("source") or "unknown")
            item = files.setdefault(
                source,
                {
                    "source": source,
                    "file_type": str(
                        metadata.get("file_type")
                        or normalize_file_type(source)
                        or "unknown"
                    ),
                    "chunk_count": 0,
                    "pages": set(),
                    "sheets": set(),
                    "create_time": str(metadata.get("create_time") or ""),
                },
            )
            item["chunk_count"] += 1
            if metadata.get("page") is not None:
                item["pages"].add(str(metadata["page"]))
            if metadata.get("sheet_name"):
                item["sheets"].add(str(metadata["sheet_name"]))
            if metadata.get("create_time"):
                item["create_time"] = str(metadata["create_time"])

        indexed_files = []
        for item in files.values():
            indexed_files.append(
                {
                    "source": item["source"],
                    "file_type": item["file_type"],
                    "chunk_count": item["chunk_count"],
                    "page_count": len(item["pages"]),
                    "sheet_count": len(item["sheets"]),
                    "create_time": item["create_time"],
                }
            )
        return sorted(indexed_files, key=lambda item: item["source"].lower())


if __name__ == "__main__":
    r1 = get_string_md5("你好")
    r2 = get_string_md5("你好")
    r3 = get_string_md5("你好1")
    print(r1, r2, r3)
    print(check_md5(r1), check_md5(r2), check_md5(r3))
