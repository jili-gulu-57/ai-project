"""Central configuration for the enterprise RAG customer service system."""

import os


# File and Chroma persistence
md5_path = "./md5.txt"
upload_directory = "./uploaded_files"
collection_name = "rag"
persist_directory = "./chroma_db"

# Text splitter
chunk_size = 1000
chunk_overlap = 100
splitter_type = "recursive"
separator = "\n\n"
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ","]
min_split_char_number = 1000

# Hybrid retrieval
retriever_k = 4
vector_top_k = 6
bm25_top_k = 6
final_top_k = 4
similarity_threshold = 0.35
bm25_min_score = 0.15
bm25_min_coverage = 0.2
rerank_min_score = 0.15
rerank_relative_threshold = 0.6

# Models
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
chat_model = os.getenv("CHAT_MODEL", "qwen-max")
openai_api_key_env = "DASHSCOPE_API_KEY"
openai_base_url = os.getenv(
    "OPENAI_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
