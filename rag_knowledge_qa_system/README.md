# 基于 RAG 的企业知识库智能客服系统

这是一个基于 Streamlit、LangChain、Chroma 和大模型 API 的企业知识库智能客服系统。项目支持企业文档上传、文本抽取、可配置切分、Embedding 向量化、Chroma 持久化、Query Rewrite、混合检索、相似度过滤、轻量 rerank、多轮会话、检索详情展示和客服兜底拒答。

## 系统架构

文档加载 → 文本切分 → Embedding → Chroma → 混合检索 → 相似度过滤 → rerank → Prompt → LLM

数据准备阶段：

本地企业文档加载 → 文本抽取 → 文本切分 → Embedding 向量化 → 写入 Chroma 向量数据库。

应用问答阶段：

用户提问 → Query Rewrite → 向量检索 + BM25 关键词检索 → 合并去重 → 相似度过滤 → rerank 重排序 → 结果聚合 → 注入 Prompt → LLM 生成答案。

## 功能列表

- Streamlit 前端页面，支持知识库问答和普通聊天双模式。
- 支持上传 PDF、Markdown、Word docx、Excel xlsx/xls、TXT。
- 根据文件后缀自动选择加载器，统一转成 LangChain `Document`。
- metadata 保留 `source`、`file_type`、`page`、`sheet_name`、`chunk_id`、`file_md5`。
- 支持 `fixed`、`recursive`、`separator`、`markdown_header`、`semantic_optional` 五种切分策略。
- 使用 `text-embedding-v4` 写入 Chroma 持久化向量库。
- 保留 MD5 文件去重和强制重建逻辑。
- 支持多轮会话保存，并基于历史对话做 Query Rewrite。
- 混合检索：Chroma 向量召回 + BM25 关键词召回。
- 支持 `vector_top_k`、`bm25_top_k`、`final_top_k`、`similarity_threshold`、`rerank_min_score` 配置。
- 对 Chroma relevance score 做相似度阈值过滤，低于阈值的向量片段不会进入最终上下文。
- 轻量 rerank 综合向量相似度、BM25 命中、关键词覆盖率和文本长度惩罚。
- 当召回为空或综合分过低时拒答，不允许模型编造。
- 每次回答末尾展示引用来源，包括文件名、页码/表名和 chunk_id。
- 检索详情展示原始问题、改写问题、向量召回、BM25 召回、合并去重、过滤结果和 rerank 最终结果。

## 安装依赖

```powershell
cd C:\Users\22632\Desktop\项目与简历\ai-project\rag_knowledge_qa_system
python -m pip install -r requirements.txt
```

如果本机没有 `python` 命令，请使用你的 Python 安装路径或虚拟环境中的 Python。

核心依赖包括：

- `streamlit`
- `langchain`
- `langchain-openai`
- `langchain-chroma`
- `chromadb`
- `rank-bm25`
- `pypdf`
- `python-docx`
- `pandas`
- `openpyxl`
- `xlrd`

`semantic_optional` 语义切分为可选能力。如需启用真实语义切分，可额外安装：

```powershell
python -m pip install langchain-experimental
```

未安装时系统会自动降级为 `recursive` 递归字符切分。

## 运行方式

设置大模型和 Embedding API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的 API Key"
streamlit run app.py
```

默认模型配置在 `config_data.py`：

- `embedding_model = "text-embedding-v4"`
- `chat_model = "qwen-max"`
- `openai_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"`

也可以通过环境变量覆盖：

```powershell
$env:CHAT_MODEL="qwen-max"
$env:EMBEDDING_MODEL="text-embedding-v4"
$env:OPENAI_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

## 支持的文件格式

- PDF：按页抽取，metadata 保留 `page`。
- Markdown：支持普通切分，也支持 `markdown_header` 标题层级切分。
- Word docx：抽取段落和表格文本。
- Excel xlsx/xls：按工作表抽取，metadata 保留 `sheet_name`。
- TXT：支持 UTF-8、UTF-8-SIG、GB18030 读取降级。

## 配置说明

主要配置位于 `config_data.py`，也可以在 Streamlit 侧边栏动态调整：

- `chunk_size`：文本块长度。
- `chunk_overlap`：文本块重叠长度。
- `splitter_type`：切分策略，默认 `recursive`。
- `separator`：`separator` 策略使用的分隔符。
- `vector_top_k`：向量召回数量。
- `bm25_top_k`：BM25 召回数量。
- `final_top_k`：最终进入 Prompt 的片段数量。
- `similarity_threshold`：Chroma relevance score 阈值，分数越高越相关。
- `bm25_min_coverage`：查询关键词在候选文本中的最低覆盖率，用于过滤 BM25 弱命中。
- `rerank_min_score`：客服兜底阈值，最高综合分低于该值时拒答。
- `rerank_relative_threshold`：候选片段相对第一名的最低得分比例，防止 final_top_k 强行补入低相关片段。

调整检索参数或切分策略后，已有 Chroma 数据不会自动重新切分。请在页面勾选“强制重新写入”并重新上传测试文件，确保旧 chunks 和 metadata 被更新。

## 项目结构

```text
rag_knowledge_qa_system/
  app.py
  config_data.py
  knowledge_base.py
  rag.py
  vector_stores.py
  loaders/document_loader.py
  splitters/text_splitter.py
  retrievers/hybrid_retriever.py
  rerankers/reranker.py
  utils/hash_utils.py
  utils/file_utils.py
  requirements.txt
  README.md
```

## 项目亮点

- 从“单一向量检索”升级为企业客服可用的完整 RAG 链路。
- 多格式文档统一抽象为 LangChain `Document`，保留来源、页码、表名和 chunk_id。
- 检索链路透明可观测，便于调参和面试展示。
- 混合检索兼顾语义召回和关键词精确命中。
- 轻量 rerank 无需额外模型即可运行，后续可无缝替换为真实 rerank API。
- 通过拒答机制降低幻觉风险，更贴近企业客服场景。

## 评测与测试

内置基础单元测试：

```powershell
python -m unittest discover -s tests
```

内置学校规则样例检索评测：

```powershell
python evaluation.py
```

评测会写入示例知识文档，并输出 `hit@k`、MRR 和命中来源。

## 后续可扩展方向

- 接入真实 rerank 模型，例如 bge-reranker、Cohere Rerank 或企业内部重排服务。
- 接入 Milvus、Elasticsearch、OpenSearch，实现更强的生产级混合检索。
- 增加文档版本管理、权限控制、部门知识库隔离。
- 增加问答反馈闭环，用用户反馈优化召回和 Prompt。
- 增加批量目录导入、定时同步企业网盘或知识管理系统。
- 增加 LangGraph 工作流，把检索、工具调用、人工转接做成可观测节点。

## 可写入简历的项目描述

项目名称：基于 RAG 的企业知识库智能客服系统

技术栈：Python、Streamlit、LangChain、Chroma、Embedding、BM25、Rerank、RAG、大模型 API

项目描述：构建面向企业内部知识库的智能客服系统，支持多格式企业文档上传、文本抽取、切分、向量化和 Chroma 持久化存储；在问答阶段结合 Query Rewrite、向量检索、BM25 关键词检索、相似度过滤和 rerank 重排序，将高相关知识片段注入 Prompt，由大模型生成带引用来源的可追溯答案。

核心工作：

- 设计并实现 PDF、Markdown、Word、Excel、TXT 多格式文档加载链路，统一转换为 LangChain Document 并保留来源、页码、表名和 chunk_id。
- 实现 fixed、recursive、separator、markdown_header、semantic_optional 多种文本切分策略，支持在 Streamlit 页面动态配置。
- 在 Chroma 向量检索基础上新增 BM25 关键词召回与混合检索合并去重逻辑，提升企业知识问答的召回稳定性。
- 封装轻量级 rerank 模块，综合向量相似度、BM25 命中、关键词覆盖率和文本长度惩罚，对候选片段重新排序。
- 增加相似度阈值过滤、客服拒答兜底、引用来源展示和完整检索详情面板，降低大模型幻觉并提升系统可解释性。
