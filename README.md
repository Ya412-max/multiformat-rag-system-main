# Multi-format RAG System

Multi-format RAG System 是一个基于FastAPI + LangChain + Streamlit + LanceDB实现的检索增强生成(RAG)项目；支持多格式文档上传，覆盖文档解析、向量化、parent-child chunking、LanceDB 持久化、混合检索、Rerank重排序、引用溯源、FastAPI 服务和 Streamlit 前端。

## Features

- 多格式解析：PDF、Word、TXT/Markdown、HTML、Excel/CSV、图片 OCR。
- Embedding：基于 BGE-M3 和 SentenceTransformers，支持 GPU、batch encoding 和向量归一化。
- Chunking：支持 parent-child chunking，召回小片段后回填父片段上下文。
- Vector Store：基于 LanceDB 本地持久化，支持新增、删除、来源管理和向量检索。
- Hybrid Search: 融合向量检索与全文检索，支持 Top-K 参数自定义配置，提升技术文档和长文档召回质量。
- Rerank: 检索结果重排序，优化召回片段相关性，过滤无关内容。
- LLM Provider: 预置通义千问；通过 custom 模式兼容 OpenAI、DeepSeek、智谱、Ollama等所有OpenAI协议接口，切换模型仅修改环境变量，无需改动业务代码。
- API/UI：提供 FastAPI 接口和 Streamlit 交互页面，支持文档上传、流式问答、文档删除和引用来源返回。

## Project Structure

```text
├── api/                    # FastAPI 服务
│   ├── __init__.py
│   └── main.py
├── config/                 # 环境变量和配置
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── loaders/            # 各类文档加载器
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── document_loader.py
│   │   ├── docx_loader.py
│   │   ├── excel_loader.py
│   │   ├── html_loader.py
│   │   ├── image_loader.py
│   │   ├── pdf_loader.py
│   │   └── text_loader.py
│   ├── __init__.py
│   ├── embedder.py         # BGE-M3 向量化
│   ├── llm_provider.py     # LLM大模型工厂类
│   ├── models.py
│   ├── splitter.py         # parent-child chunking
│   ├── vector_store.py     # LanceDB 封装
│   └── rag_chain.py        # RAG 主链路
├── data/                   # 向量库持久化目录
│   ├── lancedb/
│   └── .gitkeep            # 占位文件，保留目录结构
├── model/                  # 本地BGE-M3 Embedding模型，不上传git
├── ui/                     # Streamlit 前端
│   ├── __init__.py
│   └── app.py
├── .env                    # 本地环境变量
├── .env.example            # 环境变量模板
├── .gitignore              # Git忽略规则
└── requirements.txt        # 项目依赖清单

```

## Quick Start

```bash
# 创建虚拟环境
python -m venv .venv
# Windows激活虚拟环境
.venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
# 复制环境变量模板
copy .env.example .env
```

编辑 `.env`，填入模型服务和 embedding 配置：

```env
LLM_PROVIDER=qwen
LLM_API_KEY=sk-your-qwen-api-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
```

启动 API：

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

启动 UI：

```bash
streamlit run ui/app.py
```

## API

- `GET /health`: 服务状态
- `POST /upload`: 上传文档并写入知识库
- `POST /chat`: 知识库问答
- `GET /documents`: 文档来源列表
- `DELETE /docs`: 删除文档

## Notes

仓库不包含本地知识库数据、OCR 中间文件、向量库文件以及 API Key。运行后生成的数据默认位于 `data/`目录，已通过 `.gitignore` 排除。

模型文件说明：
本仓库不包含BGE-M3本地权重文件，模型体积较大，不放入Git版本管理。
首次运行前，请自行下载 BGE-M3 模型，放置到项目 `model/` 目录下。
推荐从 HuggingFace 下载：https://huggingface.co/BAAI/bge-m3

## Resume Highlights

- 构建完整 RAG 工程链路：多格式解析、BGE-M3 向量化、LanceDB 持久化、混合检索、重排序、引用溯源。
- 实现父子分块(parent-child chunking)策略，优化长文档检索，提升问答上下文完整性。
- 基于FastAPI封装后端接口，搭配Streamlit前端，实现文档上传、知识库管理、RAG问答功能。

