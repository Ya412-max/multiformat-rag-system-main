import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.loaders.document_loader import DocumentLoader
from core.splitter import BGEChunkSplitter
from core.embedder import BGEEmbedder
from core.vector_store import LanceDBStore
from core.rag_chain import RAGChain
from core.llm_provider import LLMProvider
from config.settings import settings

embedder = None
vector_store = None
rag_chain = None
document_loader = None
chunk_splitter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, vector_store, rag_chain, document_loader, chunk_splitter

    print("[Init] Loading BGE-M3 embedding model...")
    embedder = BGEEmbedder()
    print(f"[Init] Embedding model loaded: {embedder.model_card}")

    print("[Init] Connecting to LanceDB...")
    vector_store = LanceDBStore(embedder=embedder)
    vector_store.create_table()
    print(f"[Init] LanceDB ready, {vector_store.count_documents()} documents in table")

    document_loader = DocumentLoader(extract_tables=True)
    chunk_splitter = BGEChunkSplitter()

    rag_chain = RAGChain(
        embedder=embedder,
        vector_store=vector_store,
        enable_parent_doc=True,
        enable_hybrid_search=True,
    )
    print("[Init] RAG chain ready")

    yield

    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="RAG System API",
    description="Multi‑format document RAG system with BGE‑M3 + LanceDB",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境建议指定具体域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    stream: bool = False
    enable_hybrid: Optional[bool] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


class SourceInfo(BaseModel):
    source: str
    file_type: str
    file_path: str


@app.get("/health")
def health_check():
    return {"status": "ok", "embedding_model": embedder.model_card}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    supported = document_loader.supported_formats
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {supported}",
        )

    suffix = f"_{file.filename}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        global chunk_splitter
        if chunk_size or chunk_overlap:
            chunk_splitter = BGEChunkSplitter(
                chunk_size=chunk_size or settings.chunk_size,
                chunk_overlap=chunk_overlap or settings.chunk_overlap,
            )

        documents = document_loader.load(tmp_path)
        if not documents:
            raise HTTPException(status_code=400, detail="No content extracted from file")

        chunks = chunk_splitter.split_documents(documents)
        count = vector_store.add_documents(chunks)

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "file_type": ext,
            "original_chunks": len(documents),
            "total_chunks": count,
            "message": f"Uploaded {file.filename}: {len(documents)} pages → {count} chunks",
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.enable_hybrid is not None:
        rag_chain.enable_hybrid_search = request.enable_hybrid

    result = rag_chain.query_with_sources(request.question, top_k=request.top_k)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.get("/documents", response_model=List[SourceInfo])
def list_documents():
    sources = vector_store.list_sources()
    return [SourceInfo(**s) for s in sources]


@app.delete("/documents")
def delete_document(file_path: str = Query(..., description="File path to delete")):
    count = vector_store.delete_by_source(file_path)
    return {"status": "success", "deleted_chunks": count, "file_path": file_path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
