from typing import List, Optional, Iterator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.embedder import BGEEmbedder
from core.vector_store import LanceDBStore
from core.llm_provider import create_llm
from config.settings import settings


RAG_SYSTEM_PROMPT = """你是一名知识丰富的AI助手，请基于提供的上下文内容回答问题。

要求：
1. 主要依据给出的上下文信息回答问题。
2. 如果上下文没有足够的信息，请如实告知。
3. 必要时引用上下文中对应的内容。
4. 回答简洁，结构清晰。
5. 如果问题是中文，则使用中文回答；如果问题是英文，则使用英文回答。

上下文：
{context}

问题：{question}

回答："""


class RAGChain:
    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        vector_store: Optional[LanceDBStore] = None,
        llm_provider: Optional[str] = None,
        enable_parent_doc: bool = True,
        enable_hybrid_search: bool = True,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
    ):
        self._embedder = embedder
        self.vector_store = vector_store
        self.llm_provider_name = llm_provider or settings.llm_provider
        self.enable_parent_doc = enable_parent_doc
        self.enable_hybrid_search = enable_hybrid_search
        self.top_k = top_k or settings.top_k
        self.temperature = temperature

        self.llm = None
        self._chain = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = BGEEmbedder()
        return self._embedder

    @property
    def store(self):
        if self.vector_store is None:
            self.vector_store = LanceDBStore(embedder=self.embedder)
            self.vector_store.create_table()
        return self.vector_store

    def _get_llm(self):
        if self.llm is None:
            self.llm = create_llm(
                provider=self.llm_provider_name,
                temperature=self.temperature,
            )
        return self.llm

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        top_k = top_k or self.top_k

        if self.enable_hybrid_search:
            child_docs = self.store.hybrid_search(
                query, top_k=top_k, chunk_type="child"
            )
        else:
            child_docs = self.store.search(
                query, top_k=top_k, chunk_type="child"
            )

        if self.enable_parent_doc:
            return self._enrich_with_parents(child_docs)
        return child_docs

    def _enrich_with_parents(self, child_docs: List[Document]) -> List[Document]:
        enriched = []
        seen_parents = set()
        for child in child_docs:
            parent_id = child.metadata.get("parent_doc_id", "")
            if parent_id and parent_id not in seen_parents:
                parent = self.store.get_parent_document(parent_id)
                if parent:
                    enriched.append(parent)
                    seen_parents.add(parent_id)
                else:
                    enriched.append(child)
            else:
                enriched.append(child)
        return enriched

    def _build_context(self, documents: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            text = doc.page_content
            parts.append(f"[Document {i}] (Source: {source})\n{text}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, top_k: Optional[int] = None) -> str:
        llm = self._get_llm()
        documents = self.retrieve(question, top_k=top_k)

        if not documents:
            return "No relevant documents found to answer this question."

        context = self._build_context(documents)
        prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return response

    def stream_query(self, question: str, top_k: Optional[int] = None) -> Iterator[str]:
        llm = self._get_llm()
        documents = self.retrieve(question, top_k=top_k)

        if not documents:
            yield "No relevant documents found to answer this question."
            return

        context = self._build_context(documents)
        prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
        chain = prompt | llm | StrOutputParser()

        for chunk in chain.stream({"context": context, "question": question}):
            yield chunk

    def query_with_sources(self, question: str, top_k: Optional[int] = None) -> dict:
        documents = self.retrieve(question, top_k=top_k)
        answer = self.query(question, top_k=top_k)

        sources = []
        seen = set()
        for doc in documents:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append({
                    "source": src,
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "page": doc.metadata.get("page_number", None),
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                })

        return {"answer": answer, "sources": sources}
