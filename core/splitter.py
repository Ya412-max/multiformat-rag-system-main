import hashlib
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from config.settings import settings


class BGEChunkSplitter:
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "：", "，", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        parent_chunk_size: Optional[int] = None,
        enable_parent_doc: bool = True,
        embedding_model_name: Optional[str] = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.enable_parent_doc = enable_parent_doc
        self.parent_chunk_size = parent_chunk_size or (self.chunk_size * 4)

        model_name = embedding_model_name or settings.embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
        )
        self.parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not self.enable_parent_doc:
            return self.child_splitter.split_documents(documents)

        result = []
        for doc in documents:
            chunks = self._split_with_parent_child(doc)
            result.extend(chunks)
        return result

    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        meta = metadata or {}
        doc = Document(page_content=text, metadata=meta)
        return self.split_documents([doc])

    def _split_with_parent_child(self, document: Document) -> List[Document]:
        """
        ✅标准父子分块：
        1.原始文档先切割出父块
        2.对每一个父块，单独切割生成它的子块
        3.子块天然绑定父块id，不需要token交集匹配 _find_best_parent
        """
        # 第一步：原始文档切父块
        parent_chunks = self.parent_splitter.split_documents([document])

        all_docs = []

        for parent in parent_chunks:
            # 生成父块唯一ID
            parent_id = self._generate_id(parent.page_content)
            parent.metadata["doc_id"] = parent_id
            parent.metadata["chunk_type"] = "parent"
            all_docs.append(parent)

            child_chunks = self.child_splitter.split_documents([parent])

            for child in child_chunks:
                child_id = self._generate_id(child.page_content)
                child.metadata["doc_id"] = child_id
                child.metadata["chunk_type"] = "child"
                child.metadata["parent_doc_id"] = parent_id
                all_docs.append(child)

        return all_docs


    def _generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def merge_child_with_parent(self, child_doc: Document, all_docs: Dict[str, Document]) -> str:
        parent_id = child_doc.metadata.get("parent_doc_id", "")
        if parent_id and parent_id in all_docs:
            return all_docs[parent_id].page_content
        return child_doc.page_content
