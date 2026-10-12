import os
from typing import List
from langchain_core.documents import Document
from core.loaders.base import BaseLoader


class HTMLLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".html", ".htm"):
            raise ValueError(f"HTMLLoader only supports .html/.htm files, got {ext}")

        try:
            documents = self._load_with_unstructured(file_path)
            if documents:
                return documents
        except Exception:
            pass

        try:
            documents = self._load_with_bs4(file_path)
            return documents
        except Exception:
            pass

        return []

    def _load_with_unstructured(self, file_path: str) -> List[Document]:
        from unstructured.partition.html import partition_html

        elements = partition_html(filename=file_path)
        documents = []
        for el in elements:
            text = str(el)
            meta = self._build_metadata(
                file_path, "html",
                element_type=getattr(el, "category", "text"),
            )
            doc = self._wrap_document(text, meta)
            if doc:
                documents.append(doc)
        return documents

    def _load_with_bs4(self, file_path: str) -> List[Document]:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            return []

        meta = self._build_metadata(file_path, "html")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            meta["title"] = title_tag.string.strip()
        doc = self._wrap_document(clean_text, meta)
        return [doc] if doc else []
