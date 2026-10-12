import os
from typing import List
from langchain_core.documents import Document
from core.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    def __init__(self, extract_tables: bool = True, extract_images: bool = False):
        super().__init__()
        self.extract_tables = extract_tables
        self.extract_images = extract_images

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".pdf":
            raise ValueError(f"PDFLoader only supports .pdf files, got {ext}")

        documents = []

        try:
            documents = self._load_with_pdfplumber(file_path)
            if documents:
                return documents
        except Exception:
            pass

        try:
            documents = self._load_with_unstructured(file_path)
            return documents
        except Exception:
            pass

        try:
            documents = self._load_with_pymupdf(file_path)
            return documents
        except Exception:
            pass

        return []

    def _load_with_pdfplumber(self, file_path: str) -> List[Document]:
        import pdfplumber
        documents = []
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    meta = self._build_metadata(
                        file_path, "pdf",
                        page_number=i + 1,
                        total_pages=total,
                        title=f"Page {i + 1}",
                    )
                    doc = self._wrap_document(text, meta)
                    if doc:
                        documents.append(doc)

                if self.extract_tables:
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        table_text = self._table_to_text(table)
                        if table_text:
                            meta = self._build_metadata(
                                file_path, "pdf",
                                page_number=i + 1,
                                total_pages=total,
                                title=f"Page {i + 1} Table {j + 1}",
                                element_type="table",
                                table_index=j,
                            )
                            doc = self._wrap_document(table_text, meta)
                            if doc:
                                documents.append(doc)
        return documents

    def _load_with_unstructured(self, file_path: str) -> List[Document]:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=file_path,
            strategy="auto",
        )
        return self._process_unstructured_elements(elements, file_path)

    def _load_with_pymupdf(self, file_path: str) -> List[Document]:
        import fitz
        documents = []
        doc = fitz.open(file_path)
        total = len(doc)
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text and text.strip():
                meta = self._build_metadata(
                    file_path, "pdf",
                    page_number=i + 1,
                    total_pages=total,
                    title=f"Page {i + 1}",
                )
                d = self._wrap_document(text, meta)
                if d:
                    documents.append(d)
        doc.close()
        return documents

    def _process_unstructured_elements(self, elements, file_path: str) -> List[Document]:
        documents = []
        current_page = 0
        for el in elements:
            page_num = getattr(el.metadata, "page_number", None) or current_page
            current_page = page_num
            text = str(el)

            meta = self._build_metadata(
                file_path, "pdf",
                page_number=page_num,
                title=getattr(el, "category", ""),
                element_type=getattr(el, "category", "text"),
            )
            doc = self._wrap_document(text, meta)
            if doc:
                documents.append(doc)
        return documents

    def _table_to_text(self, table) -> str:
        if not table:
            return ""
        lines = []
        for row in table:
            row_text = " | ".join([str(cell) if cell else "" for cell in row])
            if row_text.strip():
                lines.append(row_text)
        return "\n".join(lines)
