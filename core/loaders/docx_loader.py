import os
from typing import List
from langchain_core.documents import Document
from core.loaders.base import BaseLoader


class DocxLoader(BaseLoader):
    def __init__(self, extract_tables: bool = True):
        super().__init__()
        self.extract_tables = extract_tables

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".docx", ".doc"):
            raise ValueError(f"DocxLoader only supports .docx/.doc files, got {ext}")

        try:
            return self._load_with_python_docx(file_path)
        except ImportError:
            print("⚠️ 请安装 python-docx: pip install python-docx")
            return []
        except Exception as e:
            print(f"❌ DOCX 加载失败: {e}")
            return []

    def _load_with_python_docx(self, file_path: str) -> List[Document]:
        from docx import Document as DocxDoc

        doc = DocxDoc(file_path)
        documents = []

        all_paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                all_paragraphs.append(text)

        if all_paragraphs:
            full_text = "\n\n".join(all_paragraphs)
            meta = self._build_metadata(file_path, "docx", element_type="paragraph")
            d = self._wrap_document(full_text, meta)
            if d:
                documents.append(d)

        if self.extract_tables:
            for i, table in enumerate(doc.tables):
                rows = []
                for row in table.rows:
                    cells = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        cells.append(cell_text if cell_text else "")
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    table_text = "\n".join(rows)
                    meta = self._build_metadata(
                        file_path,
                        "docx",
                        element_type="table",
                        table_index=i
                    )
                    d = self._wrap_document(table_text, meta)
                    if d:
                        documents.append(d)

        return documents