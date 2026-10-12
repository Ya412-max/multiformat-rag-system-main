import os
from typing import List
from langchain_core.documents import Document
from core.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".log", ".rst"}

    def __init__(self):
        super().__init__()

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"TextLoader supports {self.SUPPORTED_EXTENSIONS}, got {ext}")

        file_type = ext.lstrip(".")
        encoding = self._detect_encoding(file_path)

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            text = f.read()

        if not text or not text.strip():
            return []

        meta = self._build_metadata(file_path, file_type)
        doc = self._wrap_document(text, meta)
        return [doc] if doc else []

    def _detect_encoding(self, file_path: str) -> str:
        encodings = ["utf-8", "gbk", "gb2312", "gb18030"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    f.read(1024)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return "latin-1"
