import os
from typing import List, Optional, Dict
from langchain_core.documents import Document
from core.loaders.base import BaseLoader
from core.loaders.pdf_loader import PDFLoader
from core.loaders.docx_loader import DocxLoader
from core.loaders.text_loader import TextLoader
from core.loaders.html_loader import HTMLLoader
from core.loaders.excel_loader import ExcelLoader
from core.loaders.image_loader import ImageLoader


class DocumentLoader:

    def __init__(
        self,
        extract_tables: bool = True,
        ocr_language: str = "ch_sim+en",
    ):
        self.pdf_loader = PDFLoader(extract_tables=extract_tables)
        self.docx_loader = DocxLoader(extract_tables=extract_tables)
        self.text_loader = TextLoader()
        self.html_loader = HTMLLoader()
        self.excel_loader = ExcelLoader()
        self.image_loader = ImageLoader(language=ocr_language)

    def _get_loader(self, file_path: str) -> Optional[BaseLoader]:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self.pdf_loader
        elif ext in (".docx", ".doc"):
            return self.docx_loader
        elif ext in TextLoader.SUPPORTED_EXTENSIONS:
            return self.text_loader
        elif ext in (".html", ".htm"):
            return self.html_loader
        elif ext in ExcelLoader.SUPPORTED_EXTENSIONS:
            return self.excel_loader
        elif ext in ImageLoader.SUPPORTED_EXTENSIONS:
            return self.image_loader
        return None

    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = self._get_loader(file_path)
        if loader is None:
            ext = os.path.splitext(file_path)[1].lower()
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load(file_path)

    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        all_documents = []
        if recursive:
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    try:
                        docs = self.load(file_path)
                        all_documents.extend(docs)
                    except ValueError:
                        pass
        else:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    try:
                        docs = self.load(file_path)
                        all_documents.extend(docs)
                    except ValueError:
                        pass
        return all_documents

    @property
    def supported_formats(self) -> list:
        return [
            ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
            ".html", ".htm", ".xlsx", ".xls", ".csv",
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
        ]
