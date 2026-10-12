import os
from typing import List
from langchain_core.documents import Document
from core.loaders.base import BaseLoader


class ExcelLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

    def __init__(self, max_rows_per_chunk: int = 50):
        super().__init__()
        self.max_rows_per_chunk = max_rows_per_chunk

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"ExcelLoader supports {self.SUPPORTED_EXTENSIONS}, got {ext}")

        documents = []

        try:
            documents = self._load_with_unstructured(file_path)
            return documents
        except Exception:
            pass

        try:
            documents = self._load_with_pandas(file_path)
            return documents
        except Exception:
            pass

        return []

    def _load_with_unstructured(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            from unstructured.partition.csv import partition_csv
            elements = partition_csv(filename=file_path)
        else:
            from unstructured.partition.xlsx import partition_xlsx
            elements = partition_xlsx(filename=file_path)

        file_type = ext.lstrip(".")
        documents = []
        current_sheet = ""
        for el in elements:
            text = str(el)
            sheet_name = getattr(el.metadata, "sheet_name", "") if hasattr(el.metadata, "sheet_name") else current_sheet
            current_sheet = sheet_name
            meta = self._build_metadata(
                file_path, file_type,
                sheet_name=sheet_name,
                element_type=getattr(el, "category", "table"),
            )
            doc = self._wrap_document(text, meta)
            if doc:
                documents.append(doc)
        return documents

    def _load_with_pandas(self, file_path: str) -> List[Document]:
        import pandas as pd

        ext = os.path.splitext(file_path)[1].lower()
        file_type = ext.lstrip(".")

        if ext == ".csv":
            sheets = {"Sheet1": pd.read_csv(file_path, encoding="utf-8")}
        else:
            xls = pd.ExcelFile(file_path)
            sheets = {}
            for sheet_name in xls.sheet_names:
                sheets[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)

        documents = []
        for sheet_name, df in sheets.items():
            total_rows = len(df)
            for start in range(0, total_rows, self.max_rows_per_chunk):
                end = min(start + self.max_rows_per_chunk, total_rows)
                chunk = df.iloc[start:end]
                chunk_text = chunk.to_csv(index=False, encoding="utf-8")
                meta = self._build_metadata(
                    file_path, file_type,
                    sheet_name=sheet_name,
                    row_start=start,
                    row_end=end,
                    total_rows=total_rows,
                )
                doc = self._wrap_document(chunk_text, meta)
                if doc:
                    documents.append(doc)
        return documents
