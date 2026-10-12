import os
import hashlib
from typing import List, Dict, Any
from langchain_core.documents import Document
from core.models import DocumentMeta


class BaseLoader:
    def __init__(self):
        pass

    def load(self, file_path: str) -> List[Document]:
        raise NotImplementedError

    def _build_metadata(self, file_path: str, file_type: str, **kwargs) -> Dict[str, Any]:
        base_meta = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": file_type,
            "content_hash": self._hash_file(file_path),
        }
        base_meta.update(kwargs)
        return base_meta

    def _hash_file(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _wrap_document(self, text: str, metadata: Dict[str, Any]) -> Document:
        if not text or not text.strip():
            return None
        return Document(page_content=text.strip(), metadata=metadata)
