import os
from typing import List
from langchain_core.documents import Document
from core.loaders.base import BaseLoader


class ImageLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, language: str = "ch_sim+en"):
        super().__init__()
        self.language = language

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"ImageLoader supports {self.SUPPORTED_EXTENSIONS}, got {ext}")

        file_type = ext.lstrip(".")

        try:
            text = self._ocr_with_easyocr(file_path)
            if text and text.strip():
                meta = self._build_metadata(file_path, file_type, ocr_engine="easyocr")
                doc = self._wrap_document(text, meta)
                return [doc] if doc else []
        except Exception:
            pass

        return []

    def _ocr_with_easyocr(self, file_path: str) -> str:
        import easyocr
        import numpy as np
        from PIL import Image

        langs = []
        if "ch_sim" in self.language:
            langs.append("ch_sim")
        if "en" in self.language:
            langs.append("en")
        if not langs:
            langs = ["en"]

        reader = easyocr.Reader(langs, gpu=False)
        image = Image.open(file_path)
        np_image = np.array(image)
        results = reader.readtext(np_image)
        lines = [item[1] for item in results if item[1].strip()]
        return "\n".join(lines)
