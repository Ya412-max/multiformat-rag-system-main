import os
import json
from typing import List, Optional, Dict, Any
import lancedb
from langchain_core.documents import Document
from config.settings import settings
import pyarrow.compute as pc


class LanceDBStore:
    TABLE_NAME = "rag_documents"

    def __init__(
            self,
            db_path: Optional[str] = None,
            embedder=None,
    ):
        self.db_path = os.path.abspath(db_path or settings.lance_db_path)
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from core.embedder import BGEEmbedder
            self._embedder = BGEEmbedder()
        return self._embedder

    def _get_table(self):
        if self.TABLE_NAME not in self.db.table_names():
            return None
        return self.db.open_table(self.TABLE_NAME)

    def create_table(self, overwrite: bool = False):
        if overwrite and self.TABLE_NAME in self.db.table_names():
            self.db.drop_table(self.TABLE_NAME)

        if self.TABLE_NAME not in self.db.table_names():
            import pyarrow as pa
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), list_size=self.embedder.dimension)),
                pa.field("text", pa.string()),
                pa.field("chunk_type", pa.string()),
                pa.field("parent_doc_id", pa.string()),
                pa.field("metadata", pa.string()),
            ])
            self.db.create_table(self.TABLE_NAME, schema=schema)

    def add_documents(self, documents: List[Document]) -> int:
        table = self._get_table()
        if table is None:
            self.create_table()
            table = self._get_table()

        rows = []
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_documents(texts)

        for doc, vector in zip(documents, embeddings):
            metadata = doc.metadata.copy()
            if "source" not in metadata:
                metadata["source"] = metadata.get("file_path") or metadata.get("filename") or "unknown"
            if "file_type" not in metadata:
                metadata["file_type"] = metadata.get("extension") or "unknown"
            if "file_path" not in metadata:
                metadata["file_path"] = metadata.get("source") or "unknown"

            rows.append({
                "id": doc.metadata.get("doc_id", ""),
                "vector": [float(v) for v in vector],
                "text": doc.page_content,
                "chunk_type": doc.metadata.get("chunk_type", "child"),
                "parent_doc_id": doc.metadata.get("parent_doc_id", ""),
                "metadata": json.dumps(metadata, ensure_ascii=False),
            })

        table.add(rows)
        return len(rows)

    def search(
            self,
            query: str,
            top_k: Optional[int] = None,
            chunk_type: str = "child",
    ) -> List[Document]:
        table = self._get_table()
        if table is None:
            return []

        top_k = top_k or settings.top_k
        query_vector = self.embedder.embed_query(query)

        results = (
            table.search(query_vector, vector_column_name="vector")
            .where(f"chunk_type = '{chunk_type}'", prefilter=True)
            .limit(top_k)
            .to_list()
        )

        return self._results_to_documents(results)


    def hybrid_search(
            self,
            query: str,
            top_k: Optional[int] = None,
            chunk_type: str = "child",
    ) -> List[Document]:
        table = self._get_table()
        if table is None:
            return []

        top_k = top_k or settings.top_k
        query_vector = self.embedder.embed_query(query)
        half_k = max(top_k // 2, 2)

        vector_results = (
            table.search(query_vector, vector_column_name="vector")
            .where(f"chunk_type = '{chunk_type}'", prefilter=True)
            .limit(half_k)
            .to_list()
        )

        fts_result = []
        try:
            fts_result = (
                table.search(query, query_type="fts")
                .where(f"chunk_type = '{chunk_type}'", prefilter=True)
                .limit(half_k)
                .to_list()
            )
        except Exception:
            pass

        rrf_k = 60
        score_dict: Dict[str, float] = {}
        item_map: Dict[str, Dict] = {}

        # 向量召回打分
        for rank, item in enumerate(vector_results, start=1):
            doc_id = item["id"]
            item_map[doc_id] = item
            score_dict[doc_id] = score_dict.get(doc_id, 0.0) + 1.0 / (rank + rrf_k)

        # 关键词FTS召回打分
        for rank, item in enumerate(fts_result, start=1):
            doc_id = item["id"]
            item_map[doc_id] = item
            score_dict[doc_id] = score_dict.get(doc_id, 0.0) + 1.0 / (rank + rrf_k)

        # RRF分数降序排序，截取top_k
        sorted_doc_ids = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)
        merged = [item_map[doc_id] for doc_id in sorted_doc_ids[:top_k]]

        return self._results_to_documents(merged)

    def get_parent_document(self, parent_doc_id: str) -> Optional[Document]:
        table = self._get_table()
        if table is None:
            return None

        results = table.search().where(f"id = '{parent_doc_id}'", prefilter=True).limit(1).to_list()
        if results:
            docs = self._results_to_documents(results)
            return docs[0]
        return None


    def delete_by_source(self, file_path: str) -> int:
        table = self._get_table()
        if table is None:
            return 0

        file_name = os.path.basename(file_path)

        table_ref = table.to_arrow()
        mask = pc.match_substring(pc.field("metadata"), file_name)

        ids_to_delete = table_ref.filter(mask).column("id").to_pylist()

        if ids_to_delete:
            ids_str = ",".join([repr(i) for i in ids_to_delete])
            table.delete(f"id IN ({ids_str})")
            return len(ids_to_delete)

        return 0

    def list_sources(self) -> List[Dict[str, Any]]:
        """列出所有文档来源"""
        table = self._get_table()
        if table is None:
            return []

        df = table.to_pandas()
        sources = {}

        for metadata_json in df["metadata"]:
            try:
                meta = json.loads(metadata_json)
                src = meta.get("source")
                if src and src not in sources:
                    sources[src] = {
                        "source": src,
                        "file_type": meta.get("file_type", "unknown"),
                        "file_path": meta.get("file_path", src)
                    }
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return list(sources.values())

    def count_documents(self) -> int:
        table = self._get_table()
        if table is None:
            return 0
        return table.count_rows()

    def _results_to_documents(self, results: List[dict]) -> List[Document]:
        documents = []
        for r in results:
            metadata = {}
            try:
                metadata = json.loads(r.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                pass
            metadata["chunk_type"] = r.get("chunk_type", "")
            metadata["parent_doc_id"] = r.get("parent_doc_id", "")
            metadata["doc_id"] = r.get("id", "")

            doc = Document(
                page_content=r.get("text", ""),
                metadata=metadata,

            )
            documents.append(doc)
        return documents