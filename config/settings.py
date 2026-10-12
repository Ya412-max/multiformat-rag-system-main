from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional
import os


class Settings(BaseSettings):
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4.1", alias="LLM_MODEL")

    embedding_model: str = Field(default="model/bge-base-zh-v1.5",
                                 alias="EMBEDDING_MODEL")

    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    lance_db_path: str = Field(default="./data/lancedb", alias="LANCE_DB_PATH")

    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    top_k: int = Field(default=5, alias="TOP_K")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @property
    def is_local_llm(self) -> bool:
        return self.llm_provider in ("ollama",)


settings = Settings()
