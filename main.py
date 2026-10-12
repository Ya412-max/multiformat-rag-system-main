from config.settings import settings


def main():
    print("=" * 50)
    print("RAG System Configuration")
    print("=" * 50)
    print(f"LLM Provider:   {settings.llm_provider}")
    print(f"LLM Model:      {settings.llm_model}")
    print(f"Base URL:       {settings.llm_base_url}")
    print(f"Embedding:      {settings.embedding_model}")
    print(f"Device:         {settings.embedding_device}")
    print(f"Vector DB:      {settings.lance_db_path}")
    print(f"Chunk Size:     {settings.chunk_size}")
    print(f"Chunk Overlap:  {settings.chunk_overlap}")
    print(f"Top K:          {settings.top_k}")
    print("=" * 50)


if __name__ == "__main__":
    main()
