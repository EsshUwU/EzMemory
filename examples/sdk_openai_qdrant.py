"""
SDK example: OpenAI embeddings + Qdrant vector store (no EzMemory config file).

Requirements:
    pip install ezmemory openai qdrant-client

Environment variables:
    OPENAI_API_KEY   - required
    QDRANT_URL       - optional (if using Qdrant Cloud)
    QDRANT_API_KEY   - optional (for Qdrant Cloud)
    QDRANT_HOST      - optional, default "localhost"
    QDRANT_PORT      - optional, default "6333"
"""

from __future__ import annotations

import os

from ezmemory import (  # type: ignore[import]
    Memory,
    MemoryEncoder,
    MemoryRetrieval,
    MemoryStorage,
    OpenAIEmbedding,
    QdrantVectorStore,
)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    # 1) Embedding provider
    embedding = OpenAIEmbedding(api_key=api_key, model="text-embedding-3-small")
    dim = embedding.get_dimension()

    # 2) Vector store client
    vector_store = QdrantVectorStore(
        url=qdrant_url or None,
        api_key=qdrant_api_key or None,
        host=None if qdrant_url else qdrant_host,
        port=None if qdrant_url else qdrant_port,
        prefer_grpc=False,
    )

    collection_name = "ezmemory_sdk_openai_qdrant"
    if not vector_store.collection_exists(collection_name):
        vector_store.create_collection(
            collection_name=collection_name,
            vector_size=dim,
            distance_metric="cosine",
        )

    # 3) Memory components
    encoder = MemoryEncoder(embedding)
    storage = MemoryStorage(vector_store, collection_name)
    retrieval = MemoryRetrieval(vector_store, collection_name)

    # 4) Add memories
    examples = [
        "User prefers dark mode for all applications.",
        "User's favorite programming language is Python.",
        "User enjoys hiking on weekends.",
    ]

    ids = []
    for text in examples:
        mem = Memory(content=text)
        mem = encoder.encode(mem)
        mid = storage.store(mem)
        ids.append(mid)
        print(f"Stored: {text} (id={mid[:8]}...)")

    # 5) Search
    query = "What does the user like to do?"
    qvec = encoder.encode_query(query)
    results = retrieval.search(qvec, limit=3)

    print(f"\nQuery: {query}")
    for i, res in enumerate(results, 1):
        print(f"{i}. score={res.score:.4f}  content={res.memory.content}")


if __name__ == "__main__":
    main()

