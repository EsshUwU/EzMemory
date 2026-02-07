"""
SDK example: OpenRouter embeddings + Qdrant vector store (no EzMemory config file).

Requirements:
    pip install ezmemory openai qdrant-client

Environment variables:
    OPENROUTER_API_KEY  - required
    OPENROUTER_REFERRER - optional (HTTP referer for rankings)
    OPENROUTER_SITE     - optional (site name for rankings)
    QDRANT_URL          - optional (if using Qdrant Cloud)
    QDRANT_API_KEY      - optional (for Qdrant Cloud)
    QDRANT_HOST         - optional, default "localhost"
    QDRANT_PORT         - optional, default "6333"
"""

from __future__ import annotations

import os

from ezmemory import (  # type: ignore[import]
    Memory,
    MemoryEncoder,
    MemoryRetrieval,
    MemoryStorage,
    OpenRouterEmbedding,
    QdrantVectorStore,
)


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENROUTER_API_KEY in your environment.")

    referer = os.getenv("OPENROUTER_REFERRER")
    site = os.getenv("OPENROUTER_SITE")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    # 1) Embedding provider
    embedding = OpenRouterEmbedding(
        api_key=api_key,
        model="google/gemini-embedding-001",
        http_referer=referer,
        site_name=site,
    )
    dim = embedding.get_dimension()

    # 2) Vector store client
    vector_store = QdrantVectorStore(
        url=qdrant_url or None,
        api_key=qdrant_api_key or None,
        host=None if qdrant_url else qdrant_host,
        port=None if qdrant_url else qdrant_port,
        prefer_grpc=False,
    )

    collection_name = "ezmemory_sdk_openrouter_qdrant"
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
        "User is building an AI agent using OpenRouter.",
        "User prefers providers with good community rankings.",
        "User often experiments with different embedding models.",
    ]

    ids = []
    for text in examples:
        mem = Memory(content=text)
        mem = encoder.encode(mem)
        mid = storage.store(mem)
        ids.append(mid)
        print(f"Stored: {text} (id={mid[:8]}...)")

    # 5) Search
    query = "What is the user building?"
    qvec = encoder.encode_query(query)
    results = retrieval.search(qvec, limit=3)

    print(f"\nQuery: {query}")
    for i, res in enumerate(results, 1):
        print(f"{i}. score={res.score:.4f}  content={res.memory.content}")


if __name__ == "__main__":
    main()

