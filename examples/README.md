# EzMemory Examples

This directory contains **SDK-focused** examples for EzMemory. They assume
you've installed the package from PyPI and are importing it in your own code.

The distribution name is `ezmemory`, and the recommended import is:

```python
from ezmemory import Memory, MemoryEncoder, MemoryStorage, MemoryRetrieval
```

You can also import the underlying package directly:

```python
import memory_service
```

## Prerequisites

Install the core package:

```bash
pip install ezmemory
```

Depending on which example you run, you may also need:

- OpenAI: `pip install openai`
- VoyageAI: `pip install voyageai`
- Qdrant: `pip install qdrant-client`

## Examples

### `basic_usage.py` – SDK + EzMemory config file

Uses the **EzMemory config file** created by the CLI to:

- Load configuration (`~/.ezmemory/config.json`)
- Initialize the active embedding provider (OpenAI, VoyageAI, or OpenRouter)
- Initialize the configured vector store (Qdrant, Pinecone, or Zilliz)
- Add, search, list, and delete memories

Run:

```bash
python -m memory_service.main   # one-time setup wizard
python examples/basic_usage.py
```

### `sdk_openai_qdrant.py` – pure SDK, OpenAI + Qdrant

Does **not** depend on EzMemory’s config file. You wire everything up yourself
using environment variables and the SDK classes:

- `OpenAIEmbedding`
- `QdrantVectorStore`
- `Memory`, `MemoryEncoder`, `MemoryStorage`, `MemoryRetrieval`

Environment:

- `OPENAI_API_KEY` (required)
- `QDRANT_URL` / `QDRANT_API_KEY` (optional, for Qdrant Cloud)
- `QDRANT_HOST` / `QDRANT_PORT` (optional, defaults to `localhost:6333`)

Run:

```bash
python examples/sdk_openai_qdrant.py
```

### `sdk_voyage_qdrant.py` – pure SDK, VoyageAI + Qdrant

Same pattern as above but using VoyageAI:

- `VoyageEmbedding`
- `QdrantVectorStore`

Environment:

- `VOYAGE_API_KEY` (required)
- Qdrant variables as above

Run:

```bash
python examples/sdk_voyage_qdrant.py
```

### `sdk_openrouter_qdrant.py` – pure SDK, OpenRouter + Qdrant

Uses OpenRouter as the embedding provider:

- `OpenRouterEmbedding`
- `QdrantVectorStore`

Environment:

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_REFERRER` / `OPENROUTER_SITE` (optional, for rankings)
- Qdrant variables as above

Run:

```bash
python examples/sdk_openrouter_qdrant.py
```

## Minimal SDK template

Here is a minimal pattern you can adapt in your own project (OpenAI + Qdrant):

```python
from ezmemory import (
    Memory,
    MemoryEncoder,
    MemoryStorage,
    MemoryRetrieval,
    OpenAIEmbedding,
    QdrantVectorStore,
)

embedding = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
vector_store = QdrantVectorStore(host="localhost", port=6333)

collection = "my_memory_collection"
if not vector_store.collection_exists(collection):
    vector_store.create_collection(
        collection_name=collection,
        vector_size=embedding.get_dimension(),
        distance_metric="cosine",
    )

encoder = MemoryEncoder(embedding)
storage = MemoryStorage(vector_store, collection)
retrieval = MemoryRetrieval(vector_store, collection)

memory = Memory(content="Your content here")
memory = encoder.encode(memory)
memory_id = storage.store(memory)
```

