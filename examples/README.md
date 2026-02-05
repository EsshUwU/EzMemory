# EzMemory Examples

This directory contains example scripts demonstrating how to use EzMemory.

## Prerequisites

Make sure you've run the setup first:

```bash
python ezmemory.py
```

## Examples

### basic_usage.py

Demonstrates core functionality:
- Loading configuration
- Initializing components
- Adding memories
- Searching memories
- Listing all memories
- Deleting memories

Run it:
```bash
python examples/basic_usage.py
```

## Creating Your Own Scripts

Here's a minimal template:

```python
from src.memory_service.config import config_manager
from src.memory_service.embeddings import OpenAIEmbedding
from src.memory_service.vector_store import QdrantVectorStore
from src.memory_service.memory import Memory, MemoryEncoder, MemoryStorage, MemoryRetrieval

# Load config
config = config_manager.load()

# Initialize components
embedding = OpenAIEmbedding(
    api_key=config.embedding.api_key,
    model=config.embedding.model
)
vector_store = QdrantVectorStore(
    host=config.vector_store.host,
    port=config.vector_store.port
)

# Create memory system
encoder = MemoryEncoder(embedding)
storage = MemoryStorage(vector_store, config.vector_store.collection_name)
retrieval = MemoryRetrieval(vector_store, config.vector_store.collection_name)

# Use it!
memory = Memory(content="Your content here")
memory = encoder.encode(memory)
memory_id = storage.store(memory)
```

## More Examples Coming Soon

- Advanced search with filtering
- Memory lifecycle management
- Batch operations
- Integration with LangChain
- Integration with LlamaIndex
