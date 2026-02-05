# 🧠 EzMemory

**AI Agent Memory System** - A standalone, agent-agnostic memory subsystem with embeddings and vector storage.

## Overview

EzMemory is a production-ready memory service designed to be plugged into AI agents, providing persistent memory capabilities with semantic search. It supports multiple embedding providers and vector databases, with a clean FastMCP interface for easy integration.

## Features

- 🔌 **Multiple Embedding Providers**: OpenAI, VoyageAI, OpenRouter
- 🗄️ **Vector Database Support**: Qdrant, Pinecone (Local storage coming soon)
- 🔍 **Semantic Search**: Fast similarity search with configurable algorithms
- 🚀 **FastMCP Server**: Ready-to-use MCP tools for memory operations
- ⚙️ **Flexible Configuration**: Easy setup with interactive CLI
- 🎨 **Modern CLI**: Beautiful terminal interface with Rich
- 📦 **Production Ready**: Built with best practices and type safety

## 📚 Documentation

- **[Getting Started](GETTING_STARTED.md)** - Complete setup guide for new users
- **[Quick Start](QUICKSTART.md)** - Get running in 5 minutes
- **[Installation](INSTALL.md)** - Detailed installation instructions
- **[Architecture](docs/ARCHITECTURE.md)** - System design and technical details
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Code organization
- **[Changelog](CHANGELOG.md)** - Version history and updates

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Docker (for local Qdrant) OR Pinecone API key
- Embedding API key (OpenAI, VoyageAI, or OpenRouter)

### Installation (5 minutes)

```bash
# 1. Clone and install
git clone <repository-url>
cd EzMemory
pip install -r requirements.txt

# 2. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 3. Run setup (in new terminal)
python ezmemory.py

# 4. Test installation
python test_setup.py
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed instructions.

The setup wizard will guide you through:
1. **Vector Database Selection** - Choose and configure Qdrant or Pinecone
2. **Embedding Provider** - Select OpenAI, VoyageAI, or OpenRouter
3. **API Keys & Configuration** - Enter credentials and settings
4. **Connection Testing** - Verify everything works
5. **MCP Server** - Optionally start the server

Configuration is saved to `~/.ezmemory/config.json`

### Usage

#### Start MCP Server

```bash
python ezmemory.py
```

The server will start and display the MCP URL:

```
┌────────────────────────────────────────────┐
│ 🚀 EzMemory MCP Server                    │
│                                            │
│ MCP Server URL:                            │
│                                            │
│ http://localhost:8080/sse                  │
│                                            │
│ Add this URL to your MCP client config.   │
└────────────────────────────────────────────┘
```

#### MCP Tools

EzMemory exposes 4 MCP tools:

**`add_memory`** - Store new memories
```python
await add_memory(content="User prefers dark mode for UI")
# Returns: {"status": "success", "memory_id": "abc123..."}
```

**`search_memory`** - Search for relevant memories
```python
await search_memory(query="What are the user's UI preferences?")
# Returns: {"status": "success", "count": 3, "results": [...]}
```

**`list_memory`** - List all stored memories
```python
await list_memory(limit=10, offset=0)
# Returns: {"status": "success", "count": 10, "total": 156, "memories": [...]}
```

**`delete_memory`** - Delete specific memories
```python
await delete_memory(memory_id="abc123...")
# Returns: {"status": "success", "memory_id": "abc123..."}
```

## Configuration

Configuration is stored at `~/.ezmemory/config.json`. You can manually edit this file or use the setup wizard.

For detailed configuration documentation, see `~/.ezmemory/config.md` (auto-generated during setup).

### Example Configuration

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "sk-..."
  },
  "vector_store": {
    "provider": "qdrant",
    "host": "localhost",
    "port": 6333,
    "collection_name": "ezmemory_collection",
    "distance_metric": "cosine"
  },
  "search": {
    "default_limit": 5
  },
  "memory": {
    "ttl_days": 90
  },
  "mcp": {
    "host": "localhost",
    "port": 8080
  }
}
```

## Architecture

```
EzMemory
├── Embeddings Layer (OpenAI, VoyageAI, OpenRouter)
├── Vector Store Layer (Qdrant, Pinecone)
├── Memory Core (Schemas, Encoder, Storage, Retrieval, Lifecycle)
├── FastMCP Server (add, search, list, delete)
└── CLI (Interactive setup & management)
```

## Project Structure

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed documentation.

## Development

### Running Tests

```bash
pytest tests/
```

### Docker Setup

```bash
# Start Qdrant locally
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

## Roadmap

- [x] OpenAI embeddings
- [x] VoyageAI embeddings  
- [x] OpenRouter embeddings
- [x] Qdrant vector store
- [x] Pinecone vector store
- [x] FastMCP server
- [x] Interactive CLI setup
- [ ] Local file-based storage
- [ ] Gemini embeddings
- [ ] Memory decay and pruning automation
- [ ] Advanced filtering and tagging
- [ ] Multi-collection support
- [ ] Web UI dashboard

## License

MIT

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📖 Documentation Structure

```
EzMemory/
├── GETTING_STARTED.md    ⭐ Start here!
├── QUICKSTART.md         - 5-minute setup
├── README.md             - This file
├── INSTALL.md            - Detailed installation
├── CHANGELOG.md          - Version history
├── SUMMARY.md            - Build overview
├── docs/
│   ├── ARCHITECTURE.md   - Technical design
│   └── PROJECT_STRUCTURE.md - Code layout
├── examples/             - Working examples
│   ├── basic_usage.py
│   └── README.md
└── test_setup.py         - Verify installation
```

## Support

- **Documentation**: `~/.ezmemory/config.md` (auto-generated)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Examples**: See `examples/` directory
