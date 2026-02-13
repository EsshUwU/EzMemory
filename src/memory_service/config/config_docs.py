CONFIG_DOCS_CONTENT = """# EzMemory Configuration Guide

This file explains all configuration options available in `config.json`.

## Configuration Structure

```json
{
  "embedding": { ... },
  "vector_store": { ... },
  "search": { ... },
  "memory": { ... },
  "mcp": { ... },
  "known_collections": { ... }
}
```

---

## Embedding Configuration

Configure the embedding provider used to convert text into vector embeddings.

EzMemory uses a **nested provider structure**: you set `active_provider` to choose which provider is used, and each provider has its own config block. This lets you pre-configure multiple providers and switch between them without re-entering API keys.

### Fields

#### `active_provider` (string, required)
The embedding provider currently in use.

**Available options:**
- `"openai"` - OpenAI embeddings (requires API key)
- `"voyageai"` - Voyage AI embeddings (requires API key)
- `"openrouter"` - OpenRouter embeddings (requires API key)
- `"nvidia"` - NVIDIA NIM / integrate API embeddings (requires API key)
- `"gemini"` - Google Gemini embeddings (requires API key)

**Example:**
```json
"active_provider": "openai"
```

#### Provider-specific config blocks

Each provider has its own nested config. Only the active provider's config is used at runtime.

**`openai`** – OpenAI provider config:
- `api_key` (string, optional) – OpenAI API key
- `model` (string) – Model name, default `"text-embedding-3-small"`
- `embedding_dimension` (integer, optional) – Override vector dimension

**`voyageai`** – Voyage AI provider config:
- `api_key` (string, optional) – Voyage AI API key
- `model` (string) – Model name, default `"voyage-4"`
- `embedding_dimension` (integer, optional) – Override vector dimension

**`openrouter`** – OpenRouter provider config:
- `api_key` (string, optional) – OpenRouter API key
- `model` (string) – Model name, default `"google/gemini-embedding-001"`
- `base_url` (string) – API base URL, default `"https://openrouter.ai/api/v1"`
- `http_referer` (string, optional) – HTTP referer for rankings
- `site_name` (string, optional) – Site name for rankings
- `embedding_dimension` (integer, optional) – Override vector dimension

**`nvidia`** – NVIDIA provider config:
- `api_key` (string, optional) – NVIDIA API key
- `model` (string) – Model name, default `"nvidia/nv-embedqa-e5-v5"`
- `base_url` (string) – API base URL, default `"https://integrate.api.nvidia.com/v1"`
- `embedding_dimension` (integer, optional) – Override vector dimension

**`gemini`** – Gemini provider config:
- `api_key` (string, optional) – Google AI / Gemini API key
- `model` (string) – Model name, default `"gemini-embedding-001"`
- `embedding_dimension` (integer, optional) – 768, 1536, or 3072

### Available models by provider

**OpenAI:**
- `"text-embedding-3-small"` (1536 dimensions, default, recommended)
- `"text-embedding-3-large"` (3072 dimensions, higher quality)
- `"text-embedding-ada-002"` (1536 dimensions, legacy)

**VoyageAI:**
- `"voyage-4"` (1024 dimensions, default, best general purpose)
- `"voyage-4-large"` (1024 dimensions, highest quality)
- `"voyage-3.5"`, `"voyage-3.5-lite"` (1024 dimensions)
- `"voyage-3"`, `"voyage-3-large"` (1024 dimensions)
- `"voyage-3-lite"` (512 dimensions, faster)
- `"voyage-large-2"`, `"voyage-large-2-instruct"`, `"voyage-2"`, `"voyage-01"`

**Note:** Voyage 4 series models support multiple dimensions (256, 512, 1024, 2048). Set `embedding_dimension` to use non-default sizes.

**OpenRouter:**
- `"google/gemini-embedding-001"` (3072 dimensions, default)
- `"openai/text-embedding-3-small"`, `"openai/text-embedding-3-large"`, `"openai/text-embedding-ada-002"`
- `"mistralai/mistral-embed-2312"`, `"mistralai/codestral-embed-2505"`
- `"qwen/qwen3-embedding-8b"`, `"qwen/qwen3-embedding-4b"`
- `"thenlper/gte-base"`, `"thenlper/gte-large"`
- `"baai/bge-base-en-v1.5"`, `"baai/bge-large-en-v1.5"`, `"baai/bge-m3"`
- And many more via https://openrouter.ai/models

**NVIDIA:**
- `"nvidia/nv-embedqa-e5-v5"` (default), `"nvidia/nv-embed-v1"`
- `"nvidia/llama-3_2-nemoretriever-300m-embed-v2"`, `"nvidia/llama-3_2-nemoretriever-300m-embed-v1"`
- `"nvidia/llama-3.2-nv-embedqa-1b-v2"`, `"nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"`
- `"nvidia/bge-m3"` – Dimension is auto-detected at setup.

**Gemini:**
- `"gemini-embedding-001"` (3072 dimensions default, supports 768, 1536, or 3072 via `embedding_dimension`)

### API key sources

- OpenAI: https://platform.openai.com/api-keys
- VoyageAI: https://dash.voyageai.com/
- OpenRouter: https://openrouter.ai/keys
- NVIDIA: https://build.nvidia.com/ (API key for integrate API)
- Gemini: https://aistudio.google.com/apikey

### `embedding_dimension` (integer, optional)

The vector dimension size for embeddings. If not specified, the system auto-detects based on the model.

**When to set this:**
- Using a model not in the default list
- Using a model that supports multiple dimensions (e.g., Voyage 4: 256, 512, 1024, 2048)
- Want to override the default dimension for a known model

**Default:** Auto-detected based on model, or 1024 if unknown

**Note:** If you change this value, you'll need to recreate your collection with the new dimension.

---

## Vector Store Configuration

Configure the vector database used to store and search embeddings.

### Fields

#### `provider` (string, required)
The vector store provider to use.

**Available options:**
- `"qdrant"` – Qdrant (local or cloud)
- `"pinecone"` – Pinecone serverless
- `"zilliz"` – Zilliz Cloud (Milvus)

**Coming soon:**
- `"local"` – Local file-based storage

**Example:**
```json
"provider": "qdrant"
```

#### Qdrant-specific fields

**`host`** (string, optional) – Host for self-hosted Qdrant. Use with `port`. **Default:** `"localhost"`

**`port`** (integer, optional) – Port for self-hosted Qdrant. **Default:** `6333`

**`url`** (string, optional) – Full URL for Qdrant Cloud. Use instead of host/port for cloud.

**`api_key`** (string, optional) – API key for Qdrant Cloud.

**`prefer_grpc`** (boolean, optional) – Use gRPC for better performance. **Default:** `false`

#### Pinecone-specific fields

**`api_key`** (string, required) – Pinecone API key.

**`cloud`** (string, optional) – Cloud provider: `"aws"`, `"gcp"`, or `"azure"`. **Default:** `"aws"`

**`region`** (string, optional) – Cloud region, e.g. `"us-east-1"`, `"us-central1"`, `"eastus"`. **Default:** `"us-east-1"` (aws)

#### Zilliz-specific fields

**`url`** (string, required) – Zilliz Cloud URI, e.g. `https://...zillizcloud.com:19530`

**`api_key`** (string, required) – Zilliz token (user:password or API key)

#### Common fields (all providers)

**`collection_name`** (string)
Name of the collection/index. Auto-generated during setup based on embedding provider and model, or chosen when selecting an existing collection.

**Format:** `{prefix}-{provider}-{model}` (e.g. `default-openai-text-embedding-3-small`)

**Important:** Do not manually edit unless switching to a known collection. Use "Edit Embedding Model" → "Select existing collection" in the CLI to switch.

**`distance_metric`** (string, optional)
Distance metric for similarity search.

**Available options:**
- `"cosine"` – Cosine similarity (default, recommended for text)
- `"euclid"` – Euclidean distance
- `"dot"` – Dot product
- `"manhattan"` – Manhattan distance

**Default:** `"cosine"`

---

## Search Configuration

Configure how memory search operates.

### Fields

#### `default_limit` (integer, optional)
Default number of search results to return.

**Default:** `5`

**Example:**
```json
"default_limit": 10
```

#### `score_threshold` (float, optional)
Minimum similarity score for results. Results below this threshold are filtered out.

**Default:** `null` (no threshold)

**Range:** Depends on distance_metric:
- Cosine: 0.0 to 2.0
- Dot: can be negative

**Example:**
```json
"score_threshold": 0.7
```

#### `hnsw_ef` (integer, optional)
HNSW algorithm ef parameter for search. Higher values = better accuracy but slower. **Qdrant only** (ignored by Pinecone/Zilliz).

**Default:** `null` (uses Qdrant default)

**Typical range:** 100-500

**Example:**
```json
"hnsw_ef": 200
```

---

## Memory Configuration

Configure memory lifecycle and retention policies.

### Fields

#### `ttl_days` (integer, optional)
Time to live for memories in days. Memories older than this are considered expired.

**Default:** `90` (3 months)

**Example:**
```json
"ttl_days": 365
```

#### `enable_decay` (boolean, optional)
Whether to enable memory decay over time. When enabled, older memories get lower scores.

**Default:** `false`

**Example:**
```json
"enable_decay": true
```

#### `decay_factor` (float, optional)
Decay factor applied per day. Only used if `enable_decay` is true.

**Default:** `0.95`

**Range:** 0.0 to 1.0 (closer to 1 = slower decay)

**Example:**
```json
"decay_factor": 0.98
```

---

## MCP Server Configuration

Configure the FastMCP server that exposes memory operations.

### Fields

#### `host` (string, optional)
Server host address.

**Default:** `"localhost"`

**Example:**
```json
"host": "0.0.0.0"
```

#### `port` (integer, optional)
Server port number.

**Default:** `8080`

**Example:**
```json
"port": 3000
```

#### `auto_start` (boolean, optional)
Whether to automatically start the MCP server on initialization.

**Default:** `false`

**Example:**
```json
"auto_start": true
```

---

## Known Collections

**`known_collections`** (object, auto-managed)
Maps collection names to their embedding metadata. Used when switching between collections via "Edit Embedding Model" → "Select existing collection".

**Structure:** `{ "collection_name": { "provider": "...", "model": "...", "dim": N, "http_referer": "...", "site_name": "..." } }`

**Important:** Do not manually edit. This is populated and updated by the setup wizard and when editing embedding/collection settings.

---

## Complete Example

**Qdrant Cloud:**
```json
{
  "embedding": {
    "active_provider": "openai",
    "openai": {
      "api_key": "sk-...",
      "model": "text-embedding-3-small",
      "embedding_dimension": null
    },
    "voyageai": {
      "api_key": null,
      "model": "voyage-4",
      "embedding_dimension": null
    },
    "openrouter": {},
    "nvidia": {},
    "gemini": {}
  },
  "vector_store": {
    "provider": "qdrant",
    "host": null,
    "port": null,
    "url": "https://your-cluster.cloud.qdrant.io",
    "api_key": "your-qdrant-api-key",
    "collection_name": "default-openai-text-embedding-3-small",
    "distance_metric": "cosine",
    "prefer_grpc": false
  },
  "search": { "default_limit": 5, "score_threshold": null, "hnsw_ef": null },
  "memory": { "ttl_days": 90, "enable_decay": false, "decay_factor": 0.95 },
  "mcp": { "host": "localhost", "port": 8080, "auto_start": false },
  "known_collections": {}
}
```

**Pinecone:**
```json
{
  "embedding": { "active_provider": "voyageai", ... },
  "vector_store": {
    "provider": "pinecone",
    "api_key": "your-pinecone-api-key",
    "cloud": "aws",
    "region": "us-east-1",
    "collection_name": "default-voyageai-voyage-4",
    "distance_metric": "cosine"
  },
  ...
}
```

**Zilliz:**
```json
{
  "embedding": { "active_provider": "gemini", ... },
  "vector_store": {
    "provider": "zilliz",
    "url": "https://xxx.zillizcloud.com:19530",
    "api_key": "user:password",
    "collection_name": "default_gemini_gemini_embedding_001",
    "distance_metric": "cosine"
  },
  ...
}
```

---

## Tips and Best Practices

1. **Choose the right embedding model:**
   - For most use cases, `text-embedding-3-small` (OpenAI) or `voyage-4` (VoyageAI) are good defaults
   - Use larger models for higher quality at the cost of speed and cost
   - Use specialized models (code, finance, law) for domain-specific content

2. **Distance metric selection:**
   - Stick with `cosine` for text embeddings (default, works best)
   - Consider `dot` for recommendation systems
   - Use `euclid` or `manhattan` for spatial data

3. **Search tuning:**
   - Increase `default_limit` if you want more context
   - Set `score_threshold` to filter out low-quality matches
   - Increase `hnsw_ef` for better accuracy (slower searches)

4. **Memory retention:**
   - Adjust `ttl_days` based on your use case (90 days is reasonable default)
   - Enable `enable_decay` if you want older memories to naturally fade
   - Lower `decay_factor` for faster decay

5. **Performance:**
   - Enable `prefer_grpc` for better performance with Qdrant
   - Use local Qdrant instance for development (Docker)
   - Use Qdrant Cloud, Pinecone, or Zilliz for production

6. **Switching collections:**
   - Use "Edit Embedding Model" → "Select existing collection" to switch between pre-configured collections without re-entering API keys

---

## Troubleshooting

### "Collection not found" error
Make sure to run the initialization setup first. The collection is created during setup.

### Slow searches
- Try enabling `prefer_grpc` in vector_store config (Qdrant only)
- Lower the `default_limit` in search config
- Consider using a smaller embedding model

### Out of memory errors
- Use a smaller embedding model (e.g., `voyage-3-lite`)
- Reduce `ttl_days` to store fewer memories
- Enable pruning of old memories

### API key errors
- Verify your API key is correct and has proper permissions
- Check that you're using the right provider (openai, voyageai, openrouter, nvidia, gemini)
- Ensure your API key has sufficient credits/quota

### Pinecone / Zilliz connection errors
- For Pinecone: ensure `cloud` and `region` match your index setup
- For Zilliz: ensure `url` includes the port (e.g. `:19530`) and `api_key` is the correct token
"""
