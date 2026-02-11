CONFIG_DOCS_CONTENT = """# EzMemory Configuration Guide

This file explains all configuration options available in `config.json`.

## Configuration Structure

```json
{
  "embedding": { ... },
  "vector_store": { ... },
  "search": { ... },
  "memory": { ... },
  "mcp": { ... }
}
```

---

## Embedding Configuration

Configure the embedding provider used to convert text into vector embeddings.

### Fields

#### `provider` (string, required)
The embedding provider to use.

**Available options:**
- `"openai"` - OpenAI embeddings (requires API key)
- `"voyageai"` - Voyage AI embeddings (requires API key)
- `"openrouter"` - OpenRouter embeddings (requires API key)
- `"nvidia"` - NVIDIA NIM / integrate API embeddings (requires API key)
- `"gemini"` - Google Gemini embeddings (requires API key)

**Example:**
```json
"provider": "openai"
```

#### `model` (string, required)
The specific model to use for embeddings.

**Available models by provider:**

**OpenAI:**
- `"text-embedding-3-small"` (1536 dimensions, default, recommended)
- `"text-embedding-3-large"` (3072 dimensions, higher quality)
- `"text-embedding-ada-002"` (1536 dimensions, legacy)

**VoyageAI:**
- `"voyage-4"` (1024 dimensions default, best general purpose - latest)
- `"voyage-4-large"` (1024 dimensions default, highest quality - latest)
- `"voyage-4-lite"` (1024 dimensions default, cost optimized - latest)
- `"voyage-code-4"` (1024 dimensions default, optimized for code - latest)
- `"voyage-3"` (1024 dimensions, general purpose)
- `"voyage-3-lite"` (512 dimensions, faster)
- `"voyage-code-3"` (1024 dimensions, optimized for code)

**Note:** Voyage 4 series models support multiple dimensions (256, 512, 1024, 2048). Set `embedding_dimension` to use non-default sizes.

**OpenRouter:**
- `"google/gemini-embedding-001"` (768 dimensions, default)

**NVIDIA:**
- `"nvidia/nv-embedqa-e5-v5"` (default), `"nvidia/nv-embed-v1"`, `"nvidia/bge-m3"`, and others. Dimension is auto-detected at setup.

**Gemini:**
- `"gemini-embedding-001"` (3072 dimensions default, supports 768, 1536, or 3072 via `embedding_dimension`)

**Example:**
```json
"model": "text-embedding-3-small"
```

#### `api_key` (string, required)
API key for the embedding provider.

**How to get:**
- OpenAI: https://platform.openai.com/api-keys
- VoyageAI: https://dash.voyageai.com/
- OpenRouter: https://openrouter.ai/keys
- NVIDIA: https://build.nvidia.com/ (API key for integrate API)
- Gemini: https://aistudio.google.com/apikey

**Example:**
```json
"api_key": "sk-..."
```

#### `base_url` (string, optional)
Custom base URL for the API. Only used for OpenRouter.

**Default:** `null`

**Example:**
```json
"base_url": "https://openrouter.ai/api/v1"
```

#### `http_referer` (string, optional)
HTTP referer for rankings on OpenRouter. Only used for OpenRouter.

**Default:** `null`

**Example:**
```json
"http_referer": "https://yourdomain.com"
```

#### `site_name` (string, optional)
Site name for rankings on OpenRouter. Only used for OpenRouter.

**Default:** `null`

**Example:**
```json
"site_name": "My App"
```

#### `embedding_dimension` (integer, optional)
The vector dimension size for embeddings. If not specified, the system will auto-detect based on the model.

**When to set this:**
- Using a model not in the default list
- Using a model that supports multiple dimensions (e.g., Voyage 4 supports 256, 512, 1024, 2048)
- Want to override the default dimension for a known model

**Default:** Auto-detected based on model, or 1024 if unknown

**Common dimensions:**
- OpenAI `text-embedding-3-small`: 1536
- OpenAI `text-embedding-3-large`: 3072
- Voyage 3/4 series: 1024 (default), also supports 256, 512, 2048
- Gemini embedding: 768

**Example:**
```json
"embedding_dimension": 1024
```

**Note:** If you change this value, you'll need to recreate your collection with the new dimension.

---

## Vector Store Configuration

Configure the vector database used to store and search embeddings.

### Fields

#### `provider` (string, required)
The vector store provider to use.

**Available options:**
- `"qdrant"` - Qdrant vector database (currently supported)

**Coming soon:**
- `"pinecone"` - Pinecone vector database
- `"local"` - Local file-based storage

**Example:**
```json
"provider": "qdrant"
```

#### `host` (string, optional)
Host address for self-hosted Qdrant instance.

**Default:** `"localhost"`

**Example:**
```json
"host": "localhost"
```

#### `port` (integer, optional)
Port number for self-hosted Qdrant instance.

**Default:** `6333`

**Example:**
```json
"port": 6333
```

#### `url` (string, optional)
Full URL for Qdrant Cloud instances. Use this instead of host/port for cloud.

**Default:** `null`

**Example:**
```json
"url": "https://your-cluster.cloud.qdrant.io"
```

#### `api_key` (string, optional)
API key for Qdrant Cloud instances.

**Default:** `null`

**Example:**
```json
"api_key": "your-qdrant-api-key"
```

#### `collection_name` (string, auto-generated)
Name of the collection to store memories in. **This is automatically generated and managed by the system.**

**Auto-generated format:** `ezmemory_{provider}_{model}`
- The collection name is automatically generated based on your embedding provider and model
- This ensures each embedding model gets its own collection (since different models produce different vector dimensions)
- When you change your embedding model in the config, the collection name updates automatically
- Examples:
  - `ezmemory_openai_text_embedding_3_small`
  - `ezmemory_voyageai_voyage_4`
  - `ezmemory_openrouter_text_embedding_3_large`

**Important:** Do not manually edit this field. It will be automatically updated to match your embedding configuration.

#### `distance_metric` (string, optional)
Distance metric used for similarity search.

**Available options:**
- `"cosine"` - Cosine similarity (default, recommended for text)
  - Range: 0 to 2 (higher is more similar)
  - Best for normalized vectors like text embeddings
- `"euclid"` - Euclidean distance
  - Range: 0 to ∞ (lower is more similar)
  - Good for spatial data, image features
- `"dot"` - Dot product
  - Range: -∞ to ∞ (higher is more similar)
  - Good for recommendations, unnormalized vectors
- `"manhattan"` - Manhattan distance
  - Range: 0 to ∞ (lower is more similar)
  - Good for sparse features, discrete data

**Default:** `"cosine"`

**Example:**
```json
"distance_metric": "cosine"
```

#### `prefer_grpc` (boolean, optional)
Whether to use gRPC instead of HTTP for better performance.

**Default:** `false`

**Example:**
```json
"prefer_grpc": true
```

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
HNSW algorithm ef parameter for search. Higher values = better accuracy but slower.

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

## Complete Example

Here's a complete example configuration:

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
    "collection_name": "ezmemory_openai_text_embedding_3_small",
    "distance_metric": "cosine",
    "prefer_grpc": false
  },
  "search": {
    "default_limit": 5,
    "score_threshold": null,
    "hnsw_ef": null
  },
  "memory": {
    "ttl_days": 90,
    "enable_decay": false,
    "decay_factor": 0.95
  },
  "mcp": {
    "host": "localhost",
    "port": 8080,
    "auto_start": false
  }
}
```

---

## Tips and Best Practices

1. **Choose the right embedding model:**
   - For most use cases, `text-embedding-3-small` (OpenAI) or `voyage-3` (VoyageAI) are good defaults
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
   - Use Qdrant Cloud for production

---

## Troubleshooting

### "Collection not found" error
Make sure to run the initialization setup first. The collection is created during setup.

### Slow searches
- Try enabling `prefer_grpc` in vector_store config
- Lower the `default_limit` in search config
- Consider using a smaller embedding model

### Out of memory errors
- Use a smaller embedding model (e.g., `voyage-3-lite`)
- Reduce `ttl_days` to store fewer memories
- Enable pruning of old memories

### API key errors
- Verify your API key is correct and has proper permissions
- Check that you're using the right provider (openai, voyageai, openrouter, nvidia)
- Ensure your API key has sufficient credits/quota
"""
