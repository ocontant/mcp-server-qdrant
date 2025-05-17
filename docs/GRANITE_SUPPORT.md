# Using IBM Granite Embedding Models with Qdrant MCP Server

This document explains how to use IBM Granite embedding models with the Qdrant MCP server through LM Studio. IBM Granite models are high-quality embedding models that can provide excellent semantic search capabilities for your applications.

## IBM Granite Embedding Models Overview

IBM Granite embedding models are a family of models designed for retrieval-based applications. They use a bi-encoder architecture to generate high-quality embeddings for textual inputs such as queries, passages, and documents, enabling semantic comparison through cosine similarity.

The Granite Embedding lineup includes:

| Model | Description | Vector Size | Language Support |
|-------|-------------|-------------|-----------------|
| `ibm-granite/granite-embedding-30m-english` | Small English model | 384 | English |
| `ibm-granite/granite-embedding-125m-english` | Large English model | 768 | English |
| `ibm-granite/granite-embedding-107m-multilingual` | Small multilingual model | 384 | 12 languages |
| `ibm-granite/granite-embedding-278m-multilingual` | Large multilingual model | 768 | 12 languages |

These models are built on carefully curated, permissibly licensed public datasets and are available under the Apache 2.0 license.

## Prerequisites

To use IBM Granite embedding models with the Qdrant MCP server, you need:

1. **LM Studio**: Version 0.2.19 or later installed on your machine
   - Download from: https://lmstudio.ai/

2. **IBM Granite Models**: Load the appropriate IBM Granite embedding model in LM Studio
   - Available on Hugging Face: https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb

## Setting Up LM Studio with IBM Granite

1. **Launch LM Studio** and go to the Local Server tab (look for the <-> icon on the left sidebar)

2. **Load an IBM Granite Embedding Model**:
   - Click on the "Embedding Model Settings" dropdown
   - Select or search for one of the IBM Granite embedding models:
     - `ibm-granite/granite-embedding-30m-english` (recommended for getting started)
     - `ibm-granite/granite-embedding-125m-english`
     - `ibm-granite/granite-embedding-107m-multilingual`
     - `ibm-granite/granite-embedding-278m-multilingual`

3. **Start the Server**:
   - Click the "Start Server" button
   - Note the URL and port (default is `http://localhost:1234`)

## Configuration for Qdrant MCP Server

To use IBM Granite embedding models via LM Studio, set the following environment variables:

| Environment Variable      | Description                                  | Example Value                               |
|---------------------------|----------------------------------------------|---------------------------------------------|
| `EMBEDDING_PROVIDER`      | Set to "lmstudio" to use LM Studio           | `lmstudio`                                  |
| `EMBEDDING_MODEL`         | The IBM Granite model name                   | `ibm-granite/granite-embedding-30m-english` |
| `EMBEDDING_VECTOR_SIZE`   | Size of embedding vectors                    | `384` (for small models)                    |
| `LMSTUDIO_API_BASE`       | Base URL of the LM Studio API                | `http://localhost:1234/v1`                  |
| `LMSTUDIO_API_KEY`        | API key for authentication (optional)        | `lm-studio`                                 |
| `COLLECTION_NAME`         | Unique collection name for IBM Granite       | `code-memory-granite`                       |

The vector size depends on the model you're using:
- Small models (30m/107m): 384 dimensions
- Large models (125m/278m): 768 dimensions

## Usage Examples

### Basic Usage with IBM Granite (Small English Model)

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory-granite" \
EMBEDDING_PROVIDER="lmstudio" \
EMBEDDING_MODEL="ibm-granite/granite-embedding-30m-english" \
EMBEDDING_VECTOR_SIZE=384 \
LMSTUDIO_API_BASE="http://localhost:1234/v1" \
uvx mcp-server-qdrant
```

### Using with IBM Granite Large English Model

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory-granite" \
EMBEDDING_PROVIDER="lmstudio" \
EMBEDDING_MODEL="ibm-granite/granite-embedding-125m-english" \
EMBEDDING_VECTOR_SIZE=768 \
LMSTUDIO_API_BASE="http://localhost:1234/v1" \
uvx mcp-server-qdrant
```

### Using with IBM Granite Multilingual Model

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory-granite" \
EMBEDDING_PROVIDER="lmstudio" \
EMBEDDING_MODEL="ibm-granite/granite-embedding-107m-multilingual" \
EMBEDDING_VECTOR_SIZE=384 \
LMSTUDIO_API_BASE="http://localhost:1234/v1" \
uvx mcp-server-qdrant
```

## Adding to Claude Code

To add the IBM Granite MCP server to Claude Code:

```bash
claude mcp add qdrant-granite \
  -e QDRANT_URL="http://localhost:6333" \
  -e COLLECTION_NAME="code-memory-granite" \
  -e EMBEDDING_PROVIDER="lmstudio" \
  -e EMBEDDING_MODEL="ibm-granite/granite-embedding-30m-english" \
  -e EMBEDDING_VECTOR_SIZE=384 \
  -e LMSTUDIO_API_BASE="http://localhost:1234/v1" \
  -e TOOL_STORE_DESCRIPTION="Store code context and business information using IBM Granite embeddings. Use metadata.domain to categorize: 'code', 'architecture', 'business-logic', 'requirements'. Include key points in metadata.summary." \
  -e TOOL_FIND_DESCRIPTION="Search for information using IBM Granite's embedding model. Great for precise business and code queries. Be specific in your search to get the most relevant results." \
  -- uvx mcp-server-qdrant
```

## Using the Makefile

The project includes Makefile commands to simplify working with IBM Granite:

```bash
# Start LM Studio and load the IBM Granite model first

# Check if IBM Granite model is available in LM Studio
make check-granite

# Start Qdrant with IBM Granite MCP server
make start-granite

# Run IBM Granite MCP server locally
make run-granite

# Add IBM Granite MCP server to Claude Code
make add-granite-to-claude
```

You can customize the IBM Granite model by setting environment variables:

```bash
# Use the large English model
make add-granite-to-claude GRANITE_MODEL=ibm-granite/granite-embedding-125m-english GRANITE_VECTOR_SIZE=768

# Use the multilingual model
make add-granite-to-claude GRANITE_MODEL=ibm-granite/granite-embedding-107m-multilingual GRANITE_VECTOR_SIZE=384
```

## When to Use IBM Granite Models

IBM Granite embedding models are particularly well-suited for:

1. **Enterprise Applications**: Built to meet enterprise-grade expectations with transparency and reliability
2. **Business Domain Information**: Particularly strong for business and technical contexts
3. **Code Understanding**: Great for storing and retrieving code components, architecture information, and technical documentation
4. **Multilingual Support**: The multilingual variants support 12 different languages, making them ideal for international teams and documentation

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure the model is properly loaded in LM Studio
   - Check that the model is visible in the Embedding Model dropdown
   - Verify the exact model name matches what's in the configuration

2. **Vector Size Mismatch**: Make sure the vector size matches the model
   - 384 for small/multilingual models
   - 768 for large models

3. **Collection Conflicts**: If you've used different models with different vector sizes, you may need separate collections
   - Each collection can only have one vector size
   - Create separate collections for different model families

To check if LM Studio recognizes the IBM Granite model:

```bash
curl -X POST "http://localhost:1234/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"ibm-granite/granite-embedding-30m-english\",\"input\":\"test\"}"
```

## Technical Details

The IBM Granite integration works by:

1. Using LM Studio as the embedding model server
2. Connecting to LM Studio's OpenAI-compatible API
3. Automatically detecting the appropriate vector size based on the model name
4. Using a separate collection in Qdrant specifically for IBM Granite embeddings

This approach combines the ease of use of LM Studio with the performance of IBM's enterprise-grade embedding models.