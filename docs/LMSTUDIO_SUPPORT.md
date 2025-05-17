# Using LM Studio API with Qdrant MCP Server

This document explains how to use LM Studio's API for embedding generation with the Qdrant MCP server. This allows you to leverage LM Studio's local server capabilities for fast embedding generation.

## Prerequisites

To use LM Studio for embedding generation, you need:

1. **LM Studio**: Version 0.2.19 or later installed on your machine
   - Download from: https://lmstudio.ai/
   - Or beta versions from: https://lmstudio.ai/beta-releases.html

2. **Embedding Model**: A text embedding model loaded in LM Studio
   - LM Studio supports various embedding models like:
     - `nomic-ai/nomic-embed-text-v1.5`
     - `sentence-transformers/all-MiniLM-L6-v2`
     - Snowflake Arctic models
     - And many others

## Setting Up LM Studio

1. **Launch LM Studio** and go to the Local Server tab (look for the <-> icon on the left sidebar)

2. **Load an Embedding Model**:
   - Click on the "Embedding Model Settings" dropdown
   - Select your preferred embedding model (like `nomic-ai/nomic-embed-text-v1.5`)

3. **Start the Server**:
   - Click the "Start Server" button
   - Note the URL and port (default is `http://localhost:1234`)

## Configuration for Qdrant MCP Server

To use LM Studio for embeddings, set the following environment variables:

| Environment Variable      | Description                                  | Example Value                              |
|---------------------------|----------------------------------------------|-------------------------------------------|
| `EMBEDDING_PROVIDER`      | Set to "lmstudio" to use LM Studio           | `lmstudio`                                |
| `EMBEDDING_MODEL`         | Model name loaded in LM Studio               | `nomic-ai/nomic-embed-text-v1.5`           |
| `EMBEDDING_VECTOR_SIZE`   | Size of embedding vectors (optional)         | `1024` (for nomic-embed-text-v1.5)         |
| `LMSTUDIO_API_BASE`       | Base URL of the LM Studio API                | `http://localhost:1234/v1`                 |
| `LMSTUDIO_API_KEY`        | API key for authentication (usually optional)| `lm-studio`                               |

### Vector Size Reference

If `EMBEDDING_VECTOR_SIZE` is not provided, the server will try to detect the correct size based on the model name:

| Model Family                    | Vector Size |
|---------------------------------|-------------|
| nomic-embed-text-v1.5           | 1024        |
| all-MiniLM-L6-v2                | 384         |
| snowflake-arctic-embed-l-*      | 768         |
| snowflake-arctic-embed-m-*      | 384         |
| Default (if unknown)            | 768         |

## Usage Examples

### Basic Usage with Nomic Embed

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="lmstudio" \
EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5" \
LMSTUDIO_API_BASE="http://localhost:1234/v1" \
uvx mcp-server-qdrant
```

### Using with Snowflake Arctic Models

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="lmstudio" \
EMBEDDING_MODEL="Snowflake/snowflake-arctic-embed-l-v2.0" \
EMBEDDING_VECTOR_SIZE=768 \
LMSTUDIO_API_BASE="http://localhost:1234/v1" \
uvx mcp-server-qdrant
```

### Using with a Remote LM Studio Server

If LM Studio is running on another machine:

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="lmstudio" \
EMBEDDING_MODEL="all-MiniLM-L6-v2" \
EMBEDDING_VECTOR_SIZE=384 \
LMSTUDIO_API_BASE="http://192.168.1.100:1234/v1" \
uvx mcp-server-qdrant
```

## Adding to Claude Code

To add this MCP server with LM Studio support to Claude Code:

```bash
claude mcp add qdrant-lmstudio \
  -e QDRANT_URL="http://localhost:6333" \
  -e COLLECTION_NAME="code-memory" \
  -e EMBEDDING_PROVIDER="lmstudio" \
  -e EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5" \
  -e LMSTUDIO_API_BASE="http://localhost:1234/v1" \
  -- uvx mcp-server-qdrant
```

## Docker Configuration

When using Docker, you'll need to ensure the container can reach the LM Studio server. If LM Studio is running on the host:

```yaml
services:
  qdrant-mcp:
    # ... other configuration ...
    environment:
      - QDRANT_URL=http://qdrant:6333
      - COLLECTION_NAME=code-memory
      - EMBEDDING_PROVIDER=lmstudio
      - EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
      - LMSTUDIO_API_BASE=http://host.docker.internal:1234/v1
    # ... other configuration ...
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure LM Studio server is running and the port is correct
   - Check the server status in LM Studio's interface
   - Verify the `LMSTUDIO_API_BASE` URL is correct

2. **Model Not Found**: Ensure the model name matches exactly what's loaded in LM Studio
   - LM Studio logs will show the exact model name it's using

3. **API Errors**: Verify that the embedding endpoint is available
   - Try a simple test with curl:
     ```bash
     curl -X POST http://localhost:1234/v1/embeddings \
       -H "Content-Type: application/json" \
       -d '{"model":"your-model-name","input":"test"}'
     ```

4. **Incorrect Vector Size**: If you're getting index errors, verify that the `EMBEDDING_VECTOR_SIZE` matches what the model actually produces

## Technical Details

The Qdrant MCP server connects to LM Studio through its OpenAI-compatible API. This approach:

1. Allows you to use any embedding model supported by LM Studio
2. Takes advantage of LM Studio's optimized inference for embedding generation
3. Simplifies model management (no need to download models separately)

The LM Studio provider in Qdrant MCP server:
- Makes API calls to the LM Studio server
- Automatically detects vector sizes when possible
- Handles batching of document embeddings for better performance