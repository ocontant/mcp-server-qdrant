# Using GGUF Embedding Models with Qdrant MCP Server

This document explains how to use GGUF-format embedding models with the Qdrant MCP server. This feature allows you to use quantized embedding models like those from Nomic AI and other providers that have been converted to GGUF format.

## Prerequisites

To use GGUF embedding models, you need:

1. **llama.cpp**: The server uses the `embedding` binary from llama.cpp to generate embeddings
   - Install from: https://github.com/ggerganov/llama.cpp
   - Make sure the `embedding` binary is in your PATH

2. **GGUF Model Files**: Download the GGUF-format embedding models you want to use
   - Common sources include Hugging Face (nomic-ai/nomic-embed-text-v1.5-GGUF)
   - You can convert models to GGUF using llama.cpp tools

## Configuration

To use GGUF models, set the following environment variables:

| Environment Variable            | Description                                  | Example Value                                        |
|---------------------------------|----------------------------------------------|------------------------------------------------------|
| `EMBEDDING_PROVIDER`            | Set to "gguf" to use GGUF models            | `gguf`                                               |
| `EMBEDDING_MODEL`               | Path to the GGUF model file                  | `/path/to/nomic-embed-text-v1.5-Q8_0.gguf`           |
| `EMBEDDING_VECTOR_SIZE`         | Size of embedding vectors (optional)         | `1024` (for nomic-embed-text-v1.5)                   |
| `EMBEDDING_MAX_CONTEXT_LENGTH`  | Maximum context length (optional)            | `2048` (default) or `8192` (for extended context)    |
| `LLAMA_CPP_PATH`                | Path to llama.cpp embedding binary (optional)| `/usr/local/bin/embedding`                           |

### Vector Size Reference

If `EMBEDDING_VECTOR_SIZE` is not provided, the server will try to detect the correct size based on the model filename:

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
EMBEDDING_PROVIDER="gguf" \
EMBEDDING_MODEL="/path/to/nomic-embed-text-v1.5-Q8_0.gguf" \
EMBEDDING_VECTOR_SIZE=1024 \
uvx mcp-server-qdrant
```

### Using with All-MiniLM GGUF Model

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="gguf" \
EMBEDDING_MODEL="/path/to/all-MiniLM-L6-v2-Q8_0.gguf" \
EMBEDDING_VECTOR_SIZE=384 \
uvx mcp-server-qdrant
```

### Using with Snowflake Arctic Models

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="gguf" \
EMBEDDING_MODEL="/path/to/snowflake-arctic-embed-l-v2.0-8bit.gguf" \
EMBEDDING_VECTOR_SIZE=768 \
uvx mcp-server-qdrant
```

### Custom Llama.cpp Path

If the `embedding` binary is not in your PATH:

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="gguf" \
EMBEDDING_MODEL="/path/to/model.gguf" \
LLAMA_CPP_PATH="/custom/path/to/embedding" \
uvx mcp-server-qdrant
```

## Adding to Claude Code

To add this MCP server with GGUF support to Claude Code:

```bash
claude mcp add qdrant-gguf \
  -e QDRANT_URL="http://localhost:6333" \
  -e COLLECTION_NAME="code-memory" \
  -e EMBEDDING_PROVIDER="gguf" \
  -e EMBEDDING_MODEL="/path/to/nomic-embed-text-v1.5-Q8_0.gguf" \
  -e EMBEDDING_VECTOR_SIZE=1024 \
  -- uvx mcp-server-qdrant
```

## Troubleshooting

### Common Issues

1. **Model Not Found**: Make sure the path to the GGUF model file is correct and accessible

2. **Embedding Binary Not Found**: Make sure the llama.cpp `embedding` binary is in your PATH or specify it with `LLAMA_CPP_PATH`

3. **Incorrect Vector Size**: If you're getting index errors, verify that the `EMBEDDING_VECTOR_SIZE` matches what the model actually produces

4. **Performance Issues**: GGUF models can be slower than optimized ONNX models. Use the most appropriate quantization level for your needs (Q4_0 is faster but less accurate than Q8_0)

### Checking Installation

To verify your llama.cpp installation:

```bash
which embedding
embedding -h
```

Make sure it supports the `--embedding-only` flag.

## Technical Details

The MCP server launches the llama.cpp `embedding` binary as a subprocess to generate embeddings. This approach allows for:

1. Support for different quantization levels (Q4_0, Q8_0, etc.)
2. Compatibility with any model converted to GGUF format
3. Lower memory usage due to quantization

However, this approach is generally slower than using optimized libraries like FastEmbed's ONNX Runtime integration for non-quantized models.

## Security Considerations

When using GGUF models, be aware that the MCP server:

1. Creates temporary files containing the text to embed
2. Executes a subprocess (the `embedding` binary)

Ensure that proper file permissions and security measures are in place, especially if running in a multi-user environment.