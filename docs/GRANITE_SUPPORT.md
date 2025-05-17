# Using IBM Granite Embedding Models with Qdrant MCP Server

This document explains how to use IBM Granite embedding models with the Qdrant MCP server. IBM Granite models are high-quality embedding models that can provide excellent semantic search capabilities for your applications.

There are two ways to use IBM Granite models with the Qdrant MCP server:
1. **Direct Provider**: Using the built-in Granite provider (recommended for best performance)
2. **LM Studio Integration**: Through LM Studio's API (easier setup if you already use LM Studio)

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

## Using the Direct Granite Provider

### Prerequisites

To use IBM Granite embedding models directly with the Qdrant MCP server, you need:

1. **Python Dependencies**: The server will automatically install the required dependencies
   - Transformers
   - PyTorch (CPU version is sufficient)

2. **IBM Granite Models**: The models will be downloaded automatically from Hugging Face
   - Available models: https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb
   - Models are cached locally in the Hugging Face cache directory

### Configuration for Direct Granite Provider

To use the direct Granite provider, set the following environment variables:

| Environment Variable      | Description                                  | Example Value                               |
|---------------------------|----------------------------------------------|---------------------------------------------|
| `EMBEDDING_PROVIDER`      | Set to "granite" to use the direct provider   | `granite`                                   |
| `EMBEDDING_MODEL`         | The IBM Granite model name                   | `IBM/granite-13b-embeddings`                |
| `EMBEDDING_VECTOR_SIZE`   | Size of embedding vectors (optional)         | `1024` (auto-detected for known models)     |
| `GRANITE_DEVICE`          | Device to run the model on (optional)        | `cpu` (default) or `cuda:0` for GPU         |
| `GRANITE_NORMALIZE_EMBEDDINGS` | Whether to normalize vectors (optional) | `true` (default)                           |
| `GRANITE_MAX_LENGTH`      | Maximum token length (optional)              | `512` (default)                             |

The vector size depends on the model you're using:
- Small models (30m/107m): 384 dimensions
- Large models (125m/278m): 768 dimensions
- Larger models: 1024 dimensions

### Usage Examples for Direct Provider

#### Basic Usage with IBM Granite

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="granite" \
EMBEDDING_MODEL="IBM/granite-13b-embeddings" \
uvx mcp-server-qdrant
```

#### Using with a Specific Device

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="granite" \
EMBEDDING_MODEL="IBM/granite-13b-embeddings" \
GRANITE_DEVICE="cuda:0" \
uvx mcp-server-qdrant
```

#### Using the Small English Model

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-memory" \
EMBEDDING_PROVIDER="granite" \
EMBEDDING_MODEL="IBM/granite-embedding-30m-english" \
uvx mcp-server-qdrant
```

## Using LM Studio with IBM Granite

### Prerequisites for LM Studio Method

To use IBM Granite embedding models via LM Studio, you need:

1. **LM Studio**: Version 0.2.19 or later installed on your machine
   - Download from: https://lmstudio.ai/

2. **IBM Granite Models**: Load the appropriate IBM Granite embedding model in LM Studio
   - Available on Hugging Face: https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb

### Setting Up LM Studio with IBM Granite

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

### Adding Direct Granite Provider to Claude Code (Recommended)

To add the IBM Granite MCP server with the direct provider to Claude Code:

```bash
claude mcp add qdrant-granite \
  -e QDRANT_URL="http://localhost:6333" \
  -e COLLECTION_NAME="code-memory-granite" \
  -e EMBEDDING_PROVIDER="granite" \
  -e EMBEDDING_MODEL="IBM/granite-13b-embeddings" \
  -e TOOL_STORE_DESCRIPTION="Store code context and business information using IBM Granite embeddings. Use metadata.domain to categorize: 'code', 'architecture', 'business-logic', 'requirements'. Include key points in metadata.summary." \
  -e TOOL_FIND_DESCRIPTION="Search for information using IBM Granite's embedding model. Great for precise business and code queries. Be specific in your search to get the most relevant results." \
  -- uvx mcp-server-qdrant
```

### Adding LM Studio-based Granite to Claude Code

To add the IBM Granite MCP server using LM Studio:

```bash
claude mcp add qdrant-granite-lms \
  -e QDRANT_URL="http://localhost:6333" \
  -e COLLECTION_NAME="code-memory-granite" \
  -e EMBEDDING_PROVIDER="lmstudio" \
  -e EMBEDDING_MODEL="ibm-granite/granite-embedding-30m-english" \
  -e EMBEDDING_VECTOR_SIZE=384 \
  -e LMSTUDIO_API_BASE="http://localhost:11433/v1" \
  -e TOOL_STORE_DESCRIPTION="Store code context and business information using IBM Granite embeddings. Use metadata.domain to categorize: 'code', 'architecture', 'business-logic', 'requirements'. Include key points in metadata.summary." \
  -e TOOL_FIND_DESCRIPTION="Search for information using IBM Granite's embedding model. Great for precise business and code queries. Be specific in your search to get the most relevant results." \
  -- uvx mcp-server-qdrant
```

## Using the Makefile

The project includes Makefile commands to simplify working with IBM Granite:

### Direct Provider Commands

```bash
# Start Qdrant with IBM Granite direct provider MCP server
make start-granite-direct

# Run IBM Granite direct provider MCP server locally
make run-granite-direct

# Add IBM Granite direct provider MCP server to Claude Code
make add-granite-direct-to-claude
```

You can customize the IBM Granite model by setting environment variables:

```bash
# Use a specific IBM Granite model
make add-granite-direct-to-claude GRANITE_MODEL=IBM/granite-embedding-30m-english

# Use GPU acceleration if available
make run-granite-direct GRANITE_DEVICE=cuda:0
```

### LM Studio Provider Commands

```bash
# Start LM Studio and load the IBM Granite model first

# Check if IBM Granite model is available in LM Studio
make check-granite-lms

# Start Qdrant with IBM Granite LM Studio MCP server
make start-granite-lms

# Run IBM Granite LM Studio MCP server locally
make run-granite-lms

# Add IBM Granite LM Studio MCP server to Claude Code
make add-granite-lms-to-claude
```

You can customize the IBM Granite model for LM Studio by setting environment variables:

```bash
# Use the large English model
make add-granite-lms-to-claude GRANITE_MODEL=ibm-granite/granite-embedding-125m-english GRANITE_VECTOR_SIZE=768

# Use the multilingual model
make add-granite-lms-to-claude GRANITE_MODEL=ibm-granite/granite-embedding-107m-multilingual GRANITE_VECTOR_SIZE=384
```

## When to Use IBM Granite Models

IBM Granite embedding models are particularly well-suited for:

1. **Enterprise Applications**: Built to meet enterprise-grade expectations with transparency and reliability
2. **Business Domain Information**: Particularly strong for business and technical contexts
3. **Code Understanding**: Great for storing and retrieving code components, architecture information, and technical documentation
4. **Multilingual Support**: The multilingual variants support 12 different languages, making them ideal for international teams and documentation

## Troubleshooting

### Common Issues

#### Direct Provider Issues

1. **Dependency Installation Errors**: If you encounter errors related to missing dependencies:
   - Ensure you have an internet connection for the automatic installation
   - Try manually installing with `pip install transformers torch`
   - Check if you have sufficient disk space for model downloads

2. **CUDA/GPU Issues**: If you're trying to use GPU acceleration:
   - Make sure you have CUDA installed correctly
   - Try falling back to CPU with `GRANITE_DEVICE=cpu`
   - Check CUDA compatibility with your PyTorch version

3. **Model Download Problems**: The first time you use a model, it will be downloaded:
   - This may take some time depending on your internet connection
   - Ensure you have sufficient disk space (models can be 1-2GB)
   - Check the Hugging Face cache at `~/.cache/huggingface/hub/`

#### LM Studio Issues

1. **Model Not Found**: Ensure the model is properly loaded in LM Studio
   - Check that the model is visible in the Embedding Model dropdown
   - Verify the exact model name matches what's in the configuration

2. **Connection Issues**: If you can't connect to LM Studio:
   - Make sure LM Studio is running and the server is started
   - Check that the port in `LMSTUDIO_API_BASE` matches LM Studio's configuration
   - Try a basic curl test to verify the API is accessible

To check if LM Studio recognizes the IBM Granite model:

```bash
curl -X POST "http://localhost:11433/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"ibm-granite/granite-embedding-30m-english\",\"input\":\"test\"}"
```

#### Common Issues for Both Methods

1. **Vector Size Mismatch**: Make sure the vector size matches the model
   - 384 for small/multilingual models
   - 768 for large models
   - 1024 for the largest models

2. **Collection Conflicts**: If you've used different models with different vector sizes, you may need separate collections
   - Each collection can only have one vector size
   - Create separate collections for different model families

3. **Memory Limitations**: Large models require more memory
   - For direct provider: Try reducing `GRANITE_MAX_LENGTH` to process smaller chunks
   - For LM Studio: Adjust batch size settings in LM Studio's interface

## Technical Details

### Direct Provider Technical Details

The direct Granite provider:

1. Uses the Hugging Face Transformers library to load and run IBM Granite models
2. Processes text directly on your local machine (CPU or GPU)
3. Performs mean pooling on the model outputs to generate embeddings
4. Automatically detects the appropriate vector size based on the model name

This approach provides:
- Better performance (especially with GPU)
- Full control over model parameters
- No dependency on external services

### LM Studio Provider Technical Details

The LM Studio Granite integration works by:

1. Using LM Studio as the embedding model server
2. Connecting to LM Studio's OpenAI-compatible API
3. Automatically detecting the appropriate vector size based on the model name
4. Using a separate collection in Qdrant specifically for IBM Granite embeddings

This approach provides:
- Easier setup if you already use LM Studio
- LM Studio's optimized inference pipeline
- Consistent UI for managing embedding models