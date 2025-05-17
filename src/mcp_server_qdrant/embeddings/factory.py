import os
from typing import Dict, Any, Optional
import re
import logging

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings

logger = logging.getLogger(__name__)

def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
        return FastEmbedProvider(settings.model_name)
    
    elif settings.provider_type == EmbeddingProviderType.GGUF:
        from mcp_server_qdrant.embeddings.gguf import GGUFEmbeddingProvider
        
        # Extract vector size from model path if available
        vector_size = settings.vector_size
        
        # If vector_size isn't set, try to detect from file name
        if vector_size is None and settings.model_name:
            model_path = settings.model_name
            
            # Try to detect vector size from filename for known models
            if "nomic-embed-text-v1.5" in model_path:
                vector_size = 1024
            elif "all-MiniLM-L6-v2" in model_path:
                vector_size = 384
            elif "snowflake-arctic-embed" in model_path:
                # Check if it's a large or medium model
                if "-l-" in model_path:
                    vector_size = 768
                else:
                    vector_size = 384
            
        # Default to 768 dimensions if we couldn't determine
        if vector_size is None:
            vector_size = 768
            
        return GGUFEmbeddingProvider(
            model_path=settings.model_name,
            vector_size=vector_size,
            max_context_length=settings.max_context_length,
            llama_cpp_path=settings.llama_cpp_path
        )
    
    elif settings.provider_type == EmbeddingProviderType.LMSTUDIO:
        from mcp_server_qdrant.embeddings.lmstudio import LMStudioEmbeddingProvider
        
        # Extract vector size from model path if available
        vector_size = settings.vector_size
        
        # If vector_size isn't set, try to detect from model name
        if vector_size is None and settings.model_name:
            model_name = settings.model_name.lower()
            
            # IBM Granite model vector size detection
            if "ibm-granite" in model_name or "granite-embedding" in model_name:
                logger.info("Detected IBM Granite embedding model")
                if "30m" in model_name or "107m" in model_name:
                    vector_size = 384
                    logger.info("Using vector size 384 for IBM Granite small/multilingual model")
                elif "125m" in model_name or "278m" in model_name:
                    vector_size = 768
                    logger.info("Using vector size 768 for IBM Granite large model")
                else:
                    # Default IBM Granite vector size
                    vector_size = 768
                    logger.info(f"Could not determine IBM Granite model size, defaulting to vector size {vector_size}")
            
            # Other common models
            elif "nomic-embed-text-v1.5" in model_name:
                vector_size = 1024
            elif "all-minilm-l6-v2" in model_name:
                vector_size = 384
            elif "snowflake-arctic-embed" in model_name:
                # Check if it's a large or medium model
                if "-l-" in model_name:
                    vector_size = 768
                else:
                    vector_size = 384
            elif "e5-" in model_name:
                if "small" in model_name:
                    vector_size = 384
                elif "base" in model_name:
                    vector_size = 768
                elif "large" in model_name:
                    vector_size = 1024
            elif "bge-" in model_name:
                if "small" in model_name:
                    vector_size = 384
                elif "base" in model_name:
                    vector_size = 768
                elif "large" in model_name:
                    vector_size = 1024
        
        # Default to 768 dimensions if we couldn't determine
        if vector_size is None:
            vector_size = 768
            logger.warning(f"Could not determine vector size for model {settings.model_name}, defaulting to {vector_size}")
            
        return LMStudioEmbeddingProvider(
            model_name=settings.model_name,
            api_base=settings.lmstudio_api_base,
            api_key=settings.lmstudio_api_key,
            vector_size=vector_size
        )
    
    elif settings.provider_type == EmbeddingProviderType.GRANITE:
        from mcp_server_qdrant.embeddings.granite import GraniteEmbeddingProvider
        
        return GraniteEmbeddingProvider(
            model_name=settings.model_name,
            device=settings.granite_device,
            normalize_embeddings=settings.granite_normalize_embeddings,
            max_length=settings.granite_max_length
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")