import asyncio
import json
import aiohttp
from typing import List, Dict, Any, Optional
import logging

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

class LMStudioEmbeddingProvider(EmbeddingProvider):
    """
    LM Studio API implementation of the embedding provider.
    
    :param model_name: The name of the embedding model to use
    :param api_base: The base URL of the LM Studio API (default: http://localhost:1234/v1)
    :param api_key: The API key to use for authentication (default: lm-studio)
    :param vector_size: The size of the embeddings produced by the model
    """

    def __init__(
        self, 
        model_name: str,
        api_base: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        vector_size: Optional[int] = None
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        
        # Set vector size if provided, otherwise will be determined on first call
        self._vector_size = vector_size
        
        logger.info(f"Initialized LM Studio embedding provider with model: {model_name}")
        logger.info(f"API Base: {api_base}")

    async def _get_embeddings_from_api(self, texts: List[str]) -> List[List[float]]:
        """Make API call to LM Studio to get embeddings."""
        if not texts:
            return []
            
        # Default headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        embeddings_endpoint = f"{self.api_base}/embeddings"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    embeddings_endpoint, 
                    headers=headers, 
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"LM Studio API error: {response.status} - {error_text}")
                        raise RuntimeError(f"LM Studio API error: {response.status} - {error_text}")
                    
                    response_data = await response.json()
                    
                    # Extract embeddings from response
                    embeddings = []
                    for item in response_data.get("data", []):
                        embedding = item.get("embedding", [])
                        embeddings.append(embedding)
                        
                        # If vector size wasn't provided, determine it from the first embedding
                        if self._vector_size is None and embedding:
                            self._vector_size = len(embedding)
                            logger.info(f"Detected embedding vector size: {self._vector_size}")
                    
                    return embeddings
                    
        except Exception as e:
            logger.error(f"Error calling LM Studio API: {str(e)}")
            raise

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        return await self._get_embeddings_from_api(documents)

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        embeddings = await self._get_embeddings_from_api([query])
        if not embeddings:
            raise RuntimeError("Failed to get embedding for query")
        return embeddings[0]

    def get_vector_name(self) -> str:
        """Get the name of the vector for the Qdrant collection."""
        # Clean up model name to use as part of vector name
        # Replace any characters that might not be valid in Qdrant vector names
        cleaned_model_name = self.model_name.replace("/", "-").replace(".", "-").lower()
        return f"lms-{cleaned_model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self._vector_size is None:
            # If vector size isn't set, try to detect from model name
            if "nomic-embed-text-v1.5" in self.model_name:
                self._vector_size = 1024
            elif "all-minilm-l6-v2" in self.model_name.lower():
                self._vector_size = 384
            elif "snowflake-arctic-embed" in self.model_name:
                # Check if it's a large or medium model
                if "-l-" in self.model_name:
                    self._vector_size = 768
                else:
                    self._vector_size = 384
            else:
                # Default to 768 dimensions if we couldn't determine
                self._vector_size = 768
                logger.warning(f"Could not determine vector size for model {self.model_name}, defaulting to {self._vector_size}")
                
        return self._vector_size