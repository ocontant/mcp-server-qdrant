import asyncio
import torch
import logging
from typing import List, Dict, Any, Optional

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

class GraniteEmbeddingProvider(EmbeddingProvider):
    """
    IBM Granite implementation of the embedding provider.
    
    :param model_name: The name of the IBM Granite model to use
    :param device: The device to use for inference (cpu, cuda, mps)
    :param normalize_embeddings: Whether to normalize embeddings
    :param max_length: Maximum token length for the model
    """

    def __init__(
        self, 
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length
        
        # Lazy loading of model and tokenizer
        self._model = None
        self._tokenizer = None
        self._is_initialized = False
        
        # Extract vector name for vector naming
        self.vector_name = model_name.split("/")[-1].lower()
        
        logger.info(f"Initialized IBM Granite embedding provider with model: {model_name}")
        logger.info(f"Device: {device}, Normalize: {normalize_embeddings}")
        
        # Determine vector size from model name
        if "30m" in model_name or "107m" in model_name:
            self._vector_size = 384
        elif "125m" in model_name or "278m" in model_name:
            self._vector_size = 768
        else:
            # Default to 768 if unknown
            self._vector_size = 768
            logger.warning(f"Could not determine vector size from model name: {model_name}. Defaulting to {self._vector_size}")

    def _ensure_initialized(self):
        """Ensure the model and tokenizer are initialized."""
        if not self._is_initialized:
            try:
                from transformers import AutoTokenizer, AutoModel
                
                logger.info(f"Loading model and tokenizer for {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                
                # Move model to specified device
                self._model.to(self.device)
                self._model.eval()  # Set model to evaluation mode
                
                self._is_initialized = True
                logger.info(f"Successfully loaded model and tokenizer for {self.model_name}")
            
            except Exception as e:
                logger.error(f"Error initializing IBM Granite model: {e}")
                raise RuntimeError(f"Failed to initialize IBM Granite embeddings: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._ensure_initialized()
        
        # Tokenize input text
        inputs = self._tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Mean pooling to get sentence embedding
        attention_mask = inputs["attention_mask"]
        last_hidden_state = outputs.last_hidden_state
        
        # Apply attention mask for proper mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings if required
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to list and return
        return embeddings[0].cpu().tolist()

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        loop = asyncio.get_event_loop()
        
        # Process each document in a thread pool to prevent blocking the event loop
        embeddings = []
        for doc in documents:
            embedding = await loop.run_in_executor(None, lambda: self._get_embedding(doc))
            embeddings.append(embedding)
        
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, lambda: self._get_embedding(query))
        return embedding

    def get_vector_name(self) -> str:
        """Get the name of the vector for the Qdrant collection."""
        return f"granite-{self.vector_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self._vector_size