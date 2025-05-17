import asyncio
import os
import subprocess
import json
import tempfile
from typing import List, Dict, Any, Optional
import logging

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

class GGUFEmbeddingProvider(EmbeddingProvider):
    """
    GGUF implementation of the embedding provider using llama.cpp.
    
    :param model_path: Path to the GGUF model file
    :param vector_size: The size of the embeddings produced by the model
    :param max_context_length: Maximum context length for the model
    :param llama_cpp_path: Path to the llama-cpp embedding binary
    """

    def __init__(
        self, 
        model_path: str, 
        vector_size: int = 768,
        max_context_length: int = 2048,
        llama_cpp_path: Optional[str] = None
    ):
        self.model_path = os.path.expanduser(model_path)
        self.vector_size = vector_size
        self.max_context_length = max_context_length
        
        # Find llama-cpp embedding binary
        if llama_cpp_path:
            self.llama_cpp_path = llama_cpp_path
        else:
            # Try to find the binary in PATH
            try:
                result = subprocess.run(["which", "embedding"], capture_output=True, text=True)
                if result.returncode == 0:
                    self.llama_cpp_path = result.stdout.strip()
                else:
                    # Default to a common install location
                    self.llama_cpp_path = "/usr/local/bin/embedding"
            except Exception as e:
                logger.warning(f"Could not find llama.cpp embedding binary: {e}")
                self.llama_cpp_path = "embedding"  # Hope it's in PATH
        
        # Validate that the model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GGUF model file not found: {self.model_path}")
        
        # Extract model name for vector naming
        self.model_name = os.path.basename(self.model_path).split('.')[0]
        
        logger.info(f"Initialized GGUF embedding provider with model: {self.model_path}")
        logger.info(f"Using llama.cpp binary: {self.llama_cpp_path}")

    async def _run_embedding_process(self, input_text: str) -> List[float]:
        """Run the embedding process and return the resulting vector."""
        # Create a temporary file to store the input text
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write(input_text)
            temp_filename = temp.name
        
        try:
            # Build the command with proper parameters
            cmd = [
                self.llama_cpp_path,
                "-m", self.model_path,
                "-c", str(self.max_context_length),
                "-f", temp_filename,
                "--embedding-only"
            ]
            
            # Run the command
            logger.debug(f"Running command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"Error running embedding process: {error_msg}")
                raise RuntimeError(f"Embedding process failed: {error_msg}")
            
            # Parse the output to get the embeddings
            output = stdout.decode().strip()
            try:
                # The output should be a JSON array of floats
                embedding_vector = json.loads(output)
                if not isinstance(embedding_vector, list):
                    raise ValueError("Expected embedding output to be a list")
                return embedding_vector
            except json.JSONDecodeError:
                logger.error(f"Failed to parse embedding output: {output}")
                raise RuntimeError("Failed to parse embedding output")
        
        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        results = []
        
        # Process each document sequentially
        # You could implement batching here for better performance
        for doc in documents:
            # Prefix with 'passage:' as recommended for some embedding models
            prefixed_doc = f"passage: {doc}"
            embedding = await self._run_embedding_process(prefixed_doc)
            results.append(embedding)
        
        return results

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        # Prefix with 'query:' as recommended for some embedding models
        prefixed_query = f"query: {query}"
        return await self._run_embedding_process(prefixed_query)

    def get_vector_name(self) -> str:
        """Get the name of the vector for the Qdrant collection."""
        return f"gguf-{self.model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.vector_size