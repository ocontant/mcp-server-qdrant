import numpy as np
import pytest
import os
from unittest.mock import patch, AsyncMock

from mcp_server_qdrant.embeddings.gguf import GGUFEmbeddingProvider
from mcp_server_qdrant.settings import EmbeddingProviderSettings


@pytest.mark.asyncio
class TestGGUFProviderIntegration:
    """Integration tests for GGUFEmbeddingProvider."""

    @pytest.fixture
    def mock_settings(self):
        settings = EmbeddingProviderSettings(
            provider_type="gguf",
            model_name="/path/to/model.gguf",
            vector_size=384,
            max_context_length=2048,
        )
        return settings

    @pytest.fixture
    def mock_process(self):
        """Mock subprocess for GGUF embedding generation."""
        with patch("mcp_server_qdrant.embeddings.gguf.asyncio.create_subprocess_exec") as mock_exec:
            process_mock = AsyncMock()
            process_mock.communicate.return_value = (
                '[0.01, 0.02, 0.03, 0.04]\n[0.05, 0.06, 0.07, 0.08]'.encode(),
                b''
            )
            process_mock.returncode = 0
            mock_exec.return_value = process_mock
            yield mock_exec

    async def test_initialization(self, mock_settings):
        """Test that the provider can be initialized with a valid model."""
        provider = GGUFEmbeddingProvider(
            model_path=mock_settings.model_name, 
            vector_size=mock_settings.vector_size,
            max_context_length=mock_settings.max_context_length
        )
        assert provider.model_path == mock_settings.model_name
        assert provider.vector_size == mock_settings.vector_size
        assert provider.max_context_length == mock_settings.max_context_length

    async def test_embed_documents(self, mock_settings, mock_process):
        """Test that documents can be embedded."""
        provider = GGUFEmbeddingProvider(
            model_path=mock_settings.model_name, 
            vector_size=mock_settings.vector_size,
            max_context_length=mock_settings.max_context_length
        )
        documents = ["This is a test document.", "This is another test document."]

        embeddings = await provider.embed_documents(documents)

        # Check that we got the right number of embeddings
        assert len(embeddings) == len(documents)
        
        # With our mocked response, we expect 4-dimensional vectors
        assert len(embeddings[0]) == 4
        assert len(embeddings[1]) == 4

        # Check that embeddings are different for different documents
        embedding1 = np.array(embeddings[0])
        embedding2 = np.array(embeddings[1])
        assert not np.array_equal(embedding1, embedding2)

    async def test_embed_query(self, mock_settings, mock_process):
        """Test that queries can be embedded."""
        provider = GGUFEmbeddingProvider(
            model_path=mock_settings.model_name, 
            vector_size=mock_settings.vector_size,
            max_context_length=mock_settings.max_context_length
        )
        query = "This is a test query."

        embedding = await provider.embed_query(query)

        # With our mocked response, we expect a 4-dimensional vector
        assert len(embedding) == 4

    async def test_get_vector_name(self, mock_settings):
        """Test that the vector name is generated correctly."""
        provider = GGUFEmbeddingProvider(
            model_path="/path/to/nomic-embed-text-v1.5.f16.gguf", 
            vector_size=mock_settings.vector_size,
            max_context_length=mock_settings.max_context_length
        )
        vector_name = provider.get_vector_name()

        # Check that the vector name follows the expected format
        assert vector_name.startswith("gguf-")
        assert "nomic-embed" in vector_name