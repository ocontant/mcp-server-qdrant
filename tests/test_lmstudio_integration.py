import numpy as np
import pytest
from unittest.mock import patch, AsyncMock

from mcp_server_qdrant.embeddings.lmstudio import LMStudioProvider
from mcp_server_qdrant.settings import Settings


@pytest.mark.asyncio
class TestLMStudioProviderIntegration:
    """Integration tests for LMStudioProvider."""

    @pytest.fixture
    def mock_settings(self):
        settings = Settings(
            embedding_provider="lmstudio",
            embedding_model="nomic-embed-text-v1.5",
            vector_size=768,
            lmstudio_api_base="http://localhost:1234/v1",
        )
        return settings

    @pytest.fixture
    def mock_response(self):
        """Mock response for httpx client."""
        with patch("mcp_server_qdrant.embeddings.lmstudio.httpx.AsyncClient") as mock_client:
            client_instance = AsyncMock()
            mock_post = AsyncMock()
            mock_post.json.return_value = {
                "data": [
                    {"embedding": [0.01, 0.02, 0.03, 0.04]},
                    {"embedding": [0.05, 0.06, 0.07, 0.08]}
                ],
                "model": "nomic-embed-text-v1.5",
                "object": "list",
                "usage": {"prompt_tokens": 15, "total_tokens": 15}
            }
            mock_post.status_code = 200
            client_instance.post = AsyncMock(return_value=mock_post)
            mock_client.return_value.__aenter__.return_value = client_instance
            yield mock_client

    async def test_initialization(self, mock_settings):
        """Test that the provider can be initialized with a valid model."""
        provider = LMStudioProvider(
            mock_settings.embedding_model,
            api_base=mock_settings.lmstudio_api_base,
            vector_size=mock_settings.vector_size
        )
        assert provider.model_name == mock_settings.embedding_model
        assert provider.api_base == mock_settings.lmstudio_api_base
        assert provider.vector_size == mock_settings.vector_size

    async def test_embed_documents(self, mock_settings, mock_response):
        """Test that documents can be embedded."""
        provider = LMStudioProvider(
            mock_settings.embedding_model,
            api_base=mock_settings.lmstudio_api_base,
            vector_size=mock_settings.vector_size
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

    async def test_embed_query(self, mock_settings, mock_response):
        """Test that queries can be embedded."""
        provider = LMStudioProvider(
            mock_settings.embedding_model,
            api_base=mock_settings.lmstudio_api_base,
            vector_size=mock_settings.vector_size
        )
        query = "This is a test query."

        embedding = await provider.embed_query(query)

        # With our mocked response, we expect a 4-dimensional vector
        assert len(embedding) == 4

    async def test_get_vector_name(self, mock_settings):
        """Test that the vector name is generated correctly."""
        provider = LMStudioProvider(
            "nomic-embed-text-v1.5",
            api_base=mock_settings.lmstudio_api_base,
            vector_size=mock_settings.vector_size
        )
        vector_name = provider.get_vector_name()

        # Check that the vector name follows the expected format
        assert vector_name.startswith("lms-")
        assert "nomic-embed" in vector_name