import pytest
from unittest.mock import patch, MagicMock

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.embeddings.gguf import GGUFEmbeddingProvider
from mcp_server_qdrant.embeddings.lmstudio import LMStudioEmbeddingProvider
from mcp_server_qdrant.embeddings.granite import GraniteEmbeddingProvider
from mcp_server_qdrant.settings import EmbeddingProviderSettings


class TestEmbeddingFactory:
    """Tests for the embedding provider factory."""

    @pytest.fixture
    def mock_settings(self):
        """Create settings with different provider configurations."""
        return {
            "fastembed": EmbeddingProviderSettings(
                provider_type=EmbeddingProviderType.FASTEMBED,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            "gguf": EmbeddingProviderSettings(
                provider_type=EmbeddingProviderType.GGUF,
                model_name="/path/to/model.gguf",
                vector_size=384,
                max_context_length=2048
            ),
            "lmstudio": EmbeddingProviderSettings(
                provider_type=EmbeddingProviderType.LMSTUDIO,
                model_name="nomic-embed-text-v1.5",
                vector_size=768,
                lmstudio_api_base="http://localhost:11433/v1"
            ),
            "granite": EmbeddingProviderSettings(
                provider_type=EmbeddingProviderType.GRANITE,
                model_name="IBM/granite-13b-embeddings",
                vector_size=1024
            )
        }

    @patch("mcp_server_qdrant.embeddings.factory.FastEmbedProvider")
    def test_create_fastembed_provider(self, mock_fastembed, mock_settings):
        """Test creating a FastEmbedProvider."""
        settings = mock_settings["fastembed"]
        
        # Set up the mock to return a specific instance
        mock_instance = MagicMock(spec=FastEmbedProvider)
        mock_fastembed.return_value = mock_instance
        
        provider = create_embedding_provider(settings)
        
        # Verify the provider was created with the correct arguments
        mock_fastembed.assert_called_once_with(settings.model_name)
        assert provider is mock_instance

    @patch("mcp_server_qdrant.embeddings.factory.GGUFEmbeddingProvider")
    def test_create_gguf_provider(self, mock_gguf, mock_settings):
        """Test creating a GGUFEmbeddingProvider."""
        settings = mock_settings["gguf"]
        
        # Set up the mock to return a specific instance
        mock_instance = MagicMock(spec=GGUFEmbeddingProvider)
        mock_gguf.return_value = mock_instance
        
        provider = create_embedding_provider(settings)
        
        # Verify the provider was created with the correct arguments
        mock_gguf.assert_called_once_with(
            model_path=settings.model_name,
            vector_size=settings.vector_size,
            max_context_length=settings.max_context_length,
            llama_cpp_path=settings.llama_cpp_path
        )
        assert provider is mock_instance

    @patch("mcp_server_qdrant.embeddings.factory.LMStudioEmbeddingProvider")
    def test_create_lmstudio_provider(self, mock_lmstudio, mock_settings):
        """Test creating a LMStudioEmbeddingProvider."""
        settings = mock_settings["lmstudio"]
        
        # Set up the mock to return a specific instance
        mock_instance = MagicMock(spec=LMStudioEmbeddingProvider)
        mock_lmstudio.return_value = mock_instance
        
        provider = create_embedding_provider(settings)
        
        # Verify the provider was created with the correct arguments
        mock_lmstudio.assert_called_once_with(
            model_name=settings.model_name,
            api_base=settings.lmstudio_api_base,
            api_key=settings.lmstudio_api_key,
            vector_size=settings.vector_size
        )
        assert provider is mock_instance

    @patch("mcp_server_qdrant.embeddings.factory.GraniteEmbeddingProvider")
    def test_create_granite_provider(self, mock_granite, mock_settings):
        """Test creating a GraniteEmbeddingProvider."""
        settings = mock_settings["granite"]
        
        # Set up the mock to return a specific instance
        mock_instance = MagicMock(spec=GraniteEmbeddingProvider)
        mock_granite.return_value = mock_instance
        
        provider = create_embedding_provider(settings)
        
        # Verify the provider was created with the correct arguments
        mock_granite.assert_called_once_with(
            model_name=settings.model_name,
            device="cpu",  # Default value
            normalize_embeddings=True,  # Default value
            max_length=512  # Default value
        )
        assert provider is mock_instance

    def test_unknown_provider_type(self):
        """Test that an unknown provider type raises a ValueError."""
        settings = EmbeddingProviderSettings(
            provider_type="unknown_provider",
            model_name="some_model"
        )
        
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            create_embedding_provider(settings)