import pytest
from unittest.mock import patch, MagicMock

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.embeddings.gguf import GGUFProvider
from mcp_server_qdrant.embeddings.lmstudio import LMStudioProvider
from mcp_server_qdrant.embeddings.granite import GraniteProvider
from mcp_server_qdrant.settings import Settings


class TestEmbeddingFactory:
    """Tests for the embedding provider factory."""

    @pytest.fixture
    def mock_settings(self):
        """Create settings with different provider configurations."""
        return {
            "fastembed": Settings(
                embedding_provider="fastembed",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            ),
            "gguf": Settings(
                embedding_provider="gguf",
                embedding_model_path="/path/to/model.gguf",
                vector_size=384,
                max_context_length=2048
            ),
            "lmstudio": Settings(
                embedding_provider="lmstudio",
                embedding_model="nomic-embed-text-v1.5",
                vector_size=768,
                lmstudio_api_base="http://localhost:1234/v1"
            ),
            "granite": Settings(
                embedding_provider="granite",
                embedding_model="IBM/granite-13b-embeddings",
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
        mock_fastembed.assert_called_once_with(
            settings.embedding_model,
            vector_size=None  # FastEmbed auto-detects vector size
        )
        assert provider is mock_instance

    @patch("mcp_server_qdrant.embeddings.factory.GGUFProvider")
    def test_create_gguf_provider(self, mock_gguf, mock_settings):
        """Test creating a GGUFProvider."""
        settings = mock_settings["gguf"]
        
        # Set up the mock to return a specific instance
        mock_instance = MagicMock(spec=GGUFProvider)
        mock_gguf.return_value = mock_instance
        
        provider = create_embedding_provider(settings)
        
        # Verify the provider was created with the correct arguments
        mock_gguf.assert_called_once_with(
            settings.embedding_model_path,
            vector_size=settings.vector_size,
            max_context_length=settings.max_context_length
        )
        assert provider is mock_instance

    @patch("mcp_server_qdrant.embeddings.factory.LMStudioProvider")
    def test_create_lmstudio_provider(self, mock_lmstudio, mock_settings):
        """Test creating a LMStudioProvider."""
        settings = mock_settings["lmstudio"]
        
        # Set up the mock to return a specific instance
        mock_instance = MagicMock(spec=LMStudioProvider)
        mock_lmstudio.return_value = mock_instance
        
        provider = create_embedding_provider(settings)
        
        # Verify the provider was created with the correct arguments
        mock_lmstudio.assert_called_once_with(
            settings.embedding_model,
            api_base=settings.lmstudio_api_base,
            vector_size=settings.vector_size
        )
        assert provider is mock_instance

    @patch("mcp_server_qdrant.embeddings.factory.GraniteProvider")
    def test_create_granite_provider(self, mock_granite, mock_settings):
        """Test creating a GraniteProvider."""
        settings = mock_settings["granite"]
        
        # Set up the mock to return a specific instance
        mock_instance = MagicMock(spec=GraniteProvider)
        mock_granite.return_value = mock_instance
        
        provider = create_embedding_provider(settings)
        
        # Verify the provider was created with the correct arguments
        mock_granite.assert_called_once_with(
            settings.embedding_model,
            vector_size=settings.vector_size
        )
        assert provider is mock_instance

    def test_unknown_provider_type(self, mock_settings):
        """Test that an unknown provider type raises a ValueError."""
        settings = Settings(
            embedding_provider="unknown_provider",
            embedding_model="some_model"
        )
        
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            create_embedding_provider(settings)

    @patch("mcp_server_qdrant.embeddings.factory.FastEmbedProvider")
    def test_auto_detect_vector_size(self, mock_fastembed, mock_settings):
        """Test that vector size is auto-detected for known models."""
        settings = Settings(
            embedding_provider="fastembed",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_size=None  # Deliberately omit vector size
        )
        
        # Set up the mock
        mock_instance = MagicMock(spec=FastEmbedProvider)
        mock_fastembed.return_value = mock_instance
        
        with patch("mcp_server_qdrant.embeddings.factory.get_model_vector_size") as mock_get_size:
            mock_get_size.return_value = 384
            
            provider = create_embedding_provider(settings)
            
            # Check that get_model_vector_size was called
            mock_get_size.assert_called_once_with(settings.embedding_model)
            
            # Check that the provider was created with the detected size
            mock_fastembed.assert_called_once_with(
                settings.embedding_model,
                vector_size=384
            )