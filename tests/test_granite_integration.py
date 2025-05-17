import numpy as np
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from mcp_server_qdrant.embeddings.granite import GraniteProvider
from mcp_server_qdrant.settings import Settings


@pytest.mark.asyncio
class TestGraniteProviderIntegration:
    """Integration tests for GraniteProvider."""

    @pytest.fixture
    def mock_settings(self):
        settings = Settings(
            embedding_provider="granite",
            embedding_model="IBM/granite-13b-embeddings",
            vector_size=1024,
        )
        return settings

    @pytest.fixture
    def mock_model(self):
        """Mock transformers model and tokenizer."""
        with patch("mcp_server_qdrant.embeddings.granite.AutoModel") as mock_model, \
             patch("mcp_server_qdrant.embeddings.granite.AutoTokenizer") as mock_tokenizer:
            
            # Mock tokenizer
            tokenizer = MagicMock()
            tokenizer.return_value = tokenizer
            tokenizer.__call__.return_value = {
                "input_ids": [[101, 2023, 2003, 1037, 3231, 2653, 1012, 102]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]]
            }
            tokenizer.batch_encode_plus.return_value = {
                "input_ids": [[101, 2023, 2003, 1037, 3231, 2653, 1012, 102], 
                             [101, 2023, 2003, 2178, 3231, 2653, 1012, 102]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1], 
                                  [1, 1, 1, 1, 1, 1, 1, 1]]
            }
            mock_tokenizer.from_pretrained.return_value = tokenizer
            
            # Mock model
            model = MagicMock()
            output = MagicMock()
            # Create a tensor-like object with a shape attribute and to_list method
            output.last_hidden_state = MagicMock()
            output.last_hidden_state.shape = [1, 8, 4]  # [batch_size, seq_len, embedding_dim]
            
            # For single document
            single_embeddings = np.random.rand(1, 8, 4)
            output.last_hidden_state.cpu().numpy.return_value = single_embeddings
            
            # For batch of documents
            batch_embeddings = np.random.rand(2, 8, 4)
            
            # Set up model to return different outputs for different calls
            model.side_effect = [output, MagicMock(last_hidden_state=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=batch_embeddings)
                ))
            ))]
            
            mock_model.from_pretrained.return_value = model
            
            yield mock_model, mock_tokenizer

    async def test_initialization(self, mock_settings, mock_model):
        """Test that the provider can be initialized with a valid model."""
        provider = GraniteProvider(
            mock_settings.embedding_model,
            vector_size=mock_settings.vector_size
        )
        assert provider.model_name == mock_settings.embedding_model
        assert provider.vector_size == mock_settings.vector_size

    async def test_embed_documents(self, mock_settings, mock_model):
        """Test that documents can be embedded."""
        provider = GraniteProvider(
            mock_settings.embedding_model,
            vector_size=mock_settings.vector_size
        )
        documents = ["This is a test document.", "This is another test document."]

        # Mock the _get_embeddings method to return controlled output
        with patch.object(provider, '_get_embeddings') as mock_get_embeddings:
            mock_get_embeddings.return_value = [
                [0.1, 0.2, 0.3, 0.4], 
                [0.5, 0.6, 0.7, 0.8]
            ]
            
            embeddings = await provider.embed_documents(documents)

            # Check that we got the right number of embeddings
            assert len(embeddings) == len(documents)
            
            # Check dimensions based on our mock
            assert len(embeddings[0]) == 4
            assert len(embeddings[1]) == 4

            # Check that embeddings are different for different documents
            embedding1 = np.array(embeddings[0])
            embedding2 = np.array(embeddings[1])
            assert not np.array_equal(embedding1, embedding2)

    async def test_embed_query(self, mock_settings, mock_model):
        """Test that queries can be embedded."""
        provider = GraniteProvider(
            mock_settings.embedding_model,
            vector_size=mock_settings.vector_size
        )
        query = "This is a test query."

        # Mock the _get_embeddings method to return controlled output
        with patch.object(provider, '_get_embeddings') as mock_get_embeddings:
            mock_get_embeddings.return_value = [[0.1, 0.2, 0.3, 0.4]]
            
            embedding = await provider.embed_query(query)

            # Check dimensions based on our mock
            assert len(embedding) == 4

    async def test_get_vector_name(self, mock_settings):
        """Test that the vector name is generated correctly."""
        provider = GraniteProvider(
            "IBM/granite-13b-embeddings",
            vector_size=mock_settings.vector_size
        )
        vector_name = provider.get_vector_name()

        # Check that the vector name follows the expected format
        assert vector_name.startswith("granite-")
        assert "13b" in vector_name.lower()