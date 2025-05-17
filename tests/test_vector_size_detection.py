import pytest

from mcp_server_qdrant.embeddings.factory import get_model_vector_size


class TestVectorSizeDetection:
    """Tests for the model vector size detection functionality."""

    def test_known_model_detection(self):
        """Test that known models are correctly mapped to their vector sizes."""
        # Test a few common models
        assert get_model_vector_size("sentence-transformers/all-MiniLM-L6-v2") == 384
        assert get_model_vector_size("sentence-transformers/all-mpnet-base-v2") == 768
        assert get_model_vector_size("BAAI/bge-small-en-v1.5") == 384
        assert get_model_vector_size("BAAI/bge-base-en-v1.5") == 768
        assert get_model_vector_size("BAAI/bge-large-en-v1.5") == 1024
        
        # Test the IBM Granite model
        assert get_model_vector_size("IBM/granite-13b-embeddings") == 1024
        
        # Test Nomic Embed models
        assert get_model_vector_size("nomic-embed-text-v1.5") == 768
        assert get_model_vector_size("nomic-embed-text-v1") == 768

    def test_case_insensitive_detection(self):
        """Test that model detection is case-insensitive."""
        assert get_model_vector_size("Sentence-Transformers/all-MiniLM-L6-v2") == 384
        assert get_model_vector_size("sentence-transformers/ALL-MINILM-L6-V2") == 384
        assert get_model_vector_size("BAAI/bge-small-en-v1.5") == 384
        assert get_model_vector_size("baai/BGE-small-EN-v1.5") == 384

    def test_unknown_model_returns_none(self):
        """Test that unknown models return None."""
        assert get_model_vector_size("unknown-model") is None
        assert get_model_vector_size("custom-embedding-model") is None

    def test_invalid_input(self):
        """Test that invalid inputs are handled gracefully."""
        assert get_model_vector_size(None) is None
        assert get_model_vector_size("") is None
        assert get_model_vector_size(123) is None  # Non-string input
        
    def test_partial_match_handling(self):
        """Test that partial matches are handled correctly."""
        # These should match because they contain the key substrings
        assert get_model_vector_size("my-custom-sentence-transformers/all-MiniLM-L6-v2-fine-tuned") == 384
        assert get_model_vector_size("custom-nomic-embed-text-v1.5-model") == 768
        
        # These should not match because they're too different
        assert get_model_vector_size("minilm-custom") is None
        assert get_model_vector_size("some-model-with-bge-in-name") is None