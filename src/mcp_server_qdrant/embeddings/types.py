from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    GGUF = "gguf"
    LMSTUDIO = "lmstudio"
    GRANITE = "granite"