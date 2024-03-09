from enum import Enum


class PositionEmbeddingType(Enum):
    """
    Enum class for position embeddings
    """

    learned_absolute = "learned_absolute"
    alibi = "alibi"
    rope = "rope"
    nope = "nope"


class AttentionHeadType(Enum):
    """
    Enum class for attention head type
    """

    mha = "mha"
    mqa = "mqa"
    gqa = "gqa"


class AttentionImplementation(Enum):
    """
    Enum class for attention implementation
    """

    eager = "eager"
    math = "math"
    sdpa = "sdpa"
    flash = "flash"
    padding_free = "padding_free"


class NormalizationImplementation(Enum):
    """
    Enum class for normalization function implementation
    """

    torch = "torch"
    apex = "apex"
    apex_persistent = "apex_persistent"
