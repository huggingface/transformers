# Attention types
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


from .full_attention import FullAttentionCacheAllocator
from .sliding_attention import SlidingAttentionCacheAllocator
from .cache_allocator import CacheAllocator


__all__ = ["FullAttentionCacheAllocator", "SlidingAttentionCacheAllocator", "CacheAllocator"]
