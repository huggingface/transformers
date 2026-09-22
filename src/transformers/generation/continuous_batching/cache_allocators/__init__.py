# Attention types
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


from .cache_allocator import CacheAllocator
from .cache_pool import CachePool
from .full_attention import FullAttentionCacheAllocator
from .sliding_attention import SlidingAttentionCacheAllocator


__all__ = [
    "FullAttentionCacheAllocator",
    "SlidingAttentionCacheAllocator",
    "CacheAllocator",
    "CachePool",
]
