from .cache_allocator import CacheAllocator
from .cache_pool import CachePool
from .full_attention import FullAttentionCacheAllocator, RowSpec
from .indexed_attention import IndexedAttentionCacheAllocator
from .sliding_attention import SlidingAttentionCacheAllocator
from .states import StateCacheAllocator, StateSpec


# Attention types
FULL_ATTENTION = FullAttentionCacheAllocator.layer_type
SLIDING_ATTENTION = SlidingAttentionCacheAllocator.layer_type
INDEXED_ATTENTION = IndexedAttentionCacheAllocator.layer_type
LINEAR_ATTENTION = StateCacheAllocator.layer_type


__all__ = [
    "FullAttentionCacheAllocator",
    "IndexedAttentionCacheAllocator",
    "RowSpec",
    "StateCacheAllocator",
    "StateSpec",
    "SlidingAttentionCacheAllocator",
    "CacheAllocator",
    "CachePool",
]
