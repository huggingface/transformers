from .cache_allocator import CacheAllocator
from .cache_pool import CachePool
from .compressed_latents import IndexedMLACacheAllocator, MLACacheAllocator
from .full_attention import FullAttentionCacheAllocator
from .sliding_attention import SlidingAttentionCacheAllocator


# Attention types
FULL_ATTENTION = FullAttentionCacheAllocator.layer_type
SLIDING_ATTENTION = SlidingAttentionCacheAllocator.layer_type
INDEXED_ATTENTION = IndexedMLACacheAllocator.layer_type


__all__ = [
    "FullAttentionCacheAllocator",
    "IndexedMLACacheAllocator",
    "MLACacheAllocator",
    "SlidingAttentionCacheAllocator",
    "CacheAllocator",
    "CachePool",
]
