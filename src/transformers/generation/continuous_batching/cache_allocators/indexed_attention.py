# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .full_attention import FullAttentionCacheAllocator


class IndexedAttentionCacheAllocator(FullAttentionCacheAllocator):
    """Cache allocator for a group of indexed attention layers.

    An indexed attention layer attends to the keys an indexer selects among all the keys of its sequence, so it keeps
    every token like a full attention layer. Besides its keys and values, it caches the indexer's keys, as a third row
    it declares through `cache_rows`. The selection is a mask the layer builds, which the block table kernels cannot
    take, so the decode fast path is not available.
    """

    supports_block_sharing = True
    supports_block_table = False
    layer_type = "indexed_attention"
