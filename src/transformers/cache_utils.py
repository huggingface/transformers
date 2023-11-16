from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypeVar

import torch


T = TypeVar("T")


class Cache(ABC):
    def __init__(self) -> None:
        self.key_cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.value_cache: Dict[int, Tuple[torch.Tensor]] = {}

    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        return (
            tuple(self.key_cache[layer_idx] for layer_idx in range(len(self.key_cache))),
            tuple(self.value_cache[layer_idx] for layer_idx in range(len(self.value_cache))),
        )

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[List[torch.FloatTensor]]) -> "DynamicCache":
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
            cache.update(key_states, value_states, layer_idx)
        return cache


class DynamicCache(Cache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class PagedAttentionCache(Cache):
    def __init__(self, num_blocks: int = 1e8, block_size: int = 16) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.context_lens = []
        self.key_cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.value_cache: Dict[int, Tuple[torch.Tensor]] = {} 
        self.block_tables = [] #mapping logical block to physical blocks
        self.free_blocks = list(range(num_blocks)) #free blocks
        self.block_ref_count = [0] * self.num_blocks #init the reference count for each physical block      
        self.slots_mapping = [] #mapping logical slots to physical slots. 
    
    def allocate(self, batch_idx: int, key_len: int, context_len: int) -> torch.Tensor:        
        #return the physical slots for this sequence, 1 slot for 1 token state.
        #if batch_idx not in self.block_tables, allocate blocks for this sequence 
        slots = []
        if batch_idx not in self.block_tables:
            #allocate blocks for this sequence
            assert context_len == 0
            needed_blocks = (key_len + self.block_size - 1) // self.block_size
            assert needed_blocks <= len(self.free_blocks)
            blocks = self.free_blocks[:needed_blocks]
            self.free_blocks = self.free_blocks[needed_blocks:]
            self.block_tables[batch_idx] = blocks
            for block_idx in blocks:
                self.block_ref_count[block_idx] += 1
            #return the slots for this sequence            
            for i in range(key_len):
                slots.append(blocks[i // self.block_size] * self.block_size + i % self.block_size)
            return slots     
            
        else:
            #find free slots in the allocated blocks or find new blocks 
            seq_len = key_len + context_len
            needed_blocks = (seq_len + self.block_size - 1) // self.block_size - len(self.block_tables[batch_idx])
            assert needed_blocks <= len(self.free_blocks)
            self.block_tables[batch_idx].extend(self.free_blocks[:needed_blocks])
            self.free_blocks = self.free_blocks[needed_blocks:]
            for block_idx in self.block_tables[batch_idx]:
                self.block_ref_count[block_idx] += 1
            #return the slots for this sequence
            for i in range(seq_len):
                slots.append(self.block_tables[batch_idx][i // self.block_size] * self.block_size + i % self.block_size)
            return slots
            
    
    def free(self, batch_idx: int):
        #free the blocks allocated for this sequence
        assert batch_idx in self.block_tables
        for block_idx in self.block_tables[batch_idx]:
            self.block_ref_count[block_idx] -= 1
            if self.block_ref_count[block_idx] == 0:
                self.free_blocks.append(block_idx)
    
    def reshape_and_cache(self, slot_mapping: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        pass
     
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.context_lens[0] #assume all sequences have the same length while unpad should get better performance
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = key_states.shape[0] #[batch, head, seq, dim]
        kv_head = key_states.shape[1]
        head_size = key_states.shape[-1]
        if layer_idx not in self.key_cache: #init the cache
            self.key_cache[layer_idx] = torch.zeros((self.num_blocks, self.block_size, kv_head, head_size), dtype=key_states.dtype, device=key_states.device)
            self.value_cache[layer_idx] = torch.zeros((self.num_blocks, self.block_size, kv_head, head_size), dtype=value_states.dtype, device=value_states.device)        
        #step 1): allocate slots to store token states for each sequence
        if layer_idx == 0:
            self.slots_mapping = []  
            for seq in range(batch_size):
                seq_len = key_states[seq].shape[-2] 
                self.context_lens.append(seq_len)            
                slots = self.allocate(seq, seq_len, self.context_lens[seq])
                self.slots_mapping.append(slots)
        assert len(self.slots_mapping) == batch_size
        #step 2): cache key_states & value states
        self.reshape_and_cache(self.slots_mapping, key_states, value_states, layer_idx)
        
            
                

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(
    key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.IntTensor] = None
) -> torch.Tensor:
    if position_ids:
        cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    rotated_key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return rotated_key_states


class SinkCache(Cache):
    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        super().__init__()
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_cache = {}

    def get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            ori