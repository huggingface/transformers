"""
Fixed DynamicLayer with empty tensor handling for torch.compile compatibility.

This fix addresses GitHub issue #42027 where torch.cat receives empty tensors
during torch.compile tracing in GPT2 models.
"""

import torch
from typing import Optional, Any


class FixedDynamicLayerMixin:
    """
    Mixin to add empty tensor handling to DynamicLayer.
    
    Applies to transformers.cache_utils.DynamicLayer to fix torch.compile
    empty tensor concatenation issues.
    """
    
    def update_with_empty_handling(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with defensive empty tensor handling.
        
        This method adds a check for empty tensors before concatenation,
        preventing "Expected a non-empty list of Tensors" errors during
        torch.compile tracing.
        
        Args:
            key_states: New key states to cache
            value_states: New value states to cache
            cache_kwargs: Optional cache arguments
            
        Returns:
            Updated key and value tensors
        """
        # FIXED: Check for empty cache before concatenation
        if self.keys.numel() == 0 and self.values.numel() == 0:
            # Direct assignment for empty cache
            self.keys = key_states
            self.values = value_states
        else:
            # Normal concatenation for non-empty cache
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
            
        return self.keys, self.values


def apply_empty_tensor_fix():
    """
    Apply the empty tensor fix to DynamicLayer.
    
    This function monkeypatches the DynamicLayer class with the fix.
    Call this at module import time to enable the fix globally.
    """
    from transformers.cache_utils import DynamicLayer
    
    # Store original method
    DynamicLayer._original_update = DynamicLayer.update
    
    # Apply fixed method
    def fixed_update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        
        # Apply fix
        if self.keys.numel() == 0 and self.values.numel() == 0:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
            
        return self.keys, self.values
    
    DynamicLayer.update = fixed_update
    return DynamicLayer
