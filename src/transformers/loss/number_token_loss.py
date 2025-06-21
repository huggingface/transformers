# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""
Number Token Loss (NTL) implementation for transformers.

This module provides two variants of Number Token Loss:
- NTL-WAS: Uses Wasserstein-1 distance between numerical values
- NTL-MSE: Uses Mean Squared Error between numerical values

The loss is designed to augment cross-entropy loss for language models when dealing with numerical tokens,
providing a more meaningful loss signal for tokens that represent numbers.
"""

import re
import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_numerical_value(token: str) -> Optional[float]:
    """
    Extract numerical value from a token string.
    
    Args:
        token: The token string to extract numerical value from
        
    Returns:
        The numerical value as float, or None if the token is not numerical
        
    Examples:
        >>> extract_numerical_value("123")
        123.0
        >>> extract_numerical_value("3.14")
        3.14
        >>> extract_numerical_value("hello")
        None
    """
    token_lower = token.lower()
    
    # Handle special number words
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'million': 1000000, 'billion': 1000000000
    }
    
    if token_lower in number_words:
        return float(number_words[token_lower])
    
    # Handle ordinal words (check before removing suffixes)
    ordinal_words = {
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
        'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14, 'fifteenth': 15,
        'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20
    }
    
    if token_lower in ordinal_words:
        return float(ordinal_words[token_lower])
    
    # Remove common suffixes that might be attached to numbers
    token_clean = re.sub(r'(st|nd|rd|th|s)$', '', token_lower)
    
    # Try to parse as a regular number
    try:
        # Remove commas from numbers like "1,000"
        token_clean = token_clean.replace(',', '')
        
        # Handle scientific notation
        if 'e' in token_clean.lower():
            return float(token_clean)
        
        # Handle regular numbers (integers and floats)
        if re.match(r'^[+-]?\d*\.?\d+$', token_clean):
            return float(token_clean)
        
        # Handle numbers with currency symbols
        if re.match(r'^[+-]?[$€£¥₹₽฿₺₴₣₡₱₪₮₩₦₫﷼]?\d*\.?\d+$', token_clean):
            # Remove currency symbols and parse
            cleaned = re.sub(r'[^0-9+-.]', '', token_clean)
            return float(cleaned)
            
    except (ValueError, TypeError):
        pass
    
    return None


def build_token_to_number_map(tokenizer) -> Dict[int, float]:
    """
    Build a mapping from token IDs to their numerical values.
    
    Args:
        tokenizer: The tokenizer to extract tokens from
        
    Returns:
        Dictionary mapping token IDs to numerical values
    """
    token_to_number = {}
    
    for token_id in range(tokenizer.vocab_size):
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
            numerical_value = extract_numerical_value(token)
            if numerical_value is not None:
                token_to_number[token_id] = numerical_value
        except (KeyError, ValueError):
            # Skip tokens that can't be converted
            continue
    
    return token_to_number


def wasserstein_1_distance_numerical(
    pred_dist: torch.Tensor, 
    target_dist: torch.Tensor, 
    token_to_number_map: Dict[int, float],
    vocab_size: int
) -> torch.Tensor:
    """
    Compute Wasserstein-1 distance between predicted and target distributions
    for numerical tokens.
    
    For one-hot target distributions, this computes the expected absolute difference
    between the predicted and target numerical values.
    
    Args:
        pred_dist: Predicted distribution [batch_size, vocab_size]
        target_dist: Target distribution [batch_size, vocab_size] (one-hot)
        token_to_number_map: Mapping from token IDs to numerical values
        vocab_size: Size of the vocabulary
        
    Returns:
        Wasserstein-1 distance for each sample in the batch
    """
    batch_size = pred_dist.shape[0]
    device = pred_dist.device
    
    # Get target token indices
    target_indices = torch.argmax(target_dist, dim=-1)  # [batch_size]
    
    # Get target numerical values
    target_values = torch.zeros(batch_size, device=device)
    for i, token_id in enumerate(target_indices):
        if token_id.item() in token_to_number_map:
            target_values[i] = token_to_number_map[token_id.item()]
        else:
            # If target is not numerical, set to NaN to ignore
            target_values[i] = float('nan')
    
    # Compute expected predicted numerical values
    pred_values = torch.zeros(batch_size, device=device)
    for token_id, num_value in token_to_number_map.items():
        pred_values += pred_dist[:, token_id] * num_value
    
    # Compute absolute difference (Wasserstein-1 distance)
    # Only for positions where target is numerical
    valid_mask = ~torch.isnan(target_values)
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    
    was_distances = torch.abs(pred_values[valid_mask] - target_values[valid_mask])
    return was_distances


def ntl_was_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_to_number_map: Dict[int, float],
    vocab_size: int,
    alpha: float = 0.1,
    ignore_index: int = -100,
    **kwargs
) -> torch.Tensor:
    """
    Number Token Loss using Wasserstein-1 distance (NTL-WAS).
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        token_to_number_map: Mapping from token IDs to numerical values
        vocab_size: Size of the vocabulary
        alpha: Weight for NTL loss (default: 0.1)
        ignore_index: Index to ignore in loss computation (default: -100)
        **kwargs: Additional arguments
        
    Returns:
        Combined loss (CE + alpha * NTL-WAS)
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Create numerical value tensors
    device = logits.device
    batch_size, seq_len = labels.shape
    
    # Initialize numerical values tensor
    numerical_values = torch.full((batch_size, seq_len), float('nan'), device=device)
    
    # Fill in numerical values for tokens that have them
    for token_id, num_value in token_to_number_map.items():
        mask = (labels == token_id)
        numerical_values[mask] = num_value
    
    # Only compute NTL for positions with numerical values
    numerical_mask = ~torch.isnan(numerical_values)
    
    if not numerical_mask.any():
        return ce_loss
    
    # Extract logits and labels for numerical positions
    num_logits = logits[numerical_mask]  # [num_numerical, vocab_size]
    num_labels = labels[numerical_mask]  # [num_numerical]
    
    # Create target distribution (one-hot)
    target_dist = F.one_hot(num_labels, num_classes=vocab_size).float()
    
    # Create predicted distribution (softmax of logits)
    pred_dist = F.softmax(num_logits, dim=-1)
    
    # Compute Wasserstein-1 distance
    was_distances = wasserstein_1_distance_numerical(pred_dist, target_dist, token_to_number_map, vocab_size)
    
    # Average over numerical positions
    ntl_loss = was_distances.mean()
    
    return ce_loss + alpha * ntl_loss


def ntl_mse_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_to_number_map: Dict[int, float],
    vocab_size: int,
    alpha: float = 0.1,
    ignore_index: int = -100,
    **kwargs
) -> torch.Tensor:
    """
    Number Token Loss using Mean Squared Error (NTL-MSE).
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        token_to_number_map: Mapping from token IDs to numerical values
        vocab_size: Size of the vocabulary
        alpha: Weight for NTL loss (default: 0.1)
        ignore_index: Index to ignore in loss computation (default: -100)
        **kwargs: Additional arguments
        
    Returns:
        Combined loss (CE + alpha * NTL-MSE)
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Create numerical value tensors
    device = logits.device
    batch_size, seq_len = labels.shape
    
    # Initialize numerical values tensor
    numerical_values = torch.full((batch_size, seq_len), float('nan'), device=device)
    
    # Fill in numerical values for tokens that have them
    for token_id, num_value in token_to_number_map.items():
        mask = (labels == token_id)
        numerical_values[mask] = num_value
    
    # Only compute NTL for positions with numerical values
    numerical_mask = ~torch.isnan(numerical_values)
    
    if not numerical_mask.any():
        return ce_loss
    
    # Extract logits and labels for numerical positions
    num_logits = logits[numerical_mask]  # [num_numerical, vocab_size]
    num_labels = labels[numerical_mask]  # [num_numerical]
    num_values = numerical_values[numerical_mask]  # [num_numerical]
    
    # Create predicted distribution (softmax of logits)
    pred_dist = F.softmax(num_logits, dim=-1)
    
    # Compute MSE between predicted and target numerical values
    # For each position, compute the expected numerical value
    pred_numerical_values = torch.zeros_like(num_values)
    for token_id, num_value in token_to_number_map.items():
        pred_numerical_values += pred_dist[:, token_id] * num_value
    
    # MSE loss
    mse_loss = F.mse_loss(pred_numerical_values, num_values)
    
    return ce_loss + alpha * mse_loss


# Global token_to_number_map cache to avoid rebuilding for each loss computation
_token_to_number_map_cache = {}


def get_token_to_number_map(tokenizer):
    """
    Get or create token-to-number mapping with caching.
    
    Args:
        tokenizer: The tokenizer to extract tokens from
        
    Returns:
        Dictionary mapping token IDs to numerical values
    """
    tokenizer_id = id(tokenizer)
    if tokenizer_id not in _token_to_number_map_cache:
        _token_to_number_map_cache[tokenizer_id] = build_token_to_number_map(tokenizer)
    return _token_to_number_map_cache[tokenizer_id]


def ForCausalLMWithNTLWAS(
    logits,
    labels,
    vocab_size: int,
    tokenizer=None,
    alpha: float = 0.1,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Causal LM loss augmented with Number Token Loss using Wasserstein-1 distance.
    
    Args:
        logits: Model logits
        labels: Target labels
        vocab_size: Size of the vocabulary
        tokenizer: Tokenizer for extracting numerical tokens
        alpha: Weight for NTL loss
        num_items_in_batch: Number of items in batch (for compatibility)
        ignore_index: Index to ignore in loss computation
        shift_labels: Shifted labels (for compatibility)
        **kwargs: Additional arguments
        
    Returns:
        Combined loss (CE + alpha * NTL-WAS)
    """
    if tokenizer is None:
        # Fall back to standard CE loss if no tokenizer provided
        from .loss_utils import ForCausalLMLoss
        return ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch, ignore_index, shift_labels, **kwargs)
    
    # Handle label shifting for causal LM
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    
    # Get token-to-number mapping
    token_to_number_map = get_token_to_number_map(tokenizer)
    
    # Compute NTL-WAS loss
    return ntl_was_loss(
        logits, shift_labels, token_to_number_map, vocab_size,
        alpha, ignore_index, **kwargs
    )


def ForCausalLMWithNTLMSE(
    logits,
    labels,
    vocab_size: int,
    tokenizer=None,
    alpha: float = 0.1,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Causal LM loss augmented with Number Token Loss using MSE.
    
    Args:
        logits: Model logits
        labels: Target labels
        vocab_size: Size of the vocabulary
        tokenizer: Tokenizer for extracting numerical tokens
        alpha: Weight for NTL loss
        num_items_in_batch: Number of items in batch (for compatibility)
        ignore_index: Index to ignore in loss computation
        shift_labels: Shifted labels (for compatibility)
        **kwargs: Additional arguments
        
    Returns:
        Combined loss (CE + alpha * NTL-MSE)
    """
    if tokenizer is None:
        # Fall back to standard CE loss if no tokenizer provided
        from .loss_utils import ForCausalLMLoss
        return ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch, ignore_index, shift_labels, **kwargs)
    
    # Handle label shifting for causal LM
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    
    # Get token-to-number mapping
    token_to_number_map = get_token_to_number_map(tokenizer)
    
    # Compute NTL-MSE loss
    return ntl_mse_loss(
        logits, shift_labels, token_to_number_map, vocab_size,
        alpha, ignore_index, **kwargs
    )


class NumberTokenLoss:
    """
    Number Token Loss class that can be used as a drop-in replacement for standard loss functions.
    
    This class maintains the token-to-number mapping and provides both NTL-WAS and NTL-MSE variants.
    """
    
    def __init__(
        self,
        tokenizer,
        variant: str = "was",
        alpha: float = 0.1,
        ignore_index: int = -100
    ):
        """
        Initialize Number Token Loss.
        
        Args:
            tokenizer: The tokenizer to extract numerical tokens from
            variant: Loss variant ("was" or "mse")
            alpha: Weight for NTL loss
            ignore_index: Index to ignore in loss computation
        """
        self.token_to_number_map = build_token_to_number_map(tokenizer)
        self.variant = variant.lower()
        self.alpha = alpha
        self.ignore_index = ignore_index
        
        if self.variant not in ["was", "mse"]:
            raise ValueError(f"Unknown variant: {variant}. Must be 'was' or 'mse'")
    
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the Number Token Loss.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            vocab_size: Size of the vocabulary
            **kwargs: Additional arguments
            
        Returns:
            Combined loss (CE + alpha * NTL)
        """
        if self.variant == "was":
            return ntl_was_loss(
                logits, labels, self.token_to_number_map, vocab_size,
                self.alpha, self.ignore_index, **kwargs
            )
        else:  # mse
            return ntl_mse_loss(
                logits, labels, self.token_to_number_map, vocab_size,
                self.alpha, self.ignore_index, **kwargs
            ) 