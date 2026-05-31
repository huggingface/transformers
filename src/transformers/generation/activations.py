# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Utility for extracting and stacking hidden states from :meth:`model.generate()`.

The :class:`GenerationActivations` dataclass consumes the raw per-step
``tuple[tuple[Tensor]]`` hidden states returned by
:class:`~generation.GenerateDecoderOnlyOutput` (or the equivalent
encoder-decoder output) and stacks them into a single dense tensor with
separated prompt and generated token ranges.

This removes the need for users to manually handle the nested tuple
structure, left-padding trimming, batch-dimension management, and
prompt/generation boundary detection — all of which are common sources of
confusion when working with ``output_hidden_states=True`` during generation.
"""

from __future__ import annotations

import logging as _logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch


try:
    from ..utils import logging as _hf_logging
except ImportError:
    _hf_logging = None  # type: ignore[assignment]

if TYPE_CHECKING:
    try:
        from ..tokenization_utils_base import PreTrainedTokenizerBase
        from ..utils import ModelOutput
    except ImportError:
        PreTrainedTokenizerBase = None  # type: ignore[assignment,misc]
        ModelOutput = None  # type: ignore[assignment,misc]

logger = _hf_logging.get_logger(__name__) if _hf_logging is not None else _logging.getLogger(__name__)


@dataclass
class GenerationActivations:
    """Hidden states extracted from :meth:`~generation.GenerationMixin.generate`,
    stacked into a single dense tensor with separated prompt and generated
    token ranges.

    Consumes the raw per-step ``hidden_states`` tuple-of-tuples returned by
    :class:`~generation.GenerateDecoderOnlyOutput` (or the equivalent
    :class:`~generation.GenerateEncoderDecoderOutput`) and stacks them into a
    single ``[num_layers, total_tokens, hidden_dim]`` tensor, handling
    left-padding trimming, batch dimension management, and the prompt/generation
    boundary automatically.

    Example:

    .. code-block:: python

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.activations import GenerationActivations

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        inputs = tokenizer("What is the capital of France?", return_tensors="pt")
        gen_out = model.generate(
            **inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=32,
        )

        acts = GenerationActivations.from_generate_output(gen_out)
        # acts.hidden_states           → [37, 40, 4096]
        # acts.prompt_hidden_states    → [37, 8, 4096]
        # acts.generated_hidden_states → [37, 32, 4096]
        # acts.num_layers              → 37
        # acts.prompt_len              → 8

    Args:
        hidden_states (:obj:`torch.FloatTensor` of shape ``(num_layers, total_tokens, hidden_dim)``):
            Stacked hidden states across all transformer layers, with prompt
            tokens first followed by generated tokens. When ``batch_size > 1``,
            shape is ``(batch_size, num_layers, total_tokens, hidden_dim)``.
        prompt_len (:obj:`int`):
            Number of prompt tokens (before generation started).
        num_layers (:obj:`int`):
            Number of transformer layers captured.
        hidden_dim (:obj:`int`):
            Dimensionality of the hidden states.
        batch_size (:obj:`int`, *optional*):
            Batch size. Present only when the input had ``batch_size > 1``.
        attention_mask (:obj:`torch.BoolTensor`, *optional*):
            Boolean mask over the token axis, ``True`` for valid (non-pad)
            positions. Shape ``(total_tokens,)``. Present when
            ``batch_size > 1`` and left-padding was used.
    """

    hidden_states: torch.FloatTensor
    prompt_len: int
    num_layers: int
    hidden_dim: int
    batch_size: int | None = None
    attention_mask: torch.BoolTensor | None = None

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def from_generate_output(
        cls,
        gen_output: ModelOutput,
        tokenizer: PreTrainedTokenizerBase | None = None,
        *,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
    ) -> GenerationActivations:
        """Build from a :class:`~generation.GenerateDecoderOnlyOutput` or
        :class:`~generation.GenerateEncoderDecoderOutput`.

        The first step of ``hidden_states`` (index 0) is assumed to contain the
        full prompt plus one generated token.  Subsequent steps each contain
        one generated token.  Layers are the innermost tuple element.

        Args:
            gen_output:
                The output of :meth:`model.generate()
                <transformers.GenerationMixin.generate>` with
                ``output_hidden_states=True`` and
                ``return_dict_in_generate=True``.
            tokenizer:
                Optional tokenizer used to determine the pad token id for mask
                construction. Ignored if ``attention_mask`` is provided
                directly.
            input_ids (:obj:`torch.LongTensor`, *optional*):
                The input token ids, shape ``(batch_size, seq_len)``.  Used to
                verify prompt length when ``batch_size > 1``.  If not provided,
                prompt length is inferred from the first step's hidden states.
            attention_mask (:obj:`torch.LongTensor`, *optional*):
                Attention mask from the tokenizer, shape ``(batch_size,
                seq_len)``.  Used to build a boolean valid-token mask that
                excludes left-padding positions.

        Returns:
            :class:`GenerationActivations` with stacked hidden states.

        Raises:
            ValueError: If ``gen_output.hidden_states`` is ``None`` (i.e.
                ``output_hidden_states`` was not set to ``True``).
        """
        raw = gen_output.hidden_states
        if raw is None:
            raise ValueError("gen_output.hidden_states is None. Pass output_hidden_states=True to model.generate().")
        return cls._stack(
            raw,
            tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @classmethod
    def from_generate_dict(
        cls,
        generate_output: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase | None = None,
        *,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
    ) -> GenerationActivations:
        """Build from the internal ``generate_output`` dictionary accumulated
        by the refactored generation loop (available after
        ``clean-output-handling`` / PR #40887 lands).

        The dictionary is expected to contain ``"decoder_hidden_states"`` as a
        ``tuple`` of per-step layer tuples (the same structure as
        :attr:`GenerateDecoderOnlyOutput.hidden_states`).

        Args:
            generate_output:
                The dictionary accumulated during generation.  For
                encoder-decoder models, the key ``"decoder_hidden_states"``
                is used.
            tokenizer, input_ids, attention_mask:
                See :meth:`from_generate_output`.

        Returns:
            :class:`GenerationActivations`.

        Raises:
            ValueError: If the dictionary does not contain
                ``"decoder_hidden_states"`` or its value is ``None``.
        """
        raw = generate_output.get("decoder_hidden_states")
        if raw is None:
            raise ValueError(
                "generate_output does not contain 'decoder_hidden_states'. "
                "Ensure output_hidden_states=True was passed to generate()."
            )
        return cls._stack(
            raw,
            tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # ── Internal stacking ──────────────────────────────────────────────────

    @staticmethod
    def _stack(
        raw_hidden_states: tuple[tuple[torch.FloatTensor]],
        tokenizer: PreTrainedTokenizerBase | None = None,
        *,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
    ) -> GenerationActivations:
        """Stack the per-step ``tuple[tuple[Tensor]]`` into a dense tensor.

        The first element (step 0) contains the full prompt plus one generated
        token.  Subsequent elements each contain exactly one generated token.
        Layers are stacked along dim 0, tokens along dim 1.

        When ``batch_size > 1`` and ``attention_mask`` is provided, a boolean
        valid-token mask is constructed to exclude left-padding positions.

        Args:
            raw_hidden_states: The raw hidden states as returned by
                :attr:`GenerateDecoderOnlyOutput.hidden_states`.
            tokenizer: Optional tokenizer (unused currently, reserved for
                future use in padding detection).
            input_ids: The input token ids, shape ``(batch_size, seq_len)``.
                Reserved for future use (e.g. EOS detection in generated
                tokens). Currently unused in the stacking logic.
            attention_mask: Attention mask, shape ``(batch_size, seq_len)``.

        Returns:
            A fully-populated :class:`GenerationActivations` instance.
        """
        # Validate input
        if not raw_hidden_states:
            raise ValueError("raw_hidden_states is an empty tuple. No generation steps were recorded.")

        step0 = raw_hidden_states[0]  # tuple of num_layers tensors
        num_layers = len(step0)
        hidden_dim = step0[0].shape[-1]
        batch = step0[0].shape[0]
        prompt_len = step0[0].shape[1]

        # Stack step 0 across layers → [num_layers, batch, prompt_len, D]
        prompt_stack = torch.stack([step0[layer] for layer in range(num_layers)], dim=0)

        # Stack steps 1..T, each [num_layers, batch, 1, D]
        gen_blocks: list[torch.Tensor] = []
        for step in range(1, len(raw_hidden_states)):
            step_tensors = raw_hidden_states[step]
            if not step_tensors:
                continue  # defensive: skip empty steps
            step_stack = torch.stack([step_tensors[layer] for layer in range(num_layers)], dim=0)
            gen_blocks.append(step_stack)  # each [num_layers, batch, 1, D]

        if gen_blocks:
            gen_stack = torch.cat(gen_blocks, dim=2)  # [num_layers, batch, gen_len, D]
        else:
            # No generated tokens beyond step 0: create empty tensor
            gen_stack = prompt_stack[:, :, :0, :]

        # Concatenate prompt + generated → [num_layers, batch, total_len, D]
        stacked = torch.cat([prompt_stack, gen_stack], dim=2)

        # Handle batch dimension
        if batch == 1:
            stacked = stacked.squeeze(1)  # → [num_layers, total_len, D]
            return GenerationActivations(
                hidden_states=stacked,
                prompt_len=prompt_len,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                batch_size=None,
                attention_mask=None,
            )

        # Batch > 1: keep as [batch, num_layers, total_len, D] for easy indexing
        stacked = stacked.permute(1, 0, 2, 3)  # → [batch, num_layers, total_len, D]

        # Build per-sequence valid-token mask if attention_mask is available.
        # NOTE: generated tokens are currently marked all-True.  A future
        # enhancement could use ``tokenizer.pad_token_id`` to detect
        # post-EOS padding in the generated portion and mask those out
        # for sequences that finished early.
        valid_mask: torch.BoolTensor | None = None
        if attention_mask is not None:
            # attention_mask: [batch, prompt_seq_len]
            # Expand to total_len by padding with True for generated tokens
            total_len = stacked.shape[2]
            prompt_seq_len = attention_mask.shape[1]
            gen_len = total_len - prompt_seq_len
            if gen_len > 0:
                gen_mask = torch.ones(
                    (batch, gen_len),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                valid_mask = torch.cat([attention_mask.bool(), gen_mask], dim=1)  # [batch, total_len]
            else:
                valid_mask = attention_mask.bool()

        return GenerationActivations(
            hidden_states=stacked,
            prompt_len=prompt_len,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            batch_size=batch,
            attention_mask=valid_mask,
        )

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def prompt_hidden_states(self) -> torch.FloatTensor:
        """``[num_layers, prompt_len, hidden_dim]`` — activations for prompt
        tokens only.

        When batched (``batch_size > 1``), shape is
        ``[batch_size, num_layers, prompt_len, hidden_dim]``.
        """
        if self.batch_size:
            return self.hidden_states[:, :, : self.prompt_len, :]
        return self.hidden_states[:, : self.prompt_len, :]

    @property
    def generated_hidden_states(self) -> torch.FloatTensor:
        """``[num_layers, generated_len, hidden_dim]`` — activations for
        generated tokens only.

        When batched, shape is
        ``[batch_size, num_layers, generated_len, hidden_dim]``.
        """
        if self.batch_size:
            return self.hidden_states[:, :, self.prompt_len :, :]
        return self.hidden_states[:, self.prompt_len :, :]

    @property
    def total_len(self) -> int:
        """Total number of tokens (prompt + generated)."""
        # For batched: shape is [B, L, T, D] → token dim is index 2
        # For single: shape is [L, T, D] → token dim is index 1
        return self.hidden_states.shape[2] if self.batch_size else self.hidden_states.shape[1]

    # ── Utility methods ────────────────────────────────────────────────────

    def pool_layers(self, target_layers: int) -> torch.FloatTensor:
        """Adaptive-average-pool the layer axis to a fixed grid size.

        Useful for normalizing across architectures with different layer counts
        (e.g. Qwen has 37 layers, Llama has 33).  Uses
        :func:`torch.nn.functional.adaptive_avg_pool1d` — this averages
        neighbouring layers, it does **not** truncate.

        Args:
            target_layers (:obj:`int`):
                Desired number of layers after pooling.  Must be ≤ the
                current ``num_layers``.

        Returns:
            :obj:`torch.FloatTensor`:
                Tensor of shape ``(target_layers, total_len, hidden_dim)``
                (or ``(batch_size, target_layers, total_len, hidden_dim)``
                if batched).
        """
        if target_layers > self.num_layers:
            raise ValueError(f"target_layers ({target_layers}) must be ≤ num_layers ({self.num_layers})")
        if target_layers == self.num_layers:
            return self.hidden_states

        if self.batch_size:
            B, L, T, D = self.hidden_states.shape
            # adaptive_avg_pool1d expects [N, C, L_in] — pool the layer dim
            x = self.hidden_states.permute(0, 2, 3, 1).reshape(B * T * D, 1, L)
            x = torch.nn.functional.adaptive_avg_pool1d(x, target_layers)
            x = x.reshape(B, T, D, target_layers).permute(0, 3, 1, 2)
            return x  # [B, target_layers, T, D]
        else:
            L, T, D = self.hidden_states.shape
            x = self.hidden_states.permute(1, 2, 0).reshape(T * D, 1, L)
            x = torch.nn.functional.adaptive_avg_pool1d(x, target_layers)
            x = x.reshape(T, D, target_layers).permute(2, 0, 1)
            return x  # [target_layers, T, D]

    def to(self, *args, **kwargs) -> GenerationActivations:
        """Move all tensors to a device and/or dtype.

        Delegates to :meth:`torch.Tensor.to`.  Returns a new
        :class:`GenerationActivations` with moved tensors (metadata fields
        are preserved).

        Returns:
            :class:`GenerationActivations`
        """
        return GenerationActivations(
            hidden_states=self.hidden_states.to(*args, **kwargs),
            prompt_len=self.prompt_len,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            attention_mask=(self.attention_mask.to(*args, **kwargs) if self.attention_mask is not None else None),
        )

    def __repr__(self) -> str:
        blurb = (
            f"GenerationActivations("
            f"shape={tuple(self.hidden_states.shape)}, "
            f"num_layers={self.num_layers}, "
            f"prompt_len={self.prompt_len}, "
            f"hidden_dim={self.hidden_dim}"
        )
        if self.batch_size:
            blurb += f", batch_size={self.batch_size}"
        blurb += ")"
        return blurb
