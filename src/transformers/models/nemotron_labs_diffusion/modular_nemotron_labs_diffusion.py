# Copyright 2026 NVIDIA Corporation and the HuggingFace Inc. team. All rights reserved.
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
from typing import Literal

import torch
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache, DynamicLayer
from ...masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    create_bidirectional_mask,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.output_capturing import capture_outputs
from ..ministral3.configuration_ministral3 import Ministral3Config
from ..ministral3.modeling_ministral3 import Ministral3ForCausalLM, Ministral3Model, Ministral3PreTrainedModel


NemotronAttentionMode = Literal["causal", "bidirectional", "block", "block_diff"]


@auto_docstring(checkpoint="nvidia/Nemotron-Labs-Diffusion-8B")
@strict
class NemotronLabsDiffusionConfig(Ministral3Config):
    r"""
    mask_token_id (`int`, *optional*, defaults to 100):
        Token ID used for masking during diffusion generation and training.
    dlm_paradigm (`str`, *optional*, defaults to `"bidirectional"`):
        Diffusion language modeling paradigm (`"bidirectional"`, `"autoregressive"`, or `"block_diff"`).
    block_size (`int`, *optional*, defaults to 32):
        Block size used for block diffusion decoding and training.
    dlm_loss_weight (`float`, *optional*):
        Weight applied to the diffusion language modeling loss.
    ar_loss_weight (`float`, *optional*, defaults to 1.0):
        Weight applied to the autoregressive loss in block diffusion training.
    dp_varying_mask_ratio (`bool`, *optional*, defaults to `False`):
        Whether to use varying mask ratios across data-parallel ranks during training.

    Example:

    ```python
    >>> from transformers import AutoModelForCausalLM, NemotronLabsDiffusionConfig

    >>> # Initializing a Nemotron-Labs-Diffusion style configuration
    >>> configuration = NemotronLabsDiffusionConfig()

    >>> # Initializing a model from that configuration
    >>> model = AutoModelForCausalLM.from_config(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nemotron_labs_diffusion"

    use_cache: bool = False
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = 11
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    mask_token_id: int | None = 100
    dlm_paradigm: str = "bidirectional"
    block_size: int = 32
    dlm_loss_weight: float | None = None
    ar_loss_weight: float = 1.0
    dp_varying_mask_ratio: bool = False

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.attention_bias:
            raise ValueError(
                "NemotronLabsDiffusion with `attention_bias=True` is not supported, because the Ministral3 "
                "implementation uses bias-free attention projections."
            )
        if self.mlp_bias:
            raise ValueError(
                "NemotronLabsDiffusion with `mlp_bias=True` is not supported, because the Ministral3 "
                "implementation uses bias-free MLP projections."
            )
        if self.dlm_paradigm not in {"autoregressive", "bidirectional", "block_diff"}:
            raise ValueError(f"Unsupported Nemotron diffusion paradigm: {self.dlm_paradigm}")
        if self.block_size < 1:
            raise ValueError("`block_size` must be positive.")


class _ReadOnlyDynamicCache(Cache):
    """Expose committed KV and the current canvas without storing the canvas."""

    def __init__(self, cache: Cache):
        if (
            not isinstance(cache, DynamicCache)
            or cache.offloading
            or any(type(layer) is not DynamicLayer for layer in cache.layers)
        ):
            raise ValueError("Reading a prefix with `use_cache=False` requires a full-attention DynamicCache.")
        super().__init__(layers=cache.layers)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.layers):
            return key_states, value_states
        layer = self.layers[layer_idx]
        if not isinstance(layer, DynamicLayer) or not layer.is_initialized:
            return key_states, value_states
        return torch.cat((layer.keys, key_states), dim=-2), torch.cat((layer.values, value_states), dim=-2)


def _two_stream_mask(stream_length: int, block_size: int):
    def mask(batch_idx, head_idx, q_idx, kv_idx):
        noisy_query = q_idx < stream_length
        noisy_key = kv_idx < stream_length
        query_block = q_idx % stream_length // block_size
        key_block = kv_idx % stream_length // block_size
        return (noisy_query & noisy_key & (query_block == key_block)) | (
            ~noisy_key & torch.where(noisy_query, key_block < query_block, kv_idx <= q_idx)
        )

    return mask


def _sample_probabilities(probabilities: torch.Tensor) -> torch.LongTensor:
    return torch.multinomial(probabilities.flatten(0, -2), 1).reshape(probabilities.shape[:-1])


def _sample_ar(logits: torch.Tensor, temperature: float) -> torch.LongTensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    return _sample_probabilities(torch.softmax(logits / temperature, dim=-1))


def _confidence_reveal(confidence: torch.Tensor, pending: torch.Tensor, threshold: float) -> torch.BoolTensor:
    confidence = confidence.masked_fill(~pending, -torch.inf)
    best = confidence.argmax(dim=-1, keepdim=True)
    reveal = pending & (confidence >= threshold)
    return reveal.scatter_(1, best, pending.gather(1, best))


class NemotronLabsDiffusionPreTrainedModel(Ministral3PreTrainedModel):
    pass


@auto_docstring
class NemotronLabsDiffusionModel(Ministral3Model):
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        *,
        attention_mode: NemotronAttentionMode | None = None,
        block_sequence_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        attention_mode (`str`, *optional*):
            Defaults to the configured diffusion paradigm, with `"autoregressive"` selecting `"causal"`.
            In `"bidirectional"` mode the current tokens also see all other tokens in their canvas.
            In `"block"` mode, supply `block_sequence_ids` to make each block bidirectional within a
            causal sequence, or pass just the current canvas with a committed prefix cache.
            In `"block_diff"` mode the input contains concatenated noisy and clean streams of equal length.
            Pass one stream of `position_ids` and, if needed, a padding mask for one stream.
        block_sequence_ids (`torch.LongTensor`, *optional*):
            Group IDs for a block mask. Use `-1` for strictly causal positions and a distinct nonnegative ID
            for each contiguous bidirectional block.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        backend = self.config._attn_implementation
        has_mask_interface = backend in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
        if attention_mode is None:
            if not has_mask_interface or self.config.dlm_paradigm == "autoregressive":
                attention_mode = "causal"
            else:
                attention_mode = self.config.dlm_paradigm
        if attention_mode not in {"causal", "bidirectional", "block", "block_diff"}:
            raise ValueError(f"Unknown attention mode: {attention_mode}")
        if block_sequence_ids is not None and attention_mode != "block":
            raise ValueError("`block_sequence_ids` requires `attention_mode='block'`.")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            use_cache = False
        if attention_mode == "block_diff" and (use_cache or cache is not None):
            raise ValueError("Two-stream block diffusion does not support caching.")
        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        seq_length = inputs_embeds.shape[1]
        if position_ids is None and attention_mode != "block_diff":
            past_length = cache.get_seq_length() if cache is not None else 0
            position_ids = (torch.arange(seq_length, device=inputs_embeds.device) + past_length).unsqueeze(0)
        rope_position_ids = position_ids

        uses_custom_mask = block_sequence_ids is not None or attention_mode == "block_diff"
        if uses_custom_mask and backend not in {"eager", "sdpa", "flex_attention"}:
            raise ValueError("Block diffusion masks require eager, SDPA, or FlexAttention.")

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache": cache,
            "position_ids": position_ids,
        }
        if attention_mode == "block_diff":
            if seq_length < 2 or seq_length % 2:
                raise ValueError("Two-stream block diffusion requires equal-length noisy and clean streams.")
            stream_length = seq_length // 2
            # Positions for rotary embeddings repeat; query scaling uses the flat stream offset.
            position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0)
            if rope_position_ids is None:
                rope_position_ids = position_ids[:, :stream_length]
            if rope_position_ids.shape[-1] != stream_length:
                raise ValueError("`position_ids` must contain one stream of positions for block diffusion.")
            rope_position_ids = torch.cat((rope_position_ids, rope_position_ids), dim=-1)
            mask_kwargs["position_ids"] = position_ids
            if attention_mask is not None:
                if attention_mask.ndim != 2 or attention_mask.shape[-1] != stream_length:
                    raise ValueError("Two-stream block diffusion expects a 2D padding mask for one stream.")
                mask_kwargs["attention_mask"] = torch.cat((attention_mask, attention_mask), dim=-1)
            layer_mask = create_bidirectional_mask(
                **mask_kwargs, and_mask_function=_two_stream_mask(stream_length, self.config.block_size)
            )
        elif attention_mode == "causal":
            mask_function = (
                create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
            )
            layer_mask = mask_function(**mask_kwargs)
        elif block_sequence_ids is not None:
            layer_mask = create_causal_mask(**mask_kwargs, block_sequence_ids=block_sequence_ids)
        else:
            layer_mask = create_bidirectional_mask(**mask_kwargs)

        layer_cache = _ReadOnlyDynamicCache(cache) if cache is not None and not use_cache else cache
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=rope_position_ids)
        kwargs["is_causal"] = attention_mode == "causal"
        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                cache=layer_cache,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states), cache=cache if use_cache else None)


@auto_docstring
class NemotronLabsDiffusionForCausalLM(Ministral3ForCausalLM):
    _tied_weights_keys = AttributeError()

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        *,
        attention_mode: NemotronAttentionMode = "causal",
        block_sequence_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        attention_mode (`str`, *optional*, defaults to `"causal"`):
            Use causal next-token prediction by default. Select `"bidirectional"` for same-position
            canvas predictions, or `"block"` for a bidirectional canvas over a causal prefix.
        block_sequence_ids (`torch.LongTensor`, *optional*):
            Group IDs for `attention_mode="block"`. Use `-1` for causal positions, and a distinct
            nonnegative ID for each contiguous bidirectional block.
        """
        if labels is not None and attention_mode != "causal":
            raise ValueError(
                "Diffusion training with `labels` is unsupported. Use causal attention for next-token loss."
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            attention_mode=attention_mode,
            block_sequence_ids=block_sequence_ids,
            **kwargs,
        )
        logits = self.lm_head(slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep))
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _generation_canvas_parameters(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int,
        block_length: int | None,
        temperature: float,
        mask_token_id: int | None,
    ) -> tuple[int, int]:
        block_length = self.config.block_size if block_length is None else block_length
        mask_token_id = self.config.mask_token_id if mask_token_id is None else mask_token_id
        if prompt_ids.ndim != 2 or not prompt_ids.numel():
            raise ValueError("`prompt_ids` must contain at least one token per example.")
        if max_new_tokens < 0 or block_length < 1 or temperature < 0:
            raise ValueError("Token budget and temperature must be nonnegative, and block length must be positive.")
        if mask_token_id is None or not 0 <= mask_token_id < self.config.vocab_size:
            raise ValueError("Diffusion generation requires a mask token in the model vocabulary.")
        return block_length, mask_token_id

    @torch.no_grad()
    def diffusion_generate(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        block_length: int | None = None,
        threshold: float | None = None,
        causal_context: bool = True,
        temperature: float = 0.0,
        eos_token_id: int | list[int] | None = None,
        max_thinking_tokens: int | None = None,
        end_think_token_id: int | None = None,
    ) -> tuple[torch.LongTensor, int]:
        """Generate blocks by iteratively revealing masked tokens based on confidence.

        With `causal_context=True`, prefill and commit use causal attention and seed each new block
        with the next autoregressive token. Denoising uses bidirectional attention and reads the
        committed KV cache. Returns the prompt and generated tokens, and the number of forward
        evaluations after prefill. Input batches must have unpadded prompts of equal length.
        """
        block_length, mask_token_id = self._generation_canvas_parameters(
            prompt_ids, max_new_tokens, block_length, temperature, self.config.mask_token_id
        )
        if max_new_tokens == 0:
            return prompt_ids, 0
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_tokens = None if eos_token_id is None else prompt_ids.new_tensor(eos_token_id)

        context_mode = "causal" if causal_context else "bidirectional"
        output = self(prompt_ids, use_cache=True, logits_to_keep=1, attention_mode=context_mode)
        cache = output.cache
        batch_size, prompt_length = prompt_ids.shape
        tokens = prompt_ids.new_full((batch_size, prompt_length + max_new_tokens), mask_token_id)
        tokens[:, :prompt_length] = prompt_ids
        generated = tokens[:, prompt_length:]
        nfe = 0
        for start in range(0, max_new_tokens, block_length):
            block = generated[:, start : start + block_length]
            if causal_context:
                block[:, :1] = _sample_ar(output.logits, temperature)
            if (
                max_thinking_tokens is not None
                and end_think_token_id is not None
                and start + block.shape[1] > max_thinking_tokens
            ):
                needs_marker = ~(generated[:, :start] == end_think_token_id).any(dim=-1)
                block[needs_marker, max(0, max_thinking_tokens - start)] = end_think_token_id

            pending = block == mask_token_id
            if causal_context:
                pending[:, 0] = False
            steps = block.shape[1]
            base, remainder = pending.sum(dim=-1) // steps, pending.sum(dim=-1) % steps
            for step in range(steps):
                if not pending.any():
                    break

                logits = self(block, cache=cache, use_cache=False, attention_mode="bidirectional").logits
                nfe += 1
                if temperature > 0:
                    noise = torch.rand_like(logits, dtype=torch.float64)
                    predictions = (logits.double() - temperature * (-noise.log()).log()).argmax(dim=-1)
                else:
                    predictions = logits.argmax(dim=-1)
                confidence = torch.softmax(logits, dim=-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
                if threshold is None:
                    confidence = confidence.masked_fill(~pending, -torch.inf)
                    reveal = torch.zeros_like(pending)
                    for row, count in enumerate((base + (step < remainder)).tolist()):
                        reveal[row, confidence[row].topk(count).indices] = True
                else:
                    reveal = _confidence_reveal(confidence, pending, threshold)
                block[reveal] = predictions[reveal]
                pending &= ~reveal

                if eos_tokens is not None:
                    reached_eos = torch.isin(block, eos_tokens).cumsum(dim=-1).bool()
                    if not (pending & ~reached_eos).any():
                        break

            output = self(block, cache=cache, use_cache=True, logits_to_keep=1, attention_mode=context_mode)
            nfe += 1
            if eos_tokens is not None:
                eos = torch.isin(generated[:, : start + block.shape[1]], eos_tokens)
                if eos.any(dim=-1).all():
                    end = eos.long().argmax(dim=-1).max().item() + 1
                    return tokens[:, : prompt_length + end], nfe
        return tokens, nfe

    @torch.no_grad()
    def linear_spec_generate(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        block_length: int | None = None,
        temperature: float = 0.0,
        mask_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        max_thinking_tokens: int | None = None,
        end_think_token_id: int | None = None,
        threshold: float = 0.0,
    ) -> tuple[torch.LongTensor, int]:
        """Generate a sequence using bidirectional drafts and autoregressive verification.

        Accept the longest consecutive match and one autoregressive token per block. Unmerged PEFT
        adapters apply to draft forwards only. Returns the prompt and generated tokens, and the number
        of forward evaluations including prefill. Requires one unpadded prompt.
        """
        block_length, mask_token_id = self._generation_canvas_parameters(
            prompt_ids, max_new_tokens, block_length, temperature, mask_token_id
        )
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch size one.")
        if max_new_tokens == 0:
            return prompt_ids, 0
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_tokens = None if eos_token_id is None else prompt_ids.new_tensor(eos_token_id)

        adapters = [
            (module, module._disable_adapters) for module in self.modules() if hasattr(module, "_disable_adapters")
        ]
        if any(getattr(module, "merged", False) for module, _ in adapters):
            raise ValueError("Linear speculative decoding requires unmerged PEFT adapters.")

        def disable_adapters(disabled: bool):
            for module, _ in adapters:
                module._disable_adapters = disabled

        try:
            disable_adapters(True)
            output = self(prompt_ids, use_cache=True, logits_to_keep=1)
            cache = output.cache
            next_token = _sample_ar(output.logits, temperature)
            generated = [next_token]
            total_generated = nfe = 1
            if eos_tokens is not None and torch.isin(next_token, eos_tokens).any():
                return torch.cat((prompt_ids, next_token), dim=-1), nfe

            while total_generated < max_new_tokens:
                block = prompt_ids.new_full((1, block_length), mask_token_id)
                block[:, :1] = next_token
                pending = torch.ones_like(block, dtype=torch.bool)
                pending[:, 0] = False

                disable_adapters(False)
                while pending.any():
                    logits = self(block, cache=cache, use_cache=False, attention_mode="bidirectional").logits
                    nfe += 1
                    if threshold > 0:
                        probabilities = torch.softmax(logits / temperature if temperature > 0 else logits, dim=-1)
                        draft = _sample_probabilities(probabilities) if temperature > 0 else logits.argmax(dim=-1)
                        confidence = probabilities.gather(-1, draft.unsqueeze(-1)).squeeze(-1)
                        reveal = _confidence_reveal(confidence, pending, threshold)
                    else:
                        draft = _sample_ar(logits, temperature)
                        reveal = pending
                    block[reveal] = draft[reveal]
                    pending = pending & ~reveal

                disable_adapters(True)
                verified = _sample_ar(self(block, cache=cache, use_cache=True).logits, temperature)
                nfe += 1
                matches = (verified[:, :-1] == block[:, 1:]).long().cumprod(dim=-1).sum().item()
                accepted = min(matches + 1, max_new_tokens - total_generated)
                eos_found = False
                if eos_tokens is not None:
                    eos = torch.isin(verified[0, :accepted], eos_tokens).nonzero()
                    if len(eos):
                        accepted = eos[0, 0].item() + 1
                        eos_found = True
                cache.crop(accepted - block_length)
                accepted_tokens = verified[:, :accepted]
                generated.append(accepted_tokens)
                total_generated += accepted
                if eos_found:
                    break

                next_token = accepted_tokens[:, -1:]
                if (
                    max_thinking_tokens is not None
                    and end_think_token_id is not None
                    and total_generated > max_thinking_tokens
                    and not (torch.cat(generated, dim=-1) == end_think_token_id).any()
                ):
                    next_token = prompt_ids.new_full((1, 1), end_think_token_id)
        finally:
            for module, was_disabled in adapters:
                module._disable_adapters = was_disabled

        return torch.cat((prompt_ids, *generated), dim=-1), nfe


__all__ = [
    "NemotronLabsDiffusionConfig",
    "NemotronLabsDiffusionForCausalLM",
    "NemotronLabsDiffusionModel",
    "NemotronLabsDiffusionPreTrainedModel",
]
