"""Multi-Token Prediction (MTP) candidate generator.

MTP modules are shipped inside the main checkpoint (e.g. DeepSeek-V3 at
`model.layers.61.*`, GLM-4 MoE at `model.layers.{num_hidden_layers}.*`) but
hidden from the base model via `_keys_to_ignore_on_load_unexpected`. They are
loaded separately here, matching the base model's decoder layer class, and act
as the draft head for speculative decoding.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from torch import nn

from ...cache_utils import Cache, DynamicLayer
from ...masking_utils import create_causal_mask
from ..candidate_generator import CandidateGenerator


if TYPE_CHECKING:
    from ...configuration_utils import PreTrainedConfig
    from ...modeling_utils import PreTrainedModel


class MTPSharedHead(nn.Module):
    """Final projection inside an MTP module: RMSNorm + linear over vocab."""

    def __init__(self, config: PreTrainedConfig, rmsnorm_cls: type[nn.Module]):
        super().__init__()
        self.norm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden_states))


class MTPLayer(nn.Module):
    """One MTP depth (DeepSeek-V3 spec).

    Combines the previous hidden state `h_{t+k}` and the embedding of the
    next drafted token `x_{t+k+1}`, projects them down with `eh_proj`, runs
    a standard decoder block, then produces logits for position `t+k+2`.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        decoder_layer: nn.Module,
        rmsnorm_cls: type[nn.Module],
    ):
        super().__init__()
        self.enorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_block = decoder_layer
        self.shared_head = MTPSharedHead(config, rmsnorm_cls)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_cat = torch.cat([self.enorm(inputs_embeds), self.hnorm(previous_hidden_state)], dim=-1)
        hidden_states = self.mtp_block(
            self.eh_proj(h_cat),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.shared_head(hidden_states)
        return hidden_states, logits


class MTPCandidateGenerator(nn.Module, CandidateGenerator):
    """Speculative-decoding draft head built from a model's MTP modules.

    Holds `config.num_nextn_predict_layers` MTP depths, each a full transformer
    block surrounded by projection/norm/head machinery (see `MTPLayer`). The
    generator shares the base model's KV cache: each MTP depth's `mtp_block`
    writes to `past_key_values[num_hidden_layers + k]`, extending the cache
    in place when needed.

    Constructed either directly (`MTPCandidateGenerator(base_model)`) or via
    `from_pretrained`, which pulls MTP-specific keys out of the checkpoint.
    """

    def __init__(self, base_model: PreTrainedModel, num_mtp: int | None = None):
        super().__init__()
        config = base_model.config
        num_mtp = num_mtp if num_mtp is not None else getattr(config, "num_nextn_predict_layers", 0)
        if num_mtp <= 0:
            raise ValueError(
                "MTPCandidateGenerator requires `config.num_nextn_predict_layers > 0` "
                "or an explicit `num_mtp` argument."
            )

        inner = base_model.base_model if hasattr(base_model, "base_model_prefix") else base_model
        layers = getattr(inner, "layers", None) or getattr(getattr(inner, "model", None), "layers", None)
        if layers is None or len(layers) == 0:
            raise ValueError("Could not locate `layers` on the provided base model.")

        sample_layer = layers[0]
        decoder_cls = type(sample_layer)
        rmsnorm_cls = type(sample_layer.input_layernorm)

        self.num_mtp = num_mtp
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [MTPLayer(config, decoder_cls(config, config.num_hidden_layers + k), rmsnorm_cls) for k in range(num_mtp)]
        )
        # Weak handle for `get_candidates` — re-used for embed_tokens, rotary_emb, cache masks.
        self._base_ref = base_model
        self._config = config

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        base_model: PreTrainedModel,
        num_mtp: int | None = None,
        **kwargs,
    ) -> MTPCandidateGenerator:
        """Load MTP weights out of the base checkpoint.

        Reads the same safetensors shards as the main model, keeps only the
        keys under `model.layers.{num_hidden_layers + k}.*`, remaps them onto
        `MTPLayer`, and returns a fully-initialised generator.
        """
        from ...modeling_utils import _get_resolved_checkpoint_files  # lazy

        generator = cls(base_model, num_mtp=num_mtp)
        num_mtp = generator.num_mtp
        num_base = generator.num_hidden_layers

        # Resolve + load the checkpoint's state dict.
        resolved_files, _ = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=kwargs.pop("subfolder", ""),
            variant=kwargs.pop("variant", None),
            gguf_file=None,
            from_tf=False,
            from_flax=False,
            use_safetensors=True,
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("token", None),
            user_agent={"file_type": "model", "framework": "pytorch"},
            revision=kwargs.pop("revision", "main"),
            commit_hash=None,
        )

        mtp_layer_ids = {num_base + k for k in range(num_mtp)}
        merged: dict[str, torch.Tensor] = {}
        from safetensors.torch import load_file

        for path in resolved_files:
            shard = load_file(path)
            for key, tensor in shard.items():
                m = re.match(r"^(?:model\.)?layers\.(\d+)(?:\.(.*))?$", key)
                if m is None:
                    continue
                layer_id = int(m.group(1))
                if layer_id not in mtp_layer_ids:
                    continue
                sub = m.group(2) or ""
                k = layer_id - num_base
                mapped = f"layers.{k}.{sub}" if sub else f"layers.{k}"
                merged[mapped] = tensor

        missing, unexpected = generator.load_state_dict(merged, strict=False)
        if unexpected:
            raise ValueError(f"MTP checkpoint contained unexpected keys: {unexpected}")
        if missing:
            # Non-fatal — the checkpoint may tie `shared_head.head` to `lm_head`; surface to caller.
            import warnings

            warnings.warn(
                f"MTP generator loaded with {len(missing)} missing keys; some MTP parameters "
                "will use their random initialization. First few: " + ", ".join(missing[:5]),
                stacklevel=2,
            )
        return generator

    # ------------------------------------------------------------------
    # CandidateGenerator interface
    # ------------------------------------------------------------------
    def get_candidates(
        self,
        input_ids: torch.LongTensor,
        *,
        previous_hidden_state: torch.Tensor,
        past_key_values: Cache,
        first_token: torch.LongTensor,
        position_offset: int,
        logits_processor=None,
        do_sample: bool = False,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Draft `num_mtp` tokens beyond `first_token`.

        Returns `(candidate_ids, candidate_logits)` where `candidate_ids` has
        shape `(1, num_mtp + 1)` starting with `first_token`, and
        `candidate_logits` has shape `(1, num_mtp, vocab)` — one logit
        distribution per MTP depth (i.e. for the tokens at `position_offset + 1`
        through `position_offset + num_mtp`).
        """
        drafts = [first_token]
        logits_list: list[torch.Tensor] = []
        prev_hidden = previous_hidden_state
        embed_tokens = self._base_ref.get_input_embeddings()
        rotary_emb = getattr(self._base_ref, "rotary_emb", None) or self._base_ref.model.rotary_emb
        for depth in range(self.num_mtp):
            layer_idx = self.num_hidden_layers + depth
            if hasattr(past_key_values, "layers"):
                while len(past_key_values.layers) <= layer_idx:
                    past_key_values.layers.append(DynamicLayer())
            tok = drafts[depth]
            inputs_embeds = embed_tokens(tok)
            pos = torch.tensor([[position_offset + depth]], device=tok.device, dtype=torch.long)
            position_embeddings = rotary_emb(inputs_embeds, position_ids=pos)
            causal_mask = create_causal_mask(
                config=self._config,
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=past_key_values,
                position_ids=pos,
            )
            prev_hidden, step_logits = self.layers[depth](
                inputs_embeds=inputs_embeds,
                previous_hidden_state=prev_hidden,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            vec = step_logits[:, 0, :].to(dtype=torch.float32)
            if logits_processor is not None:
                vec = logits_processor(torch.cat([input_ids] + drafts, dim=1), vec)
            logits_list.append(vec)
            if do_sample:
                drafted = torch.multinomial(nn.functional.softmax(vec, dim=-1), num_samples=1)
            else:
                drafted = torch.argmax(vec, dim=-1, keepdim=True)
            drafts.append(drafted)

        candidate_ids = torch.cat(drafts, dim=1)
        candidate_logits = torch.stack(logits_list, dim=1)
        return candidate_ids, candidate_logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        # Fixed K = num_mtp; no heuristic schedule.
        return
