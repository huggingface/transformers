# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...generation import GenerationMixin
from ...utils import ModelOutput, logging


logger = logging.get_logger(__name__)


@dataclass
class VoxtralTtsGenerateOutput(ModelOutput):
    """
    Output of [`VoxtralTtsForTextToSpeech.generate`].

    Args:
        audio (`list[torch.Tensor]`, *optional*):
            Generated audio waveform tensors, one per batch entry. Each tensor has shape `(num_samples,)`.
        semantic_tokens (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
            Generated semantic token IDs from the flow-matching transformer.
        acoustic_values (`torch.Tensor` of shape `(batch_size, num_frames, acoustic_dim)`, *optional*):
            Generated continuous acoustic values from the flow-matching ODE.
    """

    audio: list[torch.Tensor] | None = None
    semantic_tokens: torch.LongTensor | None = None
    acoustic_values: torch.Tensor | None = None


class VoxtralTtsGenerationMixin(GenerationMixin):
    """
    Generation mixin for Voxtral TTS that orchestrates the 3-stage pipeline:

    1. **Autoregressive backbone**: Processes text + voice reference, produces hidden states with KV caching.
    2. **Flow-matching ODE** (per frame): Uses backbone hidden states to solve an Euler ODE (default 8 steps),
       producing semantic tokens and continuous acoustic values. Supports classifier-free guidance.
    3. **Codec decode**: Converts semantic tokens + acoustic values into a 24kHz audio waveform.
    """

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 4000,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float | None = None,
        n_nfe: int = 8,
        cfg_alpha: float = 1.0,
        output_audio: bool = True,
        **kwargs,
    ) -> VoxtralTtsGenerateOutput:
        r"""
        Generate audio waveform from text input and optional voice reference.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Text token IDs. Should end with `begin_audio_token_id` to signal the start of audio generation.
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_ref_frames, num_codebooks)`, *optional*):
                Voice reference audio codes from a preset voice embedding. Shape is `(B, T, 37)` where 37 =
                1 semantic + 36 acoustic codebooks.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the input sequence.
            max_new_tokens (`int`, *optional*, defaults to 4000):
                Maximum number of audio frames to generate. At 12.5 Hz frame rate, 4000 frames = 320 seconds.
            temperature (`float`, *optional*, defaults to 0.8):
                Sampling temperature for semantic token generation.
            top_k (`int`, *optional*):
                Top-k filtering for semantic token sampling.
            top_p (`float`, *optional*):
                Top-p (nucleus) filtering for semantic token sampling.
            n_nfe (`int`, *optional*, defaults to 8):
                Number of function evaluations (Euler steps) for the flow-matching ODE solver.
            cfg_alpha (`float`, *optional*, defaults to 1.0):
                Classifier-free guidance scale. Set to 1.0 to disable CFG, >1.0 to strengthen conditioning.
                Common value is 1.2 as suggested in the paper.
            output_audio (`bool`, *optional*, defaults to `True`):
                Whether to decode the generated tokens into an audio waveform via the codec model.

        Returns:
            [`VoxtralTtsGenerateOutput`] containing the generated audio, semantic tokens, and acoustic values.

        Example:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, VoxtralTtsForTextToSpeech

        >>> model = VoxtralTtsForTextToSpeech.from_pretrained("mistralai/Voxtral-4B-TTS-2603")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Voxtral-4B-TTS-2603")

        >>> input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
        >>> output = model.generate(input_ids=input_ids, max_new_tokens=100)
        >>> audio = output.audio[0]  # first batch entry, shape (num_samples,)
        ```
        """
        config = self.config
        batch_size = input_ids.shape[0] if input_ids is not None else audio_codes.shape[0]

        # --- 1. Prepare initial embeddings ---
        parts = []
        if audio_codes is not None:
            parts.append(self.backbone_model.embed_tokens(audio_codes))
        if input_ids is not None:
            parts.append(self.embed_text_tokens(input_ids))
        if not parts:
            raise ValueError("At least one of `input_ids` or `audio_codes` must be provided.")
        inputs_embeds = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        # --- 2. Backbone prefill (conditioned) ---
        backbone_outputs = self.backbone_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        past_key_values = backbone_outputs.past_key_values
        backbone_hidden = backbone_outputs.last_hidden_state[:, -1:, :]

        # --- 3. Prepare unconditional backbone for CFG ---
        uncond_cache = None
        uncond_hidden = None
        use_cfg = cfg_alpha != 1.0 and input_ids is not None
        if use_cfg:
            uncond_text_ids = torch.full_like(input_ids, config.condition_dropped_token_id)
            uncond_parts = []
            if audio_codes is not None:
                uncond_parts.append(self.backbone_model.embed_tokens(audio_codes))
            uncond_parts.append(self.embed_text_tokens(uncond_text_ids))
            uncond_embeds = torch.cat(uncond_parts, dim=1) if len(uncond_parts) > 1 else uncond_parts[0]
            uncond_outputs = self.backbone_model(inputs_embeds=uncond_embeds, use_cache=True)
            uncond_cache = uncond_outputs.past_key_values
            uncond_hidden = uncond_outputs.last_hidden_state[:, -1:, :]

        # --- 4. Generation loop ---
        fm_config = config.flow_matching_config
        acoustic_dim = fm_config.acoustic_dim
        sigma = fm_config.sigma
        sigma_max = fm_config.sigma_max

        all_semantic_tokens = []
        all_acoustic_values = []

        for frame_idx in range(max_new_tokens):
            # 4a. Run flow-matching ODE for this frame
            semantic_token, acoustic_value = self._flow_matching_step(
                backbone_hidden=backbone_hidden,
                uncond_hidden=uncond_hidden,
                n_nfe=n_nfe,
                cfg_alpha=cfg_alpha if use_cfg else 1.0,
                sigma=sigma,
                sigma_max=sigma_max,
                acoustic_dim=acoustic_dim,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            all_semantic_tokens.append(semantic_token)
            all_acoustic_values.append(acoustic_value)

            # 4b. Create next backbone input from generated codes
            clamped_semantic = torch.clamp(semantic_token, 0, config.semantic_codebook_size - 1)
            acoustic_codes_discrete = torch.clamp(
                torch.round(acoustic_value), 0, config.acoustic_codebook_size - 1
            ).long()
            frame_codes = torch.cat([clamped_semantic.unsqueeze(-1), acoustic_codes_discrete], dim=-1).unsqueeze(
                1
            )  # (B, 1, num_codebooks)

            frame_embeds = self.backbone_model.embed_tokens(frame_codes)

            # 4c. Backbone step with cache
            backbone_outputs = self.backbone_model(
                inputs_embeds=frame_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = backbone_outputs.past_key_values
            backbone_hidden = backbone_outputs.last_hidden_state

            # 4d. Update unconditional backbone if using CFG
            if use_cfg:
                uncond_outputs = self.backbone_model(
                    inputs_embeds=frame_embeds,
                    past_key_values=uncond_cache,
                    use_cache=True,
                )
                uncond_cache = uncond_outputs.past_key_values
                uncond_hidden = uncond_outputs.last_hidden_state

        # --- 5. Stack results ---
        semantic_tokens = torch.stack(all_semantic_tokens, dim=1)
        acoustic_values = torch.stack(all_acoustic_values, dim=1)

        # --- 6. Codec decode ---
        audio = None
        if output_audio:
            codec_semantic_tokens = torch.clamp(semantic_tokens, 0, config.semantic_codebook_size - 1)
            waveform = self.codec_model.decode(codec_semantic_tokens, acoustic_values)
            audio = [waveform[i, 0] for i in range(batch_size)]

        return VoxtralTtsGenerateOutput(
            audio=audio,
            semantic_tokens=semantic_tokens,
            acoustic_values=acoustic_values,
        )

    def _flow_matching_step(
        self,
        backbone_hidden: torch.Tensor,
        uncond_hidden: torch.Tensor | None = None,
        n_nfe: int = 8,
        cfg_alpha: float = 1.0,
        sigma: float = 1e-5,
        sigma_max: float = 1.0,
        acoustic_dim: int = 36,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """
        Run one flow-matching Euler ODE solve to produce a single audio frame.

        Starting from Gaussian noise x_0 ~ N(0, I), integrates the velocity field predicted by the
        flow-matching transformer over `n_nfe` Euler steps from t=sigma to t=sigma_max. Optionally
        applies classifier-free guidance by blending conditioned and unconditioned velocities.

        Returns the sampled semantic token ID and continuous acoustic values for one frame.
        """
        batch_size = backbone_hidden.shape[0]
        device = backbone_hidden.device
        dtype = backbone_hidden.dtype

        x_t = torch.randn(batch_size, 1, acoustic_dim, device=device, dtype=dtype)
        dt = (sigma_max - sigma) / n_nfe

        semantic_logits = None

        for step in range(n_nfe):
            t = sigma + (sigma_max - sigma) * step / n_nfe
            t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)

            sem_logits_cond, v_cond = self.flow_matching_transformer(
                hidden_states=backbone_hidden,
                timesteps=t_tensor,
                acoustic_embeddings=x_t,
            )

            if cfg_alpha != 1.0 and uncond_hidden is not None:
                sem_logits_uncond, v_uncond = self.flow_matching_transformer(
                    hidden_states=uncond_hidden,
                    timesteps=t_tensor,
                    acoustic_embeddings=x_t,
                )
                velocity = v_uncond + cfg_alpha * (v_cond - v_uncond)
                semantic_logits = sem_logits_uncond + cfg_alpha * (sem_logits_cond - sem_logits_uncond)
            else:
                velocity = v_cond
                semantic_logits = sem_logits_cond

            x_t = x_t + dt * velocity

        # Sample semantic token from the final step's logits
        sem_logits = semantic_logits[:, -1, :] / temperature

        if top_k is not None:
            topk_values = torch.topk(sem_logits, top_k)[0]
            sem_logits = sem_logits.masked_fill(sem_logits < topk_values[..., -1:], float("-inf"))

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(sem_logits, descending=True)
            cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            sem_logits = sem_logits.masked_fill(indices_to_remove, float("-inf"))

        probs = nn.functional.softmax(sem_logits, dim=-1)
        semantic_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        acoustic_value = x_t.squeeze(1)

        return semantic_token, acoustic_value
