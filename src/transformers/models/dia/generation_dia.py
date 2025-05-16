# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import time

import numpy as np
import torch

from ...cache_utils import DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...utils import logging
from .processing_dia import build_revert_indices, revert_audio_delay


logger = logging.get_logger(__name__)

"""
tensor([[568, 232, 852, 626, 467, 403, 667, 855, 411],
        [568, 232, 852, 626, 920, 115, 667, 855, 411]], device='cuda:0')
pred_BxC
tensor([[ 568,  151,  208,  289,  671,  266, 1024,  186,  651],
        [ 568,  778,  208,  289,   80,  266, 1024,  186,  328]],
"""


@torch.no_grad()
@torch.inference_mode()
def decode(
    model,
    audio_codes,
):
    """
    Decodes the given frames into an output audio waveform
    """
    for i, audio_code in enumerate(audio_codes):
        try:
            audio_values = model.quantizer.from_codes(audio_code.unsqueeze(0).clone())
            audio_values = model.decode(audio_values[0])
            import soundfile as sf

            sf.write(f"output_{time.time()}.wav", audio_values.cpu().numpy()[0], 44100)
            return audio_values
        except Exception as e:
            print(f"Error in decode method: {str(e)}")
            raise


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class DiaGenerationMixin(GenerationMixin):
    def _load_dac_model(self):
        try:
            import dac

            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    @torch.inference_mode()
    def generate(
        self,
        input_ids: str,
        attention_mask: torch.Tensor | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.3,
        top_p: float = 0.95,
        cfg_scale=3,
        cfg_filter_top_k: int = 35,
        use_torch_compile: bool = False,
        audio_prompt: str | torch.Tensor | None = None,
        audio_prompt_path: str | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        audio_eos_value = self.generation_config.eos_token_id
        audio_pad_value = self.generation_config.pad_token_id
        delay_pattern = self.config.delay_pattern
        max_tokens = self.generation_config.max_new_tokens or 20
        max_delay_pattern = max(delay_pattern)

        if verbose:
            total_start_time = time.time()

        bos_countdown = max_delay_pattern
        eos_detected = False
        eos_countdown = -1

        if use_torch_compile:  # TODO only compile decoding steps?
            step_fn = torch.compile(self.__call__, mode="default")
        else:
            step_fn = self.__call__

        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print("generate: by using use_torch_compile=True, the first step would take long")
            start_time = time.time()

        dec_step = 0
        cache_position = None
        audio_codes = None
        generated_codes = []
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        while dec_step < max_tokens:
            decoder_outputs = self(
                input_ids=input_ids,
                audio_codes=audio_codes,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                input_audio_codes=audio_prompt,
            )[0]
            logits = decoder_outputs.view(-1, 2, *decoder_outputs.shape[1:])

            uncond_logits_BxCxV = logits[:, 0, :]  # Shape [B, C, V]
            cond_logits_BxCxV = logits[:, 1, :]  # Shape [B, C, V]
            logits_CxV = cond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)
            logits_CxV = logits_CxV.view(-1, logits_CxV.shape[-1])  # Shape [B, C, V]
            # logits_CxV[:, audio_eos_value + 1 :] = -torch.inf
            # logits_CxV[1:, audio_eos_value:] = -torch.inf

            pred_C = _sample_next_token(
                logits_CxV.float().clone(),
                temperature=0,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
            )
            pred_C = pred_C.view(-1, 9)

            if dec_step < 9:
                audio_codes = pred_C[:, :1].clone()
            else:
                audio_codes = pred_C.clone()
            generated_codes += [audio_codes]
            cache_position = torch.tensor([dec_step + 1], device=input_ids.device).unsqueeze(0)

            if eos_countdown == 0:
                break

            dec_step += 1
            if dec_step == 2:
                break

        # sanity :

        return self._generate_output(generated_codes)

    def _generate_output(self, generated_codes: torch.Tensor) -> np.ndarray:
        num_channels = self.config.decoder_config.num_channels
        generated_codes = torch.stack(generated_codes, dim=0).transpose(1, 0)
        seq_length = generated_codes.shape[1]
        delay_pattern = self.config.delay_pattern
        audio_pad_value = self.config.pad_token_id
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=generated_codes.shape[0],
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes,
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        audio = decode(self.dac_model, codebook.transpose(1, 2))

        return audio.squeeze().cpu().numpy()
