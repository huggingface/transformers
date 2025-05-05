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

import numpy as np
import torch

from ...generation import GenerationMixin
from ...utils import logging

import time
logger = logging.get_logger(__name__)


class DiaGenerationMixin(GenerationMixin):
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_tokens: int | None = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 35,
        audio_prompt: str | torch.Tensor | None = None,
        audio_prompt_path: str | None = None,
        use_cfg_filter: bool | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_tokens = self.config.data.audio_length if max_tokens is None else max_tokens
        max_delay_pattern = max(delay_pattern)
        self.model.eval()

        if audio_prompt_path:
            print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
            audio_prompt = audio_prompt_path
        if use_cfg_filter is not None:
            print("Warning: use_cfg_filter is deprecated.")

        if verbose:
            total_start_time = time.time()

        dec_state, dec_output = self._prepare_generation(text, audio_prompt, verbose)
        dec_step = dec_output.prefill_step - 1

        bos_countdown = max_delay_pattern
        eos_detected = False
        eos_countdown = -1

        if use_torch_compile:  # TODO only compile decoding steps?
            step_fn = torch.compile(self._decoder_step, mode="default")
        else:
            step_fn = self._decoder_step

        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print("generate: by using use_torch_compile=True, the first step would take long")
            start_time = time.time()

        while dec_step < max_tokens:
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step).unsqueeze(0).expand(2, -1, -1)
            pred_C = step_fn(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
            )

            if (not eos_detected and pred_C[0] == audio_eos_value) or dec_step == max_tokens - max_delay_pattern - 1:
                eos_detected = True
                eos_countdown = max_delay_pattern

            if eos_countdown > 0:
                step_after_eos = max_delay_pattern - eos_countdown
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d:
                        pred_C[i] = audio_eos_value
                    elif step_after_eos > d:
                        pred_C[i] = audio_pad_value
                eos_countdown -= 1

            bos_countdown = max(0, bos_countdown - 1)
            dec_output.update_one(pred_C, dec_step + 1, bos_countdown > 0)

            if eos_countdown == 0:
                break

            dec_step += 1
            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                print(
                    f"generate step {dec_step}: speed={86 / duration:.3f} tokens/s, realtime factor={1 / duration:.3f}x"
                )
                start_time = time.time()

        if dec_output.prefill_step >= dec_step + 1:
            print("Warning: Nothing generated")
            return None

        generated_codes = dec_output.generated_tokens[dec_output.prefill_step : dec_step + 1, :]

        if verbose:
            total_step = dec_step + 1 - dec_output.prefill_step
            total_duration = time.time() - total_start_time
            print(f"generate: total step={total_step}, total duration={total_duration:.3f}s")

        return generated_codes
