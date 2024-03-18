# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
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

import collections
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Union

import numpy as np
import scipy
import torch

from ..utils import ModelOutput, logging
from .logits_process import WatermarkLogitsProcessor


logger = logging.get_logger(__name__)


@dataclass
class WatermarkDetectorOutput(ModelOutput):
    """
    Outputs of a watermark detector, when using non-beam methods.

    Args:
        num_tokens_scored (np.array of shape (batch_size)):
            Array containing the number of tokens scored for each batch.
        num_green_tokens (np.array of shape (batch_size)):
            Array containing the number of green tokens for each batch.
        green_fraction (np.array of shape (batch_size)):
            Array containing the fraction of green tokens for each batch.
        z_score (np.array of shape (batch_size)):
            Array containing the z-score for each batch.
        p_value (np.array of shape (batch_size)):
            Array containing the p-value for each batch.
        prediction (np.array of shape (batch_size)), *optional* (returned when return_predictions=True is passed):
            Array containing predictions for each batch.
        confidence (np.array of shape (batch_size)), *optional* (returned when return_predictions=True is passed):
            Array containing confidence scores for each batch.
    """

    num_tokens_scored: np.array = None
    num_green_tokens: np.array = None
    green_fraction: np.array = None
    z_score: np.array = None
    p_value: np.array = None
    prediction: Optional[np.array] = None
    confidence: Optional[np.array] = None


class WatermarkDetector:

    r"""
    Detector for detection of watermark generated text.  The detector needs to be given the exact same settings that were
    given during text generation  to replicate the watermark greenlist generation and so detect the watermark. This includes
    the correct device that was used during text generation, the correct watermarking arguments and the correct tokenizer vocab size.
    The code was absed on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

    See [the paper](https://arxiv.org/abs/2306.04634) for more information.

    Args:
        vocab_size (`int`):
            The model tokenizer's vocab_size.
        device (`str`):
            The device which was used during watermarked text generation.
        watermarking_args (`Dict`, *optional*):
            The exact same arguments used when generating watermarked text.
        ignore_repeated_ngrams (`bool`):
            Whether to count every unique ngram only once or not.

    Return:
            [`~generation.WatermarkDetectorOutput`] or `torch.FloatTensor`: A [`~utils.ModelOutput`] (if `return_dict=True`
            otherwise a `torch.FloatTensor`.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> from transformers.generation import WatermarkDetector

    >>> model_id = "openai-community/gpt2"
    >>> model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda:0")
    >>> tok = AutoTokenizer.from_pretrained(model_id)
    >>> tok.pad_token_id = tok.eos_token_id
    >>> tok.padding_side = "left"

    >>> inputs = tok(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt").to("cuda:0")
    >>> input_len = inputs["input_ids"].shape[-1]

    >>> # first generate text with watermark and without
    >>> args = {"bias": 2.5, "context_width": 3, "seeding_scheme": "selfhash", "greenlist_ratio": 0.25, "hashing_key": 15485863}
    >>> out_watermarked = model.generate(**inputs, watermarking_args=args, do_sample=False, max_length=20)
    >>> out = model.generate(**inputs, do_sample=False, max_length=20)

    >>> # now we can instantiate the detector and check the generated text
    >>> detector = WatermarkDetector(bos_token_id=tok.bos_token_id, vocab_size=tok.vocab_size, device="cuda:0", watermarking_args=args)
    >>> detection_out_watermarked = detector(out_watermarked[:, input_len:], return_dict=True)
    >>> detection_out = detector(out[:, input_len:], return_dict=True)
    >>> detection_out_watermarked["prediction"]
    array([ True,  True])

    >>> detection_out["prediction"]
    array([ False,  False])
    ```
    """

    def __init__(
        self,
        bos_token_id: int,
        device: str,
        vocab_size: int,
        watermarking_args: Dict,
        ignore_repeated_ngrams: bool = False,
    ):
        self.bos_token_id = bos_token_id
        self.ignore_repeated_ngrams = ignore_repeated_ngrams
        self.greenlist_ratio = watermarking_args["greenlist_ratio"]
        self.processor = WatermarkLogitsProcessor(vocab_size=vocab_size, device=device, **watermarking_args)

    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: torch.LongTensor, target: int):
        """Expensive re-seeding and sampling is cached."""
        greenlist_ids = self.processor._get_greenlist_ids(prefix)
        return target in greenlist_ids

    def _score_ngrams_in_passage(self, input_ids: torch.LongTensor):
        batch_size, seq_length = input_ids.shape
        selfhash = int(self.processor.seeding_scheme == "selfhash")
        n = self.processor.context_width + 1 - selfhash
        indices = torch.arange(n).unsqueeze(0) + torch.arange(seq_length - n + 1).unsqueeze(1)
        ngram_tensors = input_ids[:, indices]

        num_tokens_scored_batch = np.zeros(batch_size)
        green_token_count_batch = np.zeros(batch_size)
        for batch_idx in range(ngram_tensors.shape[0]):
            frequencies_table = collections.Counter(ngram_tensors[batch_idx])
            ngram_to_watermark_lookup = {}
            for ngram_example in frequencies_table.keys():
                prefix = ngram_example if selfhash else ngram_example[:-1]
                target = ngram_example[-1]
                ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(prefix, target)

            if self.ignore_repeated_ngrams:
                # counts a green/red hit once per unique ngram.
                # num total tokens scored becomes the number unique ngrams.
                num_tokens_scored_batch[batch_idx] = len(frequencies_table.keys())
                green_token_count_batch[batch_idx] = sum(ngram_to_watermark_lookup.values())
            else:
                num_tokens_scored_batch[batch_idx] = sum(frequencies_table.values())
                green_token_count_batch[batch_idx] = sum(
                    freq * outcome
                    for freq, outcome in zip(frequencies_table.values(), ngram_to_watermark_lookup.values())
                )
        return num_tokens_scored_batch, green_token_count_batch

    def _compute_z_score(self, green_token_count: np.array, total_num_tokens: np.array) -> np.array:
        expected_count = self.greenlist_ratio
        numer = green_token_count - expected_count * total_num_tokens
        denom = np.sqrt(total_num_tokens * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def __call__(
        self,
        input_ids: torch.LongTensor,
        z_threshold: float = 3.0,
        return_prediction: bool = True,
        return_dict: bool = False,
    ) -> Union[WatermarkDetectorOutput, np.array]:
        """
        Args:
        input_ids (`torch.LongTensor`):
            The watermark generated text. It is advised to remove the prompt, which can affect the detection.
        z_threshold (`Dict`, *optional*):
            Changing this threshold will change the sensitivity of the detector.
        return_prediction (`bool`):
            Whether to return user-friendly predictions as a boolean array or not.
        return_dict (`bool`):
            Whether to return `~generation.WatermarkDetectorOutput` or not. If not it will return `z_score` if
            `return_prediction=False` else `prediction`,

        Return:
            [`~generation.WatermarkDetectorOutput`] or `np.array`: A [`~utils.ModelOutput`] (if `return_dict=True`
            otherwise a `np.array`.

        """

        if input_ids.shape[-1] - self.processor.context_width < 1:
            raise ValueError(
                f"Must have at least `1` token to score after the first "
                f"min_prefix_len={self.processor.context_width} tokens required by the seeding scheme."
            )
        # Let;s assume that if one batch start with `bos`, all batched also do
        if input_ids[0, 0] == self.bos_token_id:
            input_ids = input_ids[:, 1:]

        num_tokens_scored, green_token_count = self._score_ngrams_in_passage(input_ids)
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)

        if return_dict:
            p_value = scipy.stats.norm.sf(z_score)
            prediction = z_score > z_threshold
            confidence = 1 - p_value

        if return_dict:
            return WatermarkDetectorOutput(
                num_tokens_scored=num_tokens_scored,
                num_green_tokens=green_token_count,
                green_fraction=green_token_count / num_tokens_scored,
                z_score=z_score,
                p_value=p_value,
                prediction=prediction,
                confidence=confidence,
            )
        return prediction if return_prediction else z_score
