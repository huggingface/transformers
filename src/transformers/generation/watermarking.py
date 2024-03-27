# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team
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
import copy
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import numpy as np

from ..configuration_utils import PretrainedConfig
from ..utils import is_torch_available, logging
from .logits_process import WatermarkLogitsProcessor


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@dataclass
class WatermarkingConfig:
    def __init__(
        self,
        greenlist_ratio: Optional[float] = 0.25,
        bias: Optional[float] = 2.0,
        hashing_key: Optional[int] = 15485863,
        seeding_scheme: Optional[str] = "lefthash",
        context_width: Optional[int] = 1,
    ):
        self.greenlist_ratio = greenlist_ratio
        self.bias = bias
        self.hashing_key = hashing_key
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if config_dict is None:
            return None
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def __iter__(self):
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        return json.dumps(self.__dict__, indent=2) + "\n"

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class WatermarkDetectorOutput:
    """
    Outputs of a watermark detector.

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
        prediction (np.array of shape (batch_size)), *optional*:
            Array containing predictions for each batch.
        confidence (np.array of shape (batch_size)), *optional*:
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
    Detector for detection of watermark generated text. The detector needs to be given the exact same settings that were
    given during text generation to replicate the watermark greenlist generation and so detect the watermark. This includes
    the correct device that was used during text generation, the correct watermarking arguments and the correct tokenizer vocab size.
    The code was absed on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

    See [the paper](https://arxiv.org/abs/2306.04634) for more information.

    Args:
        model_config (`PretrainedConfig`):
            The model config that will be used to get model specific arguments used when generating.
        device (`str`):
            The device which was used during watermarked text generation.
        watermarking_config (Union[`WatermarkingConfig`, `Dict`]):
            The exact same watermarking config and arguments used when generating text.
        ignore_repeated_ngrams (`bool`, *optional*, defaults to `False`):
            Whether to count every unique ngram only once or not.
        max_size (`int`, *optional*, defaults to `2**8`):
            The max size to be used for LRU caching of seeding/sampling algorithms called for every token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig

    >>> model_id = "openai-community/gpt2"
    >>> model = AutoModelForCausalLM.from_pretrained(model_id)
    >>> tok = AutoTokenizer.from_pretrained(model_id)
    >>> tok.pad_token_id = tok.eos_token_id
    >>> tok.padding_side = "left"

    >>> inputs = tok(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
    >>> input_len = inputs["input_ids"].shape[-1]

    >>> # first generate text with watermark and without
    >>> watermarking_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
    >>> out_watermarked = model.generate(**inputs, watermarking_config=watermarking_config, do_sample=False, max_length=20)
    >>> out = model.generate(**inputs, do_sample=False, max_length=20)

    >>> # now we can instantiate the detector and check the generated text
    >>> detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
    >>> detection_out_watermarked = detector(out_watermarked, return_dict=True)
    >>> detection_out = detector(out, return_dict=True)
    >>> detection_out_watermarked.prediction
    array([ True,  True])

    >>> detection_out.prediction
    array([False,  False])
    ```
    """

    def __init__(
        self,
        model_config: PretrainedConfig,
        device: str,
        watermarking_config: Union[WatermarkingConfig, Dict],
        ignore_repeated_ngrams: bool = False,
        max_size: int = 2**8,
    ):
        if isinstance(watermarking_config, WatermarkingConfig):
            watermarking_config = watermarking_config.to_dict()

        self.bos_token_id = (
            model_config.bos_token_id if not model_config.is_encoder_decoder else model_config.decoder_start_token_id
        )
        self.greenlist_ratio = watermarking_config["greenlist_ratio"]
        self.ignore_repeated_ngrams = ignore_repeated_ngrams
        self.processor = WatermarkLogitsProcessor(
            vocab_size=model_config.vocab_size, device=device, **watermarking_config
        )
        self._get_ngram_score_cached = lru_cache(maxsize=max_size)(self._get_ngram_score)

    def _get_ngram_score(self, prefix: torch.LongTensor, target: int):
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

    def _compute_pval(self, x, loc=0, scale=1):
        z = (x - loc) / scale
        return 1 - (0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z**2 / np.pi))))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        z_threshold: float = 3.0,
        return_dict: bool = False,
    ) -> Union[WatermarkDetectorOutput, np.array]:
        """
        Args:
        input_ids (`torch.LongTensor`):
            The watermark generated text. It is advised to remove the prompt, which can affect the detection.
        z_threshold (`Dict`, *optional*, defaults to `3.0`):
            Changing this threshold will change the sensitivity of the detector. Higher z threshold gives less
            sensitivity and vice versa for lower z threshold.
        return_dict (`bool`,  *optional*, defaults to `False`):
            Whether to return `~generation.WatermarkDetectorOutput` or not. If not it will return boolean predictions,

        Return:
            [`~generation.WatermarkDetectorOutput`] or `np.array`: A [`~generation.WatermarkDetectorOutput`]
            if `return_dict=True` otherwise a `np.array`.

        """

        # Let's assume that if one batch start with `bos`, all batched also do
        if input_ids[0, 0] == self.bos_token_id:
            input_ids = input_ids[:, 1:]

        if input_ids.shape[-1] - self.processor.context_width < 1:
            raise ValueError(
                f"Must have at least `1` token to score after the first "
                f"min_prefix_len={self.processor.context_width} tokens required by the seeding scheme."
            )

        num_tokens_scored, green_token_count = self._score_ngrams_in_passage(input_ids)
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)

        p_value = self._compute_pval(z_score)
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
        return prediction
