# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team and Google DeepMind.
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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCELoss

from ..modeling_utils import PreTrainedModel
from ..utils import ModelOutput, is_torch_available, logging
from .configuration_utils import PretrainedConfig, WatermarkingConfig


if is_torch_available():
    import torch

    from .logits_process import SynthIDTextWatermarkLogitsProcessor, WatermarkLogitsProcessor


logger = logging.get_logger(__name__)


@dataclass
class WatermarkDetectorOutput:
    """
    Outputs of a watermark detector.

    Args:
        num_tokens_scored (np.array of shape (batch_size)):
            Array containing the number of tokens scored for each element in the batch.
        num_green_tokens (np.array of shape (batch_size)):
            Array containing the number of green tokens for each element in the batch.
        green_fraction (np.array of shape (batch_size)):
            Array containing the fraction of green tokens for each element in the batch.
        z_score (np.array of shape (batch_size)):
            Array containing the z-score for each element in the batch. Z-score here shows
            how many standard deviations away is the green token count in the input text
            from the expected green token count for machine-generated text.
        p_value (np.array of shape (batch_size)):
            Array containing the p-value for each batch obtained from z-scores.
        prediction (np.array of shape (batch_size)), *optional*:
            Array containing boolean predictions whether a text is machine-generated for each element in the batch.
        confidence (np.array of shape (batch_size)), *optional*:
            Array containing confidence scores of a text being machine-generated for each element in the batch.
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
    The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

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
        max_cache_size (`int`, *optional*, defaults to 128):
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
        max_cache_size: int = 128,
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

        # Expensive re-seeding and sampling is cached.
        self._get_ngram_score_cached = lru_cache(maxsize=max_cache_size)(self._get_ngram_score)

    def _get_ngram_score(self, prefix: torch.LongTensor, target: int):
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
        ma
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
        prediction = z_score > z_threshold

        if return_dict:
            p_value = self._compute_pval(z_score)
            confidence = 1 - p_value

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


class BayesianDetectorConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BayesianDetectorModel`]. It is used to
    instantiate a Bayesian Detector model according to the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        watermarking_depth (`int`, *optional*):
            The number of tournament layers.
        base_rate (`float1`, *optional*, defaults to 0.5):
            Prior probability P(w) that a text is watermarked.
    """

    def __init__(self, watermarking_depth: int = None, base_rate: float = 0.5, **kwargs):
        self.watermarking_depth = watermarking_depth
        self.base_rate = base_rate
        # These can be set later to store information about this detector.
        self.model_name = None
        self.watermarking_config = None

        super().__init__(**kwargs)

    def set_detector_information(self, model_name, watermarking_config):
        self.model_name = model_name
        self.watermarking_config = watermarking_config


@dataclass
class BayesianWatermarkDetectorModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if the text is watermarked.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        posterior_probabilities (`torch.FloatTensor` of shape `(1,)`):
            Multiple choice classification loss.
    """

    loss: Optional[torch.FloatTensor] = None
    posterior_probabilities: Optional[torch.FloatTensor] = None


class BayesianDetectorWatermarkedLikelihood(nn.Module):
    """Watermarked likelihood model for binary-valued g-values.

    This takes in g-values and returns p(g_values|watermarked).
    """

    def __init__(self, watermarking_depth: int):
        """Initializes the model parameters."""
        super().__init__()
        self.watermarking_depth = watermarking_depth
        self.beta = torch.nn.Parameter(-2.5 + 0.001 * torch.randn(1, 1, watermarking_depth))
        self.delta = torch.nn.Parameter(0.001 * torch.randn(1, 1, self.watermarking_depth, watermarking_depth))

    def _compute_latents(self, g_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the unique token probability distribution given g-values.

        Args:
            g_values (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth)`):
                PRF values.

        Returns:
            p_one_unique_token and p_two_unique_tokens, both of shape
            [batch_size, seq_len, watermarking_depth]. p_one_unique_token[i,t,l]
            gives the probability of there being one unique token in a tournament
            match on layer l, on timestep t, for batch item i.
            p_one_unique_token[i,t,l] + p_two_unique_token[i,t,l] = 1.
        """
        # Tile g-values to produce feature vectors for predicting the latents
        # for each layer in the tournament; our model for the latents psi is a
        # logistic regression model psi = sigmoid(delta * x + beta).

        # [batch_size, seq_len, watermarking_depth, watermarking_depth]
        x = torch.repeat_interleave(torch.unsqueeze(g_values, dim=-2), self.watermarking_depth, axis=-2)

        # mask all elements above -1 diagonal for autoregressive factorization
        x = torch.tril(x, diagonal=-1)

        # [batch_size, seq_len, watermarking_depth]
        # (i, j, k, l) x (i, j, k, l) -> (i, j, k) einsum equivalent
        logits = (self.delta[..., None, :] @ x.type(self.delta.dtype)[..., None]).squeeze() + self.beta

        p_two_unique_tokens = torch.sigmoid(logits)
        p_one_unique_token = 1 - p_two_unique_tokens
        return p_one_unique_token, p_two_unique_tokens

    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Computes the likelihoods P(g_values|watermarked).

        Args:
            g_values (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth)`):
                g-values (values 0 or 1)

        Returns:
            p(g_values|watermarked) of shape [batch_size, seq_len, watermarking_depth].
        """
        p_one_unique_token, p_two_unique_tokens = self._compute_latents(g_values)

        # P(g_tl | watermarked) is equal to
        # 0.5 * [ (g_tl+0.5) * p_two_unique_tokens + p_one_unique_token].
        return 0.5 * ((g_values + 0.5) * p_two_unique_tokens + p_one_unique_token)


class BayesianDetectorModel(PreTrainedModel):
    r"""
    Bayesian classifier for watermark detection.

    This detector uses Bayes' rule to compute a watermarking score, which is the sigmoid of the log of ratio of the
    posterior probabilities P(watermarked|g_values) and P(unwatermarked|g_values). Please see the section on
    BayesianScore in the paper for further details.
    Paper URL: https://www.nature.com/articles/s41586-024-08025-4

    Note that this detector only works with non-distortionary Tournament-based watermarking using the Bernoulli(0.5)
    g-value distribution.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BayesianDetectorConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    config_class = BayesianDetectorConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)

        self.watermarking_depth = config.watermarking_depth
        self.base_rate = config.base_rate
        self.likelihood_model_watermarked = BayesianDetectorWatermarkedLikelihood(
            watermarking_depth=self.watermarking_depth
        )
        self.prior = torch.nn.Parameter(torch.tensor([self.base_rate]))

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Parameter):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _compute_posterior(
        self,
        likelihoods_watermarked: torch.Tensor,
        likelihoods_unwatermarked: torch.Tensor,
        mask: torch.Tensor,
        prior: float,
    ) -> torch.Tensor:
        """
        Compute posterior P(w|g) given likelihoods, mask and prior.

        Args:
            likelihoods_watermarked (`torch.Tensor` of shape `(batch, length, depth)`):
                Likelihoods P(g_values|watermarked) of g-values under watermarked model.
            likelihoods_unwatermarked (`torch.Tensor` of shape `(batch, length, depth)`):
                Likelihoods P(g_values|unwatermarked) of g-values under unwatermarked model.
            mask (`torch.Tensor` of shape `(batch, length)`):
                A binary array indicating which g-values should be used. g-values with mask value 0 are discarded.
            prior (`float`):
                the prior probability P(w) that the text is watermarked.

        Returns:
            Posterior probability P(watermarked|g_values), shape [batch].
        """
        mask = torch.unsqueeze(mask, dim=-1)
        prior = torch.clamp(prior, min=1e-5, max=1 - 1e-5)
        log_likelihoods_watermarked = torch.log(torch.clamp(likelihoods_watermarked, min=1e-30, max=float("inf")))
        log_likelihoods_unwatermarked = torch.log(torch.clamp(likelihoods_unwatermarked, min=1e-30, max=float("inf")))
        log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked

        # Sum relative surprisals (log odds) across all token positions and layers.
        relative_surprisal_likelihood = torch.einsum("i...->i", log_odds * mask)

        # Compute the relative surprisal prior
        relative_surprisal_prior = torch.log(prior) - torch.log(1 - prior)

        # Combine prior and likelihood.
        # [batch_size]
        relative_surprisal = relative_surprisal_prior + relative_surprisal_likelihood

        # Compute the posterior probability P(w|g) = sigmoid(relative_surprisal).
        return torch.sigmoid(relative_surprisal)

    def forward(
        self,
        g_values: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_batch_weight=1,
        return_dict=False,
    ) -> BayesianWatermarkDetectorModelOutput:
        """
        Computes the watermarked posterior P(watermarked|g_values).

        Args:
            g_values (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth, ...)`):
                g-values (with values 0 or 1)
            mask:
                A binary array shape [batch_size, seq_len] indicating which g-values should be used. g-values with mask
                value 0 are discarded.

        Returns:
            p(watermarked | g_values), of shape [batch_size].
        """

        likelihoods_watermarked = self.likelihood_model_watermarked(g_values)
        likelihoods_unwatermarked = 0.5 * torch.ones_like(g_values)
        out = self._compute_posterior(
            likelihoods_watermarked=likelihoods_watermarked,
            likelihoods_unwatermarked=likelihoods_unwatermarked,
            mask=mask,
            prior=self.prior,
        )

        loss = None
        if labels is not None:
            loss_fct = BCELoss()
            loss_unwweight = torch.sum(self.likelihood_model_watermarked.delta**2)
            loss_weight = loss_unwweight * loss_batch_weight
            loss = loss_fct(torch.clamp(out, 1e-5, 1 - 1e-5), labels) + loss_weight

        if not return_dict:
            return (out,) if loss is None else (out, loss)

        return BayesianWatermarkDetectorModelOutput(loss=loss, posterior_probabilities=out)


class SynthIDTextWatermarkDetector:
    r"""
    SynthID text watermark detector class.

    This class has to be initialized with the trained bayesian detector module check script
    in examples/synthid_text/detector_training.py for example in training/saving/loading this
    detector module. The folder also showcases example use case of this detector.

    Parameters:
        detector_module ([`BayesianDetectorModel`]):
            Bayesian detector module object initialized with parameters.
            Check examples/research_projects/synthid_text/detector_training.py for usage.
        logits_processor (`SynthIDTextWatermarkLogitsProcessor`):
            The logits processor used for watermarking.
        tokenizer (`Any`):
            The tokenizer used for the model.

    Examples:
    ```python
    >>> from transformers import (
    ...     AutoTokenizer, BayesianDetectorModel, SynthIDTextWatermarkLogitsProcessor, SynthIDTextWatermarkDetector
    ... )

    >>> # Load the detector. See examples/research_projects/synthid_text for training a detector.
    >>> detector_model = BayesianDetectorModel.from_pretrained("joaogante/dummy_synthid_detector")
    >>> logits_processor = SynthIDTextWatermarkLogitsProcessor(
    ...     **detector_model.config.watermarking_config, device="cpu"
    ... )
    >>> tokenizer = AutoTokenizer.from_pretrained(detector_model.config.model_name)
    >>> detector = SynthIDTextWatermarkDetector(detector_model, logits_processor, tokenizer)

    >>> # Test whether a certain string is watermarked
    >>> test_input = tokenizer(["This is a test input"], return_tensors="pt")
    >>> is_watermarked = detector(test_input.input_ids)
    ```
    """

    def __init__(
        self,
        detector_module: BayesianDetectorModel,
        logits_processor: SynthIDTextWatermarkLogitsProcessor,
        tokenizer: Any,
    ):
        self.detector_module = detector_module
        self.logits_processor = logits_processor
        self.tokenizer = tokenizer

    def __call__(self, tokenized_outputs: torch.Tensor):
        # eos mask is computed, skip first ngram_len - 1 tokens
        # eos_mask will be of shape [batch_size, output_len]
        eos_token_mask = self.logits_processor.compute_eos_token_mask(
            input_ids=tokenized_outputs,
            eos_token_id=self.tokenizer.eos_token_id,
        )[:, self.logits_processor.ngram_len - 1 :]

        # context repetition mask is computed
        context_repetition_mask = self.logits_processor.compute_context_repetition_mask(
            input_ids=tokenized_outputs,
        )
        # context repitition mask shape [batch_size, output_len - (ngram_len - 1)]

        combined_mask = context_repetition_mask * eos_token_mask

        g_values = self.logits_processor.compute_g_values(
            input_ids=tokenized_outputs,
        )
        # g values shape [batch_size, output_len - (ngram_len - 1), depth]
        return self.detector_module(g_values, combined_mask)
