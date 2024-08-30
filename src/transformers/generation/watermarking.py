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

from abc import ABC, abstractmethod
import collections
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Union

import numpy as np

from ..configuration_utils import PretrainedConfig
from ..utils import is_torch_available, logging
from .configuration_utils import WatermarkingConfig, GreenRedWatermarkingConfig


if is_torch_available():
    import torch

    from .logits_process import WatermarkLogitsProcessor


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
    >>> watermarking_config = GreenRedWatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
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
        watermarking_config: Union[GreenRedWatermarkingConfig, Dict],
        ignore_repeated_ngrams: bool = False,
        max_cache_size: int = 128,
    ):
        if isinstance(watermarking_config, GreenRedWatermarkingConfig):
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

"""Bayesian detector class."""

import abc
from collections.abc import Mapping, Sequence
import enum
import functools
import gc
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np
import optax
from sklearn import model_selection
from synthid_text import logits_processing
import torch
import tqdm
from typing_extensions import Self

def pad_to_len(
    arr: torch.tensor,
    target_len: int,
    left_pad: bool,
    eos_token: int,
    device: torch.device,
) -> torch.tensor:
  """Pad or truncate array to given length."""
  if arr.shape[1] < target_len:
    shape_for_ones = list(arr.shape)
    shape_for_ones[1] = target_len - shape_for_ones[1]
    padded = (
        torch.ones(
            shape_for_ones,
            device=device,
            dtype=torch.long,
        )
        * eos_token
    )
    if not left_pad:
      arr = torch.concatenate((arr, padded), dim=1)
    else:
      arr = torch.concatenate((padded, arr), dim=1)
  else:
    arr = arr[:, :target_len]
  return arr


def filter_and_truncate(
    outputs: torch.tensor,
    truncation_length: int,
    eos_token_mask: torch.tensor,
) -> torch.tensor:
  """Filter and truncate outputs to given length.

  Args:
   outputs: output tensor of shape [batch_size, output_len]
   truncation_length: Length to truncate the final output.
   eos_token_mask: EOS token mask of shape [batch_size, output_len]

  Returns:
   output tensor of shape [batch_size, truncation_length].
  """
  outputs = outputs[:, :truncation_length]
  truncation_mask = torch.sum(eos_token_mask, dim=1) >= truncation_length
  return outputs[truncation_mask, :]


def process_outputs_for_training(
    all_outputs: Sequence[torch.Tensor],
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    tokenizer: Any,
    *,
    truncation_length: int,
    max_length: int,
    is_cv: bool,
    is_pos: bool,
    torch_device: torch.device,
) -> tuple[Sequence[torch.tensor], Sequence[torch.tensor]]:
  """Process raw model outputs into format understandable by the detector.

  Args:
   all_outputs: sequence of outputs of shape [batch_size, output_len].
   logits_processor: logits processor used for watermarking.
   tokenizer: tokenizer used for the model.
   truncation_length: Length to truncate the outputs.
   max_length: Length to pad truncated outputs so that all processed entries.
     have same shape.
   is_cv: Process given outputs for cross validation.
   is_pos: Process given outputs for positives.
   torch_device: torch device to use.

  Returns:
    Tuple of
      all_masks: list of masks of shape [batch_size, max_length].
      all_g_values: list of g_values of shape [batch_size, max_length, depth].
  """
  all_masks = []
  all_g_values = []
  for outputs in tqdm.tqdm(all_outputs):
    # outputs is of shape [batch_size, output_len].
    # output_len can differ from batch to batch.
    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=tokenizer.eos_token_id,
    )
    if is_pos or is_cv:
      # filter with length for positives for both train and CV.
      # We also filter for length when CV negatives are processed.
      outputs = filter_and_truncate(outputs, truncation_length, eos_token_mask)

    # If no filtered outputs skip this batch.
    if outputs.shape[0] == 0:
      continue

    # All outputs are padded to max-length with eos-tokens.
    outputs = pad_to_len(
        outputs, max_length, False, tokenizer.eos_token_id, torch_device
    )
    # outputs shape [num_filtered_entries, max_length]

    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=tokenizer.eos_token_id,
    )

    context_repetition_mask = logits_processor.compute_context_repetition_mask(
        input_ids=outputs,
    )

    # context_repetition_mask of shape [num_filtered_entries, max_length -
    # (ngram_len - 1)].
    context_repetition_mask = pad_to_len(
        context_repetition_mask, max_length, True, 0, torch_device
    )
    # We pad on left to get same max_length shape.
    # context_repetition_mask of shape [num_filtered_entries, max_length].
    combined_mask = context_repetition_mask * eos_token_mask

    g_values = logits_processor.compute_g_values(
        input_ids=outputs,
    )

    # g_values of shape [num_filtered_entries, max_length - (ngram_len - 1),
    # depth].
    g_values = pad_to_len(g_values, max_length, True, 0, torch_device)

    # We pad on left to get same max_length shape.
    # g_values of shape [num_filtered_entries, max_length, depth].
    all_masks.append(combined_mask)
    all_g_values.append(g_values)
  return all_masks, all_g_values


@enum.unique
class ScoreType(enum.Enum):
  """Type of score returned by a WatermarkDetector.

  In all cases, larger score corresponds to watermarked text.
  """

  # Negative p-value where the p-value is the probability of observing equal or
  # stronger watermarking in unwatermarked text.
  NEGATIVE_P_VALUE = enum.auto()

  # Prob(watermarked | g-values).
  POSTERIOR = enum.auto()


class LikelihoodModel(abc.ABC):
  """Watermark likelihood model base class defining __call__ interface."""

  @abc.abstractmethod
  def __call__(self, g_values: jnp.ndarray) -> jnp.ndarray:
    """Computes likelihoods given g-values and a mask.

    Args:
      g_values: g-values (all are 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth, ...].

    Returns:
      an array of shape [batch_size, seq_len, watermarking_depth] or
      [batch_size, seq_len, 1] corresponding to the likelihoods
      of the g-values given either the watermarked hypothesis or
      the unwatermarked hypothesis; i.e. either P(g|watermarked)
      or P(g|unwatermarked).
    """


class LikelihoodModelWatermarked(nn.Module, LikelihoodModel):
  """Watermarked likelihood model for binary-valued g-values.

  This takes in g-values and returns p(g_values|watermarked).
  """

  watermarking_depth: int
  params: Mapping[str, Mapping[str, Any]] | None = None

  def setup(self):
    """Initializes the model parameters."""

    def noise(seed, shape):
      return jax.random.normal(key=jax.random.PRNGKey(seed), shape=shape)

    self.beta = self.param(
        "beta",
        lambda *x: (
            -2.5 + 0.001 * noise(seed=0, shape=(1, 1, self.watermarking_depth))
        ),
    )
    self.delta = self.param(
        "delta",
        lambda *x: (
            0.001
            * noise(
                seed=0,
                shape=(1, 1, self.watermarking_depth, self.watermarking_depth),
            )
        ),
    )

  def l2_loss(self) -> jnp.ndarray:
    return jnp.einsum("ijkl->", self.delta**2)

  def _compute_latents(
      self, g_values: jnp.ndarray
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the unique token probability distribution given g-values.

    Args:
      g_values: PRF values of shape [batch_size, seq_len, watermarking_depth].

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
    x = jnp.repeat(
        jnp.expand_dims(g_values, axis=-2), self.watermarking_depth, axis=-2
    )  # [batch_size, seq_len, watermarking_depth, watermarking_depth]

    x = jnp.tril(
        x, k=-1
    )  # mask all elements above -1 diagonal for autoregressive factorization

    logits = (
        jnp.einsum("ijkl,ijkl->ijk", self.delta, x) + self.beta
    )  # [batch_size, seq_len, watermarking_depth]

    p_two_unique_tokens = jax.nn.sigmoid(logits)
    p_one_unique_token = 1 - p_two_unique_tokens
    return p_one_unique_token, p_two_unique_tokens

  def __call__(self, g_values: jnp.ndarray) -> jnp.ndarray:
    """Computes the likelihoods P(g_values|watermarked).

    Args:
      g_values: g-values (values 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth]

    Returns:
      p(g_values|watermarked) of shape [batch_size, seq_len,
      watermarking_depth].
    """
    p_one_unique_token, p_two_unique_tokens = self._compute_latents(g_values)

    # P(g_tl | watermarked) is equal to
    # 0.5 * [ (g_tl+0.5) * p_two_unique_tokens + p_one_unique_token].
    return 0.5 * ((g_values + 0.5) * p_two_unique_tokens + p_one_unique_token)


class LikelihoodModelUnwatermarked(nn.Module, LikelihoodModel):
  """Unwatermarked likelihood model for binary-valued g-values.

  This takes in g-values and returns p(g_values | not watermarked).
  """

  @nn.compact
  def __call__(self, g_values: jnp.ndarray) -> jnp.ndarray:
    """Computes the likelihoods P(g-values|not watermarked).

    Args:
      g_values: g-values (0 or 1 values) of shape [batch_size, seq_len,
        watermarking_depth, ...].

    Returns:
      Likelihoods of g-values given text is unwatermarked --
      p(g_values | not watermarked) of shape [batch_size, seq_len,
      watermarking_depth].
    """
    return 0.5 * jnp.ones_like(g_values)  # all g-values have prob 0.5.


def _compute_posterior(
    likelihoods_watermarked: jnp.ndarray,
    likelihoods_unwatermarked: jnp.ndarray,
    mask: jnp.ndarray,
    prior: float,
) -> jnp.ndarray:
  """Compute posterior P(w|g) given likelihoods, mask and prior.

  Args:
    likelihoods_watermarked: shape [batch, length, depth]. Likelihoods
      P(g_values|watermarked) of g-values under watermarked model.
    likelihoods_unwatermarked: shape [batch, length, depth]. Likelihoods
      P(g_values|unwatermarked) of g-values under unwatermarked model.
    mask: A binary array shape [batch, length] indicating which g-values should
      be used. g-values with mask value 0 are discarded.
    prior: float, the prior probability P(w) that the text is watermarked.

  Returns:
    Posterior probability P(watermarked|g_values), shape [batch].
  """
  mask = jnp.expand_dims(mask, -1)
  prior = jnp.clip(prior, a_min=1e-5, a_max=1 - 1e-5)
  log_likelihoods_watermarked = jnp.log(
      jnp.clip(likelihoods_watermarked, a_min=1e-30, a_max=float("inf"))
  )
  log_likelihoods_unwatermarked = jnp.log(
      jnp.clip(likelihoods_unwatermarked, a_min=1e-30, a_max=float("inf"))
  )
  log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked

  # Sum relative surprisals (log odds) across all token positions and layers.
  relative_surprisal_likelihood = jnp.einsum(
      "i...->i", log_odds * mask
  )  # [batch_size].

  relative_surprisal_prior = jnp.log(prior) - jnp.log(1 - prior)

  # Combine prior and likelihood.
  relative_surprisal = (
      relative_surprisal_prior + relative_surprisal_likelihood
  )  # [batch_size]

  # Compute the posterior probability P(w|g) = sigmoid(relative_surprisal).
  return jax.nn.sigmoid(relative_surprisal)


class BayesianDetectorModule(nn.Module):
  """Bayesian classifier for watermark detection Flax Module.

  This detector uses Bayes' rule to compute a watermarking score, which is
  the posterior probability P(watermarked|g_values) that the text is
  watermarked, given its g_values.

  Note that this detector only works with Tournament-based watermarking using
  the Bernoulli(0.5) g-value distribution.
  """

  watermarking_depth: int  # The number of tournament layers.
  params: Mapping[str, Mapping[str, Any]] | None = None
  baserate: float = 0.5  # Prior probability P(w) that a text is watermarked.

  def train(
      self,
      *,
      g_values: jnp.ndarray,
      mask: jnp.ndarray,
      watermarked: jnp.ndarray,
      epochs: int = 250,
      learning_rate: float = 1e-3,
      minibatch_size: int = 64,
      seed: int = 0,
      l2_weight: float = 0.0,
      shuffle: bool = True,
      g_values_val: jnp.ndarray | None = None,
      mask_val: jnp.ndarray | None = None,
      watermarked_val: jnp.ndarray | None = None,
      verbose: bool = False,
      use_tpr_fpr_for_val: bool = False,
  ) -> tuple[Mapping[int, Mapping[str, PyTree]], float]:
    """Trains a Bayesian detector model.

    Args:
      g_values: g-values of shape [num_train, seq_len, watermarking_depth].
      mask: A binary array shape [num_train, seq_len] indicating which g-values
        should be used. g-values with mask value 0 are discarded.
      watermarked: A binary array of shape [num_train] indicating whether the
        example is watermarked (0: unwatermarked, 1: watermarked).
      epochs: Number of epochs to train for.
      learning_rate: Learning rate for optimizer.
      minibatch_size: Minibatch size for training. Note that a minibatch
        requires ~ 32 * minibatch_size * seq_len * watermarked_depth *
        watermarked_depth bits of memory.
      seed: Seed for parameter initialization.
      l2_weight: Weight to apply to L2 regularization for delta parameters.
      shuffle: Whether to shuffle before training.
      g_values_val: Validation g-values of shape [num_val, seq_len,
        watermarking_depth].
      mask_val: Validation mask of shape [num_val, seq_len].
      watermarked_val: Validation watermark labels of shape [num_val].
      verbose: Boolean indicating verbosity of training. If true, the loss will
        be printed. Defaulted to False.
      use_tpr_fpr_for_val: Whether to use TPR@FPR=1% as metric for validation.
        If false, use cross entropy loss.

    Returns:
      Tuple of
        training_history: Training history keyed by epoch number where the
        values are
          dictionaries containing the loss, validation loss, and model
          parameters,
          keyed by
          'loss', 'val_loss', and 'params', respectively.
        min_val_loss: Minimum validation loss achieved during training.
    """

    minibatch_inds = jnp.arange(0, len(g_values), minibatch_size)
    minibatch_inds_val = None
    if g_values_val is not None:
      minibatch_inds_val = jnp.arange(0, len(g_values_val), minibatch_size)

    rng = jax.random.PRNGKey(seed)
    param_rng, shuffle_rng = jax.random.split(rng)

    def coshuffle(*args):
      return [jax.random.permutation(shuffle_rng, x) for x in args]

    if shuffle:
      g_values, mask, watermarked = coshuffle(g_values, mask, watermarked)

    @jax.jit
    def xentropy_loss(y, y_pred) -> jnp.ndarray:
      """Calculates cross entropy loss."""
      y_pred = jnp.clip(y_pred, 1e-5, 1 - 1e-5)
      return -jnp.mean((y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred)))

    def loss_fn(
        params, detector_inputs, w_true, l2_batch_weight
    ) -> jnp.ndarray:
      """Calculates loss for a batch of data given parameters."""
      w_pred = self.apply(params, *detector_inputs, method=self.__call__)
      unweighted_l2 = self.apply(params, method=self.l2_loss)
      l2_loss = l2_batch_weight * unweighted_l2
      return xentropy_loss(w_true, w_pred) + l2_loss

    def tpr_at_fpr(
        params, detector_inputs, w_true, target_fpr=0.01
    ) -> jnp.ndarray:
      """Calculates TPR at FPR=target_fpr."""
      positive_idxs = w_true == 1
      negative_idxs = w_true == 0
      inds = jnp.arange(0, len(detector_inputs[0]), minibatch_size)
      w_preds = []
      for start in inds:
        end = start + minibatch_size
        detector_inputs_ = (
            detector_inputs[0][start:end],
            detector_inputs[1][start:end],
        )
        w_pred = self.apply(params, *detector_inputs_, method=self.__call__)
        w_preds.append(w_pred)
      w_pred = jnp.concatenate(w_preds, axis=0)
      positive_scores = w_pred[positive_idxs]
      negative_scores = w_pred[negative_idxs]
      fpr_threshold = jnp.percentile(negative_scores, 100 - target_fpr * 100)
      return jnp.mean(positive_scores >= fpr_threshold)

    def update_fn_if_fpr_tpr(params):
      """Loss function for negative TPR@FPR=1% as the validation loss."""
      tpr_ = tpr_at_fpr(
          params=params,
          detector_inputs=(g_values_val, mask_val),
          w_true=watermarked_val,
      )
      return -tpr_

    def update_with_minibatches(
        g, m, w, inds, params, opt_state=None, validation=False
    ):
      """Update params iff opt_state is not None and always returns the loss."""
      losses = []
      n_minibatches = len(g) / minibatch_size
      for start in inds:
        end = start + minibatch_size
        l2_batch_weight = l2_weight / n_minibatches
        if validation:
          l2_batch_weight = 0.0
        loss_fn_partialed = functools.partial(
            loss_fn,
            detector_inputs=(g[start:end], m[start:end]),
            w_true=w[start:end],
            l2_batch_weight=l2_batch_weight,
        )
        loss, grads = jax.value_and_grad(loss_fn_partialed)(params)
        losses.append(loss)
        if opt_state is not None:
          updates, opt_state = optimizer.update(grads, opt_state)
          params = optax.apply_updates(params, updates)
      loss = jnp.mean(jnp.array(losses))
      return loss, params, opt_state

    def update_fn(opt_state, params):
      """Updates the model parameters and returns the loss."""
      loss, params, opt_state = update_with_minibatches(
          g_values, mask, watermarked, minibatch_inds, params, opt_state
      )
      val_loss = None
      if g_values_val is not None:
        if use_tpr_fpr_for_val:
          val_loss = update_fn_if_fpr_tpr(params)
        else:
          val_loss, _, _ = update_with_minibatches(
              g_values_val,
              mask_val,
              watermarked_val,
              minibatch_inds_val,
              params,
              validation=True,
          )

      return opt_state, params, loss, val_loss

    params = self.params
    if params is None:
      params = self.init(param_rng, g_values[:1], mask[:1])

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    history = {}
    epochs_completed = 0
    while epochs_completed < epochs:
      opt_state, params, loss, val_loss = update_fn(opt_state, params)
      epochs_completed += 1

      history[epochs_completed] = {
          "loss": loss,
          "val_loss": val_loss,
          "params": params["params"],
      }
      if verbose:
        if val_loss is not None:
          print(
              f"Epoch {epochs_completed}: loss {loss} (train), {val_loss} (val)"
          )
        else:
          print(f"Epoch {epochs_completed}: loss {loss} (train)")
    self.params = params
    epochs = np.arange(1, epochs + 1)
    val_loss = np.squeeze(
        np.array([history[epoch]["val_loss"] for epoch in epochs])
    )
    best_val_epoch = np.argmin(val_loss) + 1
    min_val_loss = val_loss[best_val_epoch - 1]
    print(f"Best val Epoch: {best_val_epoch}, min_val_loss: {min_val_loss}")
    self.params = {"params": history[best_val_epoch]["params"]}
    return history, min_val_loss

  @property
  def score_type(self) -> ScoreType:
    return ScoreType.POSTERIOR

  def l2_loss(self) -> jnp.ndarray:
    return self.likelihood_model_watermarked.l2_loss()

  def setup(self):
    """Initializes the model parameters."""

    def _fetch_params():
      return {"params:": self.params["params"]["likelihood_model_watermarked"]}

    self.likelihood_model_watermarked = LikelihoodModelWatermarked(
        watermarking_depth=self.watermarking_depth,
        params=_fetch_params() if self.params is not None else None,
    )
    self.likelihood_model_unwatermarked = LikelihoodModelUnwatermarked()
    self.prior = self.param("prior", lambda *x: self.baserate, (1,))

  def __call__(
      self,
      g_values: jnp.ndarray,
      mask: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes the watermarked posterior P(watermarked|g_values).

    Args:
      g_values: g-values (with values 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth, ...]
      mask: A binary array shape [batch_size, seq_len] indicating which g-values
        should be used. g-values with mask value 0 are discarded.

    Returns:
      p(watermarked | g_values), of shape [batch_size].
    """

    likelihoods_watermarked = self.likelihood_model_watermarked(g_values)
    likelihoods_unwatermarked = self.likelihood_model_unwatermarked(g_values)
    return _compute_posterior(
        likelihoods_watermarked, likelihoods_unwatermarked, mask, self.prior
    )

  def score(
      self, g_values: jnp.ndarray | Sequence[jnp.ndarray], mask: jnp.ndarray
  ) -> jnp.ndarray:
    if self.params is None:
      raise ValueError("params must be set before calling score")
    return self.apply(self.params, g_values, mask, method=self.__call__)


class BayesianDetector:
  """Outside API class."""

  detector_module: BayesianDetectorModule
  tokenizer: Any
  logits_processor: logits_processing.SynthIDLogitsProcessor

  def __init__(
      self,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      tokenizer: Any,
      params: Mapping[str, Mapping[str, Any]],
  ):
    self.detector_module = BayesianDetectorModule(
        watermarking_depth=logits_processor.watermarking_depth,
        params=params,
    )
    self.logits_processor = logits_processor
    self.tokenizer = tokenizer

  def score(self, outputs: Sequence[np.ndarray]):
    """Temp."""
    # eos mask is computed, skip first ngram_len - 1 tokens
    # eos_mask will be of shape [batch_size, output_len]
    eos_token_mask = self.logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=self.tokenizer.eos_token_id,
    )[:, self.logits_processor.ngram_len - 1 :]

    # context repetition mask is computed
    context_repetition_mask = (
        self.logits_processor.compute_context_repetition_mask(
            input_ids=outputs,
        )
    )
    # context repitition mask shape [batch_size, output_len - (ngram_len - 1)]

    combined_mask = context_repetition_mask * eos_token_mask

    g_values = self.logits_processor.compute_g_values(
        input_ids=outputs,
    )
    # g values shape [batch_size, output_len - (ngram_len - 1), depth]
    return self.detector_module.score(
        g_values.cpu().numpy(), combined_mask.cpu().numpy()
    )

  @classmethod
  def train_best_detector(
      cls,
      *,
      tokenized_wm_outputs: Sequence[np.ndarray] | np.ndarray,
      tokenized_uwm_outputs: Sequence[np.ndarray] | np.ndarray,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      tokenizer: Any,
      torch_device: torch.device,
      test_size: float = 0.3,
      truncation_length: int = 200,
      max_padded_length: int = 2300,
      n_epochs: int = 50,
      watermarking_depth: int = 30,
      learning_rate: float = 2.1e-2,
      l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
      verbose: bool = False,
  ) -> tuple[Self, float]:
    """Construct, train and return the best detector based on wm and uwm data.

    Args:
      tokenized_wm_outputs: tokenized outputs of watermarked data.
      tokenized_uwm_outputs: tokenized outputs of unwatermarked data.
      logits_processor: logits processor used for watermarking.
      tokenizer: tokenizer used for the model.
      torch_device: torch device to use.
      test_size: test size to use for train-test split.
      truncation_length: Length to truncate wm and uwm outputs.
      max_padded_length: Length to pad truncated outputs so that all processed
        entries have same shape.
      n_epochs: Number of epochs to train the detector.
      watermarking_depth: Watermarking depth of the detector.
      learning_rate: Learning rate to use for training the detector.
      l2_weights: L2 weights to use for training the detector.

    Returns:
      Tuple of trained detector and loss achieved on CV data.
    """
    # Split data into train and CV
    train_wm_outputs, cv_wm_outputs = model_selection.train_test_split(
        tokenized_wm_outputs, test_size=test_size
    )

    train_uwm_outputs, cv_uwm_outputs = model_selection.train_test_split(
        tokenized_uwm_outputs, test_size=test_size
    )

    # Process both train and CV data for training
    wm_masks_train, wm_g_values_train = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in train_wm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=True,
        is_cv=False,
        torch_device=torch_device,
    )
    wm_masks_cv, wm_g_values_cv = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in cv_wm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=True,
        is_cv=True,
        torch_device=torch_device,
    )
    uwm_masks_train, uwm_g_values_train = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in train_uwm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=False,
        is_cv=False,
        torch_device=torch_device,
    )
    uwm_masks_cv, uwm_g_values_cv = process_outputs_for_training(
        [
            torch.tensor(outputs, device=torch_device, dtype=torch.long)
            for outputs in cv_uwm_outputs
        ],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=False,
        is_cv=True,
        torch_device=torch_device,
    )

    # We get list of data; here we concat all together to be passed to the
    # detector.
    wm_masks_train = torch.cat(wm_masks_train, dim=0)
    wm_g_values_train = torch.cat(wm_g_values_train, dim=0)
    wm_labels_train = torch.ones((wm_masks_train.shape[0],), dtype=torch.bool)
    wm_masks_cv = torch.cat(wm_masks_cv, dim=0)
    wm_g_values_cv = torch.cat(wm_g_values_cv, dim=0)
    wm_labels_cv = torch.ones((wm_masks_cv.shape[0],), dtype=torch.bool)

    uwm_masks_train = torch.cat(uwm_masks_train, dim=0)
    uwm_g_values_train = torch.cat(uwm_g_values_train, dim=0)
    uwm_labels_train = torch.zeros(
        (uwm_masks_train.shape[0],), dtype=torch.bool
    )
    uwm_masks_cv = torch.cat(uwm_masks_cv, dim=0)
    uwm_g_values_cv = torch.cat(uwm_g_values_cv, dim=0)
    uwm_labels_cv = torch.zeros((uwm_masks_cv.shape[0],), dtype=torch.bool)

    # Concat pos and negatives data together.
    train_g_values = (
        torch.cat((wm_g_values_train, uwm_g_values_train), dim=0).cpu().numpy()
    )
    train_labels = (
        torch.cat((wm_labels_train, uwm_labels_train), axis=0).cpu().numpy()
    )
    train_masks = (
        torch.cat((wm_masks_train, uwm_masks_train), axis=0).cpu().numpy()
    )

    cv_g_values = (
        torch.cat((wm_g_values_cv, uwm_g_values_cv), axis=0).cpu().numpy()
    )
    cv_labels = torch.cat((wm_labels_cv, uwm_labels_cv), axis=0).cpu().numpy()
    cv_masks = torch.cat((wm_masks_cv, uwm_masks_cv), axis=0).cpu().numpy()

    # Shuffle data.
    train_g_values = jnp.squeeze(train_g_values)
    train_labels = jnp.squeeze(train_labels)
    train_masks = jnp.squeeze(train_masks)

    cv_g_values = jnp.squeeze(cv_g_values)
    cv_labels = jnp.squeeze(cv_labels)
    cv_masks = jnp.squeeze(cv_masks)

    shuffled_idx = list(range(train_g_values.shape[0]))
    shuffled_idx = np.array(shuffled_idx)
    np.random.shuffle(shuffled_idx)
    train_g_values = train_g_values[shuffled_idx]
    train_labels = train_labels[shuffled_idx]
    train_masks = train_masks[shuffled_idx]

    shuffled_idx = list(range(cv_g_values.shape[0]))
    shuffled_idx = np.array(shuffled_idx)
    np.random.shuffle(shuffled_idx)
    cv_g_values = cv_g_values[shuffled_idx]
    cv_labels = cv_labels[shuffled_idx]
    cv_masks = cv_masks[shuffled_idx]

    # Del some variables so we free up GPU memory.
    del (
        wm_g_values_train,
        wm_labels_train,
        wm_masks_train,
        wm_g_values_cv,
        wm_labels_cv,
        wm_masks_cv,
    )
    gc.collect()
    torch.cuda.empty_cache()

    best_detector = None
    lowest_loss = float("inf")
    val_losses = []
    for l2_weight in l2_weights:
      detector = BayesianDetectorModule(
          watermarking_depth=watermarking_depth,
      )
      _, min_val_loss = detector.train(
          g_values=train_g_values,
          mask=train_masks,
          watermarked=train_labels,
          g_values_val=cv_g_values,
          mask_val=cv_masks,
          watermarked_val=cv_labels,
          learning_rate=learning_rate,
          l2_weight=l2_weight,
          epochs=n_epochs,
          verbose=verbose,
      )
      val_losses.append(min_val_loss)
      if min_val_loss < lowest_loss:
        lowest_loss = min_val_loss
        best_detector = detector

    return cls(logits_processor, tokenizer, best_detector.params), lowest_loss
