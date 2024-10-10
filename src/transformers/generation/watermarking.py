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
import enum
import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import tqdm
from sklearn import model_selection
from synthid_text import logits_processing
from torch import nn
from torch.nn import CrossEntropyLoss

from ..tokenization_utils import PreTrainedTokenizer
from ..utils import ModelOutput, add_start_docstrings, is_torch_available, logging
from .configuration_utils import GreenRedWatermarkingConfig, PreTrainedConfig
from .modeling_utils import PreTrainedModel


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
        model_config: PreTrainedConfig,
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



class BayesWatermarkingConfig(PreTrainedConfig):

    """
     This is the configuration class to store the configuration of a [`BayesianDetectorModel`]. It is used to
    instantiate a Bayesian Detector model according to the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        watermarking_depth (`int`, *optional*):
            The number of tournament layers.
        base_rate (`float1`, *optional*, defaults to `0.5`):
            Prior probability P(w) that a text is watermarked.
    """

    def __init__(
        self,
        watermarking_depth: int=None,
        base_rate: float=0.5,
        **kwargs

    ):
        self.watermarking_depth = watermarking_depth
        self.base_rate = base_rate

        super().__init__(**kwargs)


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
        self.beta = torch.nn.Parameter(
            -2.5 + 0.001 * torch.randn(1, 1, watermarking_depth)
        )
        self.delta = torch.nn.Parameter(
            0.001 * torch.randn(1, 1, self.watermarking_depth, watermarking_depth)
        )

    def _compute_latents(self, g_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        # [batch_size, seq_len, watermarking_depth, watermarking_depth]
        x = torch.repeat_interleave(
            torch.unsqueeze(g_values, dim=-2), self.watermarking_depth, axis=-2
        )

        # mask all elements above -1 diagonal for autoregressive factorization
        x = torch.tril(x, diagonal=-1)

        # [batch_size, seq_len, watermarking_depth]
        # Long tensor doesn't work with einsum, so we need to switch to the same dtype as self.delta (FP32)
        logits = torch.einsum("ijkl,ijkl->ijk", self.delta, x.type(self.delta.dtype)) + self.beta

        p_two_unique_tokens = torch.sigmoid(logits)
        p_one_unique_token = 1 - p_two_unique_tokens
        return p_one_unique_token, p_two_unique_tokens

    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
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


class BayesianDetectorPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BayesWatermarkingConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Parameter):
            module.weight.data.normal_(mean=0.0, std=0.02)


BAYESIAN_DETECTOR_START_DOCSTRING = r"""

    Bayesian classifier for watermark detection.

    This detector uses Bayes' rule to compute a watermarking score, which is
    the posterior probability P(watermarked|g_values) that the text is
    watermarked, given its g_values.

    Note that this detector only works with Tournament-based watermarking using
    the Bernoulli(0.5) g-value distribution.

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

@add_start_docstrings(
    BAYESIAN_DETECTOR_START_DOCSTRING,
)
class BayesianDetectorModel(BayesianDetectorPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.watermarking_depth = config.watermarking_depth
        self.base_rate = config.base_rate
        self.likelihood_model_watermarked = BayesianDetectorWatermarkedLikelihood(
            watermarking_depth=self.watermarking_depth
        )
        self.prior = torch.nn.Parameter(torch.tensor([self.base_rate]))


    def _compute_posterior(self,
                           likelihoods_watermarked: torch.Tensor,
                           likelihoods_unwatermarked: torch.Tensor,
                           mask: torch.Tensor,
                           prior: float,
    ) -> torch.Tensor:
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
        mask = torch.unsqueeze(mask, dim=-1)
        prior = torch.clamp(prior, min=1e-5, max=1 - 1e-5)
        log_likelihoods_watermarked = torch.log(
            torch.clamp(likelihoods_watermarked, min=1e-30, max=float("inf"))
        )
        log_likelihoods_unwatermarked = torch.log(
            torch.clamp(likelihoods_unwatermarked, min=1e-30, max=float("inf"))
        )
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

    def forward(self, g_values: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, return_dict=None) -> BayesianWatermarkDetectorModelOutput:
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
        likelihoods_unwatermarked = 0.5 * torch.ones_like(g_values)
        out = self._compute_posterior(likelihoods_watermarked=likelihoods_watermarked,
                                       likelihoods_unwatermarked=likelihoods_unwatermarked,
                                       mask=mask,
                                       prior=self.prior)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_weight = torch.sum(self.delta**2)
            loss = loss_fct(out, labels) + loss_weight

        if not return_dict:
            return (out,) if loss is None else (out, loss)

        return BayesianWatermarkDetectorModelOutput(loss=loss, posterior_probabilities=out)


def pad_to_len(
    arr: torch.Tensor,
    target_len: int,
    left_pad: bool,
    eos_token: int,
    device: torch.device,
) -> torch.Tensor:
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
    outputs: torch.Tensor,
    truncation_length: int,
    eos_token_mask: torch.Tensor,
) -> torch.Tensor:
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
) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
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



def tpr_at_fpr(
    detector, detector_inputs, w_true, minibatch_size, target_fpr=0.01
) -> torch.Tensor:
    """Calculates TPR at FPR=target_fpr."""
    positive_idxs = w_true == 1
    negative_idxs = w_true == 0
    num_samples = detector_inputs[0].size(0)

    w_preds = []
    for start in range(0, num_samples, minibatch_size):
        end = start + minibatch_size
        detector_inputs_ = (
            detector_inputs[0][start:end],
            detector_inputs[1][start:end],
        )
        with torch.no_grad():
            w_pred = detector(*detector_inputs_)
        w_preds.append(w_pred)

    w_pred = torch.cat(w_preds, dim=0)  # Concatenate predictions
    positive_scores = w_pred[positive_idxs]
    negative_scores = w_pred[negative_idxs]

    # Calculate the FPR threshold
    # Note: percentile -> quantile
    fpr_threshold = torch.quantile(negative_scores, 1 - target_fpr)
    # Note: need to switch to FP32 since torch.mean doesn't work with torch.bool
    return torch.mean((positive_scores >= fpr_threshold).to(dtype=torch.float32)).item()  # TPR


def update_fn_if_fpr_tpr(
    detector, g_values_val, mask_val, watermarked_val, minibatch_size
):
    """Loss function for negative TPR@FPR=1% as the validation loss."""
    tpr_ = tpr_at_fpr(
        detector=detector,
        detector_inputs=(g_values_val, mask_val),
        w_true=watermarked_val,
        minibatch_size=minibatch_size,
    )
    return -tpr_


@enum.unique
class ValidationMetric(enum.Enum):
    """Direction along the z-axis."""

    TPR_AT_FPR = "tpr_at_fpr"
    CROSS_ENTROPY = "cross_entropy"


def train_detector(
    detector: torch.nn.Module,
    g_values: torch.Tensor,
    mask: torch.Tensor,
    watermarked: torch.Tensor,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    minibatch_size: int = 64,
    seed: int = 0,
    l2_weight: float = 0.0,
    shuffle: bool = True,
    g_values_val: torch.Tensor | None = None,
    mask_val: torch.Tensor | None = None,
    watermarked_val: torch.Tensor | None = None,
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
) -> tuple[Mapping, float]:
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

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Shuffle the data if required
    if shuffle:
        indices = torch.randperm(len(g_values))
        g_values = g_values[indices]
        mask = mask[indices]
        watermarked = watermarked[indices]

    # Initialize optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
    history = {}
    min_val_loss = float("inf")

    for epoch in range(epochs):
        losses = []
        detector.train()
        num_batches = len(g_values) // minibatch_size
        for i in range(0, len(g_values), minibatch_size):
            end = i + minibatch_size
            if end > len(g_values):
                break
            l2_batch_weight = l2_weight / num_batches

            optimizer.zero_grad()
            loss = loss_fn(
                detector=detector,
                detector_inputs=(g_values[i:end], mask[i:end]),
                w_true=watermarked[i:end],
                l2_batch_weight=l2_batch_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = sum(losses)/len(losses)

        val_losses = []
        if g_values_val is not None:
            detector.eval()
            if validation_metric == ValidationMetric.TPR_AT_FPR:
                val_loss = update_fn_if_fpr_tpr(
                    detector,
                    g_values_val,
                    mask_val,
                    watermarked_val,
                    minibatch_size=minibatch_size,
                )
            else:
                for i in range(0, len(g_values_val), minibatch_size):
                    end = i + minibatch_size
                    if end > len(g_values_val):
                        break
                    with torch.no_grad():
                        v_loss = loss_fn(
                            detector=detector,
                            detector_inputs=(g_values_val[i:end], mask_val[i:end]),
                            w_true=watermarked_val[i:end],
                            l2_batch_weight=0,
                        )
                    val_losses.append(v_loss.item())
                val_loss = sum(val_losses)/len(val_losses)

        # Store training history
        history[epoch + 1] = {
            "loss": train_loss,
            "val_loss": val_loss
        }
        if verbose:
            if val_loss is not None:
                print(f"Epoch {epoch}: loss {loss} (train), {val_loss} (val)")
            else:
                print(f"Epoch {epoch}: loss {loss} (train)")

        if val_loss is not None and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_val_epoch = epoch

    if verbose:
        print(f"Best val Epoch: {best_val_epoch}, min_val_loss: {min_val_loss}")

    return history, min_val_loss

def process_raw_model_outputs(
    logits_processor,
    tokenizer,
    truncation_length,
    max_padded_length,
    tokenized_wm_outputs,
    test_size,
    tokenized_uwm_outputs,
    torch_device,
):
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
    # Note: Use long instead of bool. Otherwise, the entropy calculation doesn't work
    wm_labels_train = torch.ones((wm_masks_train.shape[0],), dtype=torch.long, device=torch_device)

    wm_masks_cv = torch.cat(wm_masks_cv, dim=0)
    wm_g_values_cv = torch.cat(wm_g_values_cv, dim=0)
    wm_labels_cv = torch.ones((wm_masks_cv.shape[0],), dtype=torch.long, device=torch_device)

    uwm_masks_train = torch.cat(uwm_masks_train, dim=0)
    uwm_g_values_train = torch.cat(uwm_g_values_train, dim=0)
    uwm_labels_train = torch.zeros((uwm_masks_train.shape[0],), dtype=torch.long, device=torch_device)
    uwm_masks_cv = torch.cat(uwm_masks_cv, dim=0)
    uwm_g_values_cv = torch.cat(uwm_g_values_cv, dim=0)
    uwm_labels_cv = torch.zeros((uwm_masks_cv.shape[0],), dtype=torch.long, device=torch_device)

    # Concat pos and negatives data together.
    train_g_values = torch.cat((wm_g_values_train, uwm_g_values_train), dim=0)
    train_labels = torch.cat((wm_labels_train, uwm_labels_train), axis=0)
    train_masks = torch.cat((wm_masks_train, uwm_masks_train), axis=0)

    cv_g_values = torch.cat((wm_g_values_cv, uwm_g_values_cv), axis=0)
    cv_labels = torch.cat((wm_labels_cv, uwm_labels_cv), axis=0)
    cv_masks = torch.cat((wm_masks_cv, uwm_masks_cv), axis=0)

    # Shuffle data.
    train_g_values = train_g_values.squeeze()
    train_labels = train_labels.squeeze()
    train_masks = train_masks.squeeze()

    cv_g_values = cv_g_values.squeeze()
    cv_labels = cv_labels.squeeze()
    cv_masks = cv_masks.squeeze()

    shuffled_idx = torch.randperm(
        train_g_values.shape[0]
    )  # Use torch for GPU compatibility

    train_g_values = train_g_values[shuffled_idx]
    train_labels = train_labels[shuffled_idx]
    train_masks = train_masks[shuffled_idx]

    # Shuffle the cross-validation data
    shuffled_idx_cv = torch.randperm(
        cv_g_values.shape[0]
    )  # Use torch for GPU compatibility
    cv_g_values = cv_g_values[shuffled_idx_cv]
    cv_labels = cv_labels[shuffled_idx_cv]
    cv_masks = cv_masks[shuffled_idx_cv]

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

    return train_g_values, train_masks, train_labels, cv_g_values, cv_masks, cv_labels


def train_best_detector(
    tokenized_wm_outputs: Sequence[np.ndarray] | np.ndarray,
    tokenized_uwm_outputs: Sequence[np.ndarray] | np.ndarray,
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    tokenizer: Any,
    torch_device: torch.device,
    test_size: float = 0.3,
    truncation_length: int = 200,
    max_padded_length: int = 2300,
    n_epochs: int = 50,
    learning_rate: float = 2.1e-2,
    l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
):
    l2_weights = list(l2_weights)

    (
        train_g_values,
        train_masks,
        train_labels,
        cv_g_values,
        cv_masks,
        cv_labels,
    ) = process_raw_model_outputs(
        logits_processor,
        tokenizer,
        truncation_length,
        max_padded_length,
        tokenized_wm_outputs,
        test_size,
        tokenized_uwm_outputs,
        torch_device,
    )

    best_detector = None
    lowest_loss = float("inf")
    val_losses = []
    for l2_weight in l2_weights:
        detector = BayesianDetectorModule(
            watermarking_depth=len(logits_processor.keys),
        ).to(torch_device)
        _, min_val_loss = train_detector(
            detector=detector,
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
            validation_metric=validation_metric,
        )
        val_losses.append(min_val_loss)
        if min_val_loss < lowest_loss:
            lowest_loss = min_val_loss
            best_detector = detector
    return best_detector, lowest_loss


### Temporary, will be replaced by `WatermarkDetector` class in transformers.Need to find a way to add the detector_module inside.
class BayesianDetectorPT:
  """Outside API class."""

  def __init__(
      self,
      detector_module: PreTrainedModel,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      tokenizer: PreTrainedTokenizer,
  ):
    self.detector_module = detector_module
    self.logits_processor = logits_processor
    self.tokenizer = tokenizer

  def __call__(self, outputs: Sequence[np.ndarray]):
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
    return self.detector_module(g_values, combined_mask)
