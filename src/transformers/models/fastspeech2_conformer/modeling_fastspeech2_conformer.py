# coding=utf-8
# Copyright 2023 The Espnet authors, IMS Toucan authors, and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SpeechT5 model."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy
import torch
from torch import nn

from ...modeling_outputs import BaseModelOutput, Seq2SeqSpectrogramOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_fastspeech2_conformer import FastSpeech2ConformerConfig, FastSpeech2ConformerHifiGanConfig

# General docstring
_CONFIG_FOR_DOC = "FastSpeech2ConformerConfig"

logger = logging.get_logger(__name__)

FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "connor-henderson/fastspeech2_conformer",
    # See all FastSpeech2Conformer models at https://huggingface.co/models?filter=fastspeech2_conformer
]


@dataclass
class FastSpeech2ConformerModelOutput(Seq2SeqSpectrogramOutput):
    """Output type of [`FastSpeech2ConformerModel`]."""

    duration_outputs: torch.LongTensor = None
    pitch_outputs: torch.FloatTensor = None
    energy_outputs: torch.FloatTensor = None


def pad_list(xs, pad_value):
    """
    Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (batch_size, max_text_length, `*`).

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (batch_size,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2- dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    batch_size = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    if device is not None:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=device)
    else:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        if xs.size(0) != batch_size:
            raise ValueError(f"Size mismatch in `make_pad_mask`: {xs.size(0)} != {batch_size}")

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (batch_size,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2- dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    return ~make_pad_mask(lengths, xs, length_dim, device=device)


class FastSpeech2ConformerDurationPredictor(nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`, the
        outputs are calculated in log domain but in `inference`, those are calculated in linear domain.

    """

    def __init__(self, input_dim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, log_domain_offset=1.0):
        """
        Initialize duration predictor module.

        Args:
            input_dim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            log_domain_offset (float, optional): Offset value to avoid nan in log domain.

        """
        super().__init__()
        self.log_domain_offset = log_domain_offset
        self.conv_layers = torch.nn.ModuleList()
        for layer_idx in range(n_layers):
            input_channels = input_dim if layer_idx == 0 else n_chans
            layer = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        input_channels,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    FastSpeech2ConformerLayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                ]
            )
            self.conv_layers.append(layer)
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, encoder_hidden_states, padding_masks=None, is_inference=False):
        """
        Args:
            hidden_states (Tensor): Batch of input sequences (batch_size, max_text_length, input_dim).
            padding_masks (ByteTensor, optional):
                Batch of masks indicating padded part (batch_size, max_text_length).
            is_inference (Boolean, optional): Whether or not the model is running inference.

        Returns:
            Tensor: Batch of predicted durations in log domain (batch_size, max_text_length).

        """
        # (batch_size, input_dim, max_text_length)
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            for module in layer:
                hidden_states = module(hidden_states)

        # NOTE: calculate in log domain, (batch_size, max_text_length)
        hidden_states = self.linear(hidden_states.transpose(1, -1)).squeeze(-1)

        if is_inference:
            # NOTE: calculate in linear domain
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()

        if padding_masks is not None:
            hidden_states = hidden_states.masked_fill(padding_masks, 0.0)

        return hidden_states


class FastSpeech2ConformerLengthRegulator(nn.Module):
    """
    Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_. The
    length regulator expands char or phoneme-level embedding features to frame-level by repeating each feature based on
    the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """
        Initialize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.
        """
        super().__init__()
        self.pad_value = pad_value

    def forward(self, encoded_embeddings, target_durations, alpha=1.0):
        """
        Calculate forward propagation.

        Args:
            encoded_embeddings (Tensor):
                Batch of sequences of char or phoneme embeddings (batch_size, max_text_length, embedding_dim).
            target_durations (LongTensor): Batch of durations of each frame (batch_size, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (batch_size, T*, embedding_dim).
        """

        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        elif alpha != 1.0:
            target_durations = torch.round(target_durations.float() * alpha).long()

        if target_durations.sum() == 0:
            target_durations[target_durations.sum(dim=1).eq(0)] = 1

        return pad_list(
            [
                self._repeat_one_sequence(encoded_embedding, target_duration)
                for encoded_embedding, target_duration in zip(encoded_embeddings, target_durations)
            ],
            self.pad_value,
        )

    def _repeat_one_sequence(self, encoded_embedding, target_duration):
        """
        Repeat each frame according to duration
        """
        return torch.repeat_interleave(encoded_embedding, target_duration, dim=0)


# Copied from transformers.models.speecht5.modeling_speecht5.SpeechT5BatchNormConvLayer
class FastSpeech2ConformerBatchNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()

        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units

        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.speech_decoder_postnet_kernel,
            stride=1,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)

        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.speecht5.modeling_speecht5.SpeechT5SpeechDecoderPostnet with SpeechT5->FastSpeech2Conformer
class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)

        should_postnet_compute_logits = getattr(config, "should_postnet_compute_logits", True)
        if should_postnet_compute_logits:
            self.prob_out = nn.Linear(config.hidden_size, config.reduction_factor)
        else:
            self.prob_out = None

        self.layers = nn.ModuleList(
            [FastSpeech2ConformerBatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
        )

    def forward(self, hidden_states: torch.Tensor):
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
        outputs_after_postnet = self.postnet(outputs_before_postnet)
        if self.prob_out is not None:
            logits = self.prob_out(hidden_states).view(hidden_states.size(0), -1)
        else:
            logits = None
        return outputs_before_postnet, outputs_after_postnet, logits

    def postnet(self, hidden_states: torch.Tensor):
        layer_output = hidden_states.transpose(1, 2)
        for layer in self.layers:
            layer_output = layer(layer_output)
        return hidden_states + layer_output.transpose(1, 2)


class FastSpeech2ConformerLayerNorm(torch.nn.LayerNorm):
    """Wrapper for torch.nn.LayerNorm that enables normalization along a specified dimension.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an FastSpeech2ConformerLayerNorm object."""
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super().forward(x)
        return super().forward(x.transpose(self.dim, -1)).transpose(self.dim, -1)


class FastSpeech2ConformerVariancePredictor(nn.Module):
    """
    Variance predictor module.

    This is a module of variance predictor described in `FastSpeech 2: Fast and High-Quality End-to-End Text to
    Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        input_dim,
        n_layers=2,
        n_chans=384,
        kernel_size=3,
        bias=True,
        dropout_rate=0.5,
    ):
        """
        Initilize duration predictor module.

        Args:
            input_dim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        for idx in range(n_layers):
            input_channels = input_dim if idx == 0 else n_chans
            layer = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        input_channels,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    FastSpeech2ConformerLayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                ]
            )
            self.conv_layers.append(layer)
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, encoder_hidden_states, padding_masks=None):
        """
        Calculate forward propagation.

        Args:
            encoder_hidden_states (Tensor): Batch of input sequences (batch_size, max_text_length, input_dim).
            padding_masks (ByteTensor, optional):
                Batch of masks indicating padded part (batch_size, max_text_length).

        Returns:
            Tensor: Batch of predicted sequences (batch_size, max_text_length, 1).
        """
        # (batch_size, input_dim, max_text_length)
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            for module in layer:
                hidden_states = module(hidden_states)

        hidden_states = self.linear(hidden_states.transpose(1, 2))

        if padding_masks is not None:
            hidden_states = hidden_states.masked_fill(padding_masks, 0.0)

        return hidden_states


class FastSpeech2ConformerVarianceEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=384,
        kernel_size=1,
        padding=0,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class FastSpeech2ConformerVarianceDurationPredictorLoss(torch.nn.Module):
    """
    Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, log_domain_offset=1.0, reduction="mean"):
        """
        Args:
            log_domain_offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(FastSpeech2ConformerVarianceDurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.log_domain_offset = log_domain_offset

    def forward(self, duration_predictions, target_durations):
        """
        Calculate forward propagation.

        Args:
            duration_predictions (Tensor): Batch of prediction durations in log domain (batch_size, T)
            target_durations (LongTensor): Batch of groundtruth durations in linear domain (batch_size, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `duration_predictions` is in log domain but `target_durations` is in linear domain.

        """
        # NOTE: duration_predictions is in log domain while target_durations in linear
        target_durations = torch.log(target_durations.float() + self.log_domain_offset)
        loss = self.criterion(duration_predictions, target_durations)

        return loss


class FastSpeech2ConformerLoss(torch.nn.Module):
    def __init__(self, use_masking=True, use_weighted_masking=False):
        """
        use_masking (bool):
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool):
            Whether to weighted masking in loss calculation.
        """
        super().__init__()

        if use_masking and use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")

        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = FastSpeech2ConformerVarianceDurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        outputs_after_postnet,
        outputs_before_postnet,
        duration_outputs,
        pitch_outputs,
        energy_outputs,
        target_spectrograms,
        target_durations,
        target_pitch,
        target_energy,
        input_lengths,
        target_lengths,
    ):
        """
        Args:
            outputs_after_postnet (Tensor):
                Batch of outputs after postnet (batch_size, max_spectrogram_length, num_mel_bins).
            outputs_before_postnet (Tensor):
                Batch of outputs before postnet (batch_size, max_spectrogram_length, num_mel_bins).
            duration_outputs (LongTensor): Batch of outputs of duration predictor (batch_size, max_text_length).
            pitch_outputs (Tensor): Batch of outputs of pitch predictor (batch_size, max_text_length, 1).
            energy_outputs (Tensor): Batch of outputs of energy predictor (batch_size, max_text_length, 1).
            target_spectrograms (Tensor): Batch of target features (batch_size, max_spectrogram_length, num_mel_bins).
            target_durations (LongTensor): Batch of durations (batch_size, max_text_length).
            target_pitch (Tensor): Batch of target token-averaged pitch (batch_size, max_text_length, 1).
            target_energy (Tensor): Batch of target token-averaged energy (batch_size, max_text_length, 1).
            input_lengths (LongTensor): Batch of the lengths of each input (batch_size,).
            target_lengths (LongTensor): Batch of the lengths of each target (batch_size,).

        Returns:
            Tensor: L1 loss value. Tensor: Duration predictor loss value. Tensor: Pitch predictor loss value. Tensor:
            Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(target_lengths).unsqueeze(-1).to(target_spectrograms.device)
            outputs_before_postnet = outputs_before_postnet.masked_select(out_masks)
            if outputs_after_postnet is not None:
                outputs_after_postnet = outputs_after_postnet.masked_select(out_masks)
            target_spectrograms = target_spectrograms.masked_select(out_masks)
            duration_masks = make_non_pad_mask(input_lengths).to(target_spectrograms.device)
            duration_outputs = duration_outputs.masked_select(duration_masks)
            target_durations = target_durations.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(input_lengths).unsqueeze(-1).to(target_spectrograms.device)
            pitch_outputs = pitch_outputs.masked_select(pitch_masks)
            energy_outputs = energy_outputs.masked_select(pitch_masks)
            target_pitch = target_pitch.masked_select(pitch_masks)
            target_energy = target_energy.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(outputs_before_postnet, target_spectrograms)
        if outputs_after_postnet is not None:
            l1_loss = l1_loss + self.l1_criterion(outputs_after_postnet, target_spectrograms)
        duration_loss = self.duration_criterion(duration_outputs, target_durations)
        pitch_loss = self.mse_criterion(pitch_outputs, target_pitch)
        energy_loss = self.mse_criterion(energy_outputs, target_energy)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(target_lengths).unsqueeze(-1).to(target_spectrograms.device)
            out_masks = torch.nn.functional.pad(
                out_masks.transpose(1, 2),
                [0, target_spectrograms.size(1) - out_masks.size(1), 0, 0, 0, 0],
                value=False,
            ).transpose(1, 2)

            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= target_spectrograms.size(0) * target_spectrograms.size(2)
            duration_masks = make_non_pad_mask(input_lengths).to(target_spectrograms.device)
            duration_weights = duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            duration_weights /= target_durations.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()

        return l1_loss, duration_loss, pitch_loss, energy_loss


class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastSpeech2ConformerConfig
    base_model_prefix = "fastspeech2_conformer"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    main_input_name = "input_ids"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_()
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FastSpeech2ConformerRelPositionMultiHeadedAttention):
            nn.init.xavier_uniform_(module.pos_bias_u)
            nn.init.xavier_uniform_(module.pos_bias_v)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2ConformerEncoder):
            module.gradient_checkpointing = value


FASTSPEECH2_CONFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

add_start_docstrings(FASTSPEECH2_CONFORMER_START_DOCSTRING)
class FastSpeech2ConformerModel(FastSpeech2ConformerPreTrainedModel):
    """
    FastSpeech 2 module.

    This is a module of FastSpeech 2 described in FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.
    Instead of quantized pitch and energy, we use token-averaged value introduced in FastPitch: Parallel Text-to-speech
    with Pitch Prediction. The encoder and decoder are Conformers instead of regular Transformers.

        https://arxiv.org/abs/2006.04558 https://arxiv.org/abs/2006.06873 https://arxiv.org/pdf/2005.08100

    """

    def __init__(self, config):
        # config.utt_embed_dim=None,  # confirm this, was 64 in IMS Toucan
        # config.lang_embs=None,      # confirm this, was 8000 in IMS Toucan
        super().__init__(config)
        self.config = config

        # store hyperparameters
        self.input_dim = config.input_dim
        self.num_mel_bins = config.num_mel_bins
        self.hidden_size = config.hidden_size
        self.eos_token_id = self.input_dim - 1
        self.reduction_factor = config.reduction_factor
        self.stop_gradient_from_pitch_predictor = config.stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = config.stop_gradient_from_energy_predictor
        self.multilingual_model = config.lang_embs is not None
        self.multispeaker_model = config.utt_embed_dim is not None

        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=self.input_dim, embedding_dim=self.hidden_size, padding_idx=0
        )
        self.encoder = FastSpeech2ConformerEncoder(
            attention_dim=self.hidden_size,
            attention_heads=config.num_attention_heads,
            linear_units=config.encoder_linear_units,
            num_blocks=config.encoder_layers,
            input_layer=encoder_input_layer,
            dropout_rate=config.encoder_dropout_rate,
            positional_dropout_rate=config.encoder_positional_dropout_rate,
            attention_dropout_rate=config.encoder_attention_dropout_rate,
            normalize_before=config.encoder_normalize_before,
            concat_after=config.encoder_concat_after,
            positionwise_conv_kernel_size=config.positionwise_conv_kernel_size,
            macaron_style=config.use_macaron_style_in_conformer,
            use_cnn_module=config.use_cnn_in_conformer,
            cnn_module_kernel=config.encoder_kernel_size,
            utt_embed=config.utt_embed_dim,
            lang_embs=config.lang_embs,
        )

        self.duration_predictor = FastSpeech2ConformerDurationPredictor(
            input_dim=self.hidden_size,
            n_layers=config.duration_predictor_layers,
            n_chans=config.duration_predictor_channels,
            kernel_size=config.duration_predictor_kernel_size,
            dropout_rate=config.duration_predictor_dropout_rate,
        )

        self.pitch_predictor = FastSpeech2ConformerVariancePredictor(
            input_dim=self.hidden_size,
            n_layers=config.pitch_predictor_layers,
            n_chans=config.pitch_predictor_channels,
            kernel_size=config.pitch_predictor_kernel_size,
            dropout_rate=config.pitch_predictor_dropout,
        )
        # continuous pitch + FastPitch style avg
        self.pitch_embed = FastSpeech2ConformerVarianceEmbedding(
            out_channels=self.hidden_size,
            kernel_size=config.pitch_embed_kernel_size,
            padding=(config.pitch_embed_kernel_size - 1) // 2,
            dropout_rate=config.pitch_embed_dropout,
        )

        self.energy_predictor = FastSpeech2ConformerVariancePredictor(
            input_dim=self.hidden_size,
            n_layers=config.energy_predictor_layers,
            n_chans=config.energy_predictor_channels,
            kernel_size=config.energy_predictor_kernel_size,
            dropout_rate=config.energy_predictor_dropout,
        )
        # continuous energy + FastPitch style avg
        self.energy_embed = FastSpeech2ConformerVarianceEmbedding(
            out_channels=self.hidden_size,
            kernel_size=config.energy_embed_kernel_size,
            padding=(config.energy_embed_kernel_size - 1) // 2,
            dropout_rate=config.energy_embed_dropout,
        )

        self.length_regulator = FastSpeech2ConformerLengthRegulator()

        # The decoder is an encoder
        self.decoder = FastSpeech2ConformerEncoder(
            attention_dim=self.hidden_size,
            attention_heads=config.num_attention_heads,
            linear_units=config.decoder_linear_units,
            num_blocks=config.decoder_layers,
            input_layer=None,
            dropout_rate=config.decoder_dropout_rate,
            positional_dropout_rate=config.decoder_positional_dropout_rate,
            attention_dropout_rate=config.decoder_attention_dropout_rate,
            normalize_before=config.decoder_normalize_before,
            concat_after=config.decoder_concat_after,
            positionwise_conv_kernel_size=config.positionwise_conv_kernel_size,
            macaron_style=config.use_macaron_style_in_conformer,
            use_cnn_module=config.use_cnn_in_conformer,
            cnn_module_kernel=config.decoder_kernel_size,
            utt_embed=config.utt_embed_dim,
        )

        self.speech_decoder_postnet = FastSpeech2ConformerSpeechDecoderPostnet(config)

        self.criterion = FastSpeech2ConformerLoss(
            use_masking=config.use_masking, use_weighted_masking=config.use_weighted_masking
        )

        self.post_init()

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FastSpeech2ConformerModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: Optional[torch.LongTensor] = None,
        target_spectrograms: Optional[torch.FloatTensor] = None,
        spectrogram_lengths: Optional[torch.LongTensor] = None,
        target_durations: Optional[torch.LongTensor] = None,
        target_pitch: Optional[torch.FloatTensor] = None,
        target_energy: Optional[torch.FloatTensor] = None,
        utterance_embedding: Optional[torch.FloatTensor] = None,
        alpha: Optional[float] = 1.0,
        lang_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            input_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Batch of lengths of each input sequence.
            target_spectrograms (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*):
                Batch of padded target features.
            spectrogram_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Batch of the lengths of each target spectrogram.
            target_durations (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*):
                Batch of padded durations.
            target_pitch (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*):
                Batch of padded token-averaged pitch.
            target_energy (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*):
                Batch of padded token-averaged energy.
            utterance_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`, *optional*):
                Tensor containing the utterance embeddings.
            alpha (`float`, *optional*):
                Alpha to control the speed of the spectrogram generation. Default is `1.0`.
            lang_id (`torch.LongTensor`, *optional*):
                Language id to condition the model.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`ModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:
        
        Example:

        ```python
        >>> from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan

        >>> tokenizer = FastSpeech2ConformerTokenizer.from_pretrained('connor-henderson/fastspeech2_conformer')
        >>> inputs = tokenizer("some text to convert to speech", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]

        >>> model = FastSpeech2ConformerModel.from_pretrained('connor-henderson/fastspeech2_conformer')
        >>> output_dict = model(input_ids, return_dict=True)
        >>> spectrogram = output_dict["spectrogram"]

        >>> vocoder = FastSpeech2ConformerHifiGan.from_pretrained('connor-henderson/fastspeech2_conformer_hifigan')
        >>> waveform = vocoder(spectrogram)
        >>> print(waveform.shape)
        torch.Size([1, 49664])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        input_lengths = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)

        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)
        if utterance_embedding is not None:
            utterance_embedding.unsqueeze(0)

        is_inference = target_spectrograms is None
        # Texts include EOS token from the teacher model already in this version
        outputs = self._forward(
            input_ids,
            input_lengths,
            spectrogram_lengths,
            target_durations=target_durations,
            target_pitch=target_pitch,
            target_energy=target_pitch,
            utterance_embedding=utterance_embedding,
            is_inference=is_inference,
            lang_ids=lang_id,
            alpha=alpha,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        (
            outputs_before_postnet,
            outputs_after_postnet,
            encoder_last_hidden_state,
            encoder_hidden_states,
            encoder_attentions,
            decoder_hidden_states,
            decoder_attentions,
            duration_outputs,
            pitch_outputs,
            energy_outputs,
        ) = outputs

        # modify mod part of groundtruth (speaking pace)
        if self.reduction_factor > 1:
            spectrogram_lengths = spectrogram_lengths.new(
                [original_length - original_length % self.reduction_factor for original_length in spectrogram_lengths]
            )

        loss = None
        if not is_inference:
            # calculate loss
            l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
                outputs_after_postnet=outputs_after_postnet,
                outputs_before_postnet=outputs_before_postnet,
                duration_outputs=duration_outputs,
                pitch_outputs=pitch_outputs,
                energy_outputs=energy_outputs,
                target_spectrograms=target_spectrograms,
                target_durations=target_durations,
                target_pitch=target_pitch,
                target_energy=target_energy,
                input_lengths=input_lengths,
                target_lengths=spectrogram_lengths,
            )
            loss = l1_loss + duration_loss + pitch_loss + energy_loss

        if not return_dict:
            outputs = tuple(output for output in outputs[1:] if output is not None)
            return ((loss,) + outputs) if loss is not None else outputs

        return FastSpeech2ConformerModelOutput(
            loss=loss,
            spectrogram=outputs_after_postnet,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=encoder_attentions,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attentions=decoder_attentions,
            duration_outputs=duration_outputs,
            pitch_outputs=pitch_outputs,
            energy_outputs=energy_outputs,
        )

    def _forward(
        self,
        input_ids,
        input_lengths,
        spectrogram_lengths=None,
        target_durations=None,
        target_pitch=None,
        target_energy=None,
        is_inference=False,
        alpha=1.0,
        utterance_embedding=None,
        lang_ids=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward encoder
        text_masks = self._source_mask(input_lengths)

        encoder_outputs = self.encoder(
            input_ids,
            text_masks,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        encoder_last_hidden_state = encoder_outputs.last_hidden_state

        # forward duration predictor and variance predictors
        duration_masks = make_pad_mask(input_lengths, device=input_lengths.device)

        if self.stop_gradient_from_pitch_predictor:
            pitch_predictions = self.pitch_predictor(encoder_last_hidden_state.detach(), duration_masks.unsqueeze(-1))
        else:
            pitch_predictions = self.pitch_predictor(encoder_last_hidden_state, duration_masks.unsqueeze(-1))

        if self.stop_gradient_from_energy_predictor:
            energy_predictions = self.energy_predictor(
                encoder_last_hidden_state.detach(), duration_masks.unsqueeze(-1)
            )
        else:
            energy_predictions = self.energy_predictor(encoder_last_hidden_state, duration_masks.unsqueeze(-1))

        duration_predictions = self.duration_predictor(
            encoder_last_hidden_state, duration_masks, is_inference=is_inference
        )

        if is_inference:
            # use prediction in inference
            embedded_pitch_curve = self.pitch_embed(pitch_predictions)
            embedded_energy_curve = self.energy_embed(energy_predictions)
            encoder_last_hidden_state = encoder_last_hidden_state + embedded_energy_curve + embedded_pitch_curve
            encoder_last_hidden_state = self.length_regulator(encoder_last_hidden_state, duration_predictions, alpha)
        else:
            # use groundtruth in training
            embedded_pitch_curve = self.pitch_embed(target_pitch)
            embedded_energy_curve = self.energy_embed(target_energy)
            encoder_last_hidden_state = encoder_last_hidden_state + embedded_energy_curve + embedded_pitch_curve
            encoder_last_hidden_state = self.length_regulator(encoder_last_hidden_state, target_durations)

        # forward decoder
        if spectrogram_lengths is not None and not is_inference:
            if self.reduction_factor > 1:
                target_lengths_in = spectrogram_lengths.new(
                    [original_length // self.reduction_factor for original_length in spectrogram_lengths]
                )
            else:
                target_lengths_in = spectrogram_lengths
            h_masks = self._source_mask(target_lengths_in)
        else:
            h_masks = None

        decoder_outputs = self.decoder(
            encoder_last_hidden_state,
            h_masks,
            utterance_embedding,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        outputs_before_postnet, outputs_after_postnet, _ = self.speech_decoder_postnet(
            decoder_outputs.last_hidden_state
        )

        return (
            outputs_before_postnet,
            outputs_after_postnet,
            encoder_last_hidden_state,
            encoder_outputs.hidden_states,
            encoder_outputs.attentions,
            decoder_outputs.hidden_states,
            decoder_outputs.attentions,
            duration_predictions,
            pitch_predictions,
            energy_predictions,
        )

    def _source_mask(self, input_lengths):
        """
        Make masks for self-attention.

        Args:
            input_lengths (LongTensor): Batch of lengths (batch_size,).

        Returns:
            Tensor: Mask tensor for self-attention.

        """
        x_masks = make_non_pad_mask(input_lengths, device=input_lengths.device)
        return x_masks.unsqueeze(-2)


class FastSpeech2ConformerMultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """
        Construct an FastSpeech2ConformerMultiHeadedAttention object.
        """
        super().__init__()

        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """
        Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (batch, time1, size).
            key (torch.Tensor): Key tensor (batch, time2, size).
            value (torch.Tensor): Value tensor (batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (batch, n_head, time1, d_k). torch.Tensor: Transformed key tensor
            (batch, n_head, time2, d_k). torch.Tensor: Transformed value tensor (batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """
        Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (batch, 1, time2) or (batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (batch, time1, d_model)
                weighted by the attention score (batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            # (batch_size, 1, *, time2)
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            # (batch_size, head, time1, time2)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            # (batch_size, head, time1, time2)
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        # (batch_size, head, time1, d_k)
        x = torch.matmul(p_attn, value)
        # (batch_size, time1, d_model)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        attention_output = self.linear_out(x)

        return attention_output, attn

    def forward(self, query, key, value, mask):
        """
        Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (batch, time1, size).
            key (torch.Tensor): Key tensor (batch, time2, size).
            value (torch.Tensor): Value tensor (batch, time2, size).
            mask (torch.Tensor): Mask tensor (batch, 1, time2) or
                (batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class FastSpeech2ConformerRelPositionMultiHeadedAttention(FastSpeech2ConformerMultiHeadedAttention):
    """
    Args:
    Multi-Head Attention layer with relative position encoding. Details can be found in
    https://github.com/espnet/espnet/pull/2816. Paper: https://arxiv.org/abs/1901.02860
        n_head (int): The number of heads. n_feat (int): The number of features. dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an FastSpeech2ConformerRelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))

    def rel_shift(self, x):
        """
        Args:
        Compute relative positional encoding.
            x (torch.Tensor): Input tensor (batch_size, head, time1, 2*time1-1). time1 means the length of query
            vector.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        # only keep the positions from 0 to time2
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """
        Args:
        Compute 'Scaled Dot Product Attention' with rel. positional encoding.
            query (torch.Tensor): Query tensor (batch, time1, size). key (torch.Tensor): Key tensor (batch, time2,
            size). value (torch.Tensor): Value tensor (batch, time2, size). pos_emb (torch.Tensor): Positional
            embedding tensor
                (batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (batch, 1, time2) or
                (batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        # (batch_size, time1, head, d_k)
        q = q.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # (batch_size, head, 2*time1-1, d_k)
        p = p.transpose(1, 2)

        # (batch_size, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch_size, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch_size, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch_size, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        # (batch_size, head, time1, time2)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        attention_output, attention_scores = self.forward_attention(v, scores, mask)

        return attention_output, attention_scores


class FastSpeech2ConformerConvolutionModule(nn.Module):
    """
    FastSpeech2ConformerEncoderConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()
        # kernel_size should be an odd number for 'SAME' padding

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, hidden_states):
        """
        Compute convolution module.

        Args:
            hidden_states (torch.Tensor): Input tensor (batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism, (batch_size, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # (batch_size, channel, dim)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)

        # This is currently a manual swish activation instead of using nn.functional.silu()
        # since it caused a slight (~1e-6) divergence from the original implementation
        # in ESPnet, please advise which is preferred
        hidden_states = hidden_states * torch.sigmoid(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)

        return hidden_states.transpose(1, 2)


class FastSpeech2ConformerEncoderLayer(nn.Module):
    """
    Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `FastSpeech2ConformerMultiHeadedAttention` or `FastSpeech2ConformerRelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `FastSpeech2ConformerMultiLayeredConv1d`, or `Conv1dLinear` instance can be used
            as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `FastSpeech2ConformerMultiLayeredConv1d`, or `Conv1dLinear` instance can be used
            as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x))) if False, no additional
            linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        super().__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        # for the FNN module
        self.norm_ff = FastSpeech2ConformerLayerNorm(size)

        # for the MHA module
        self.norm_mha = FastSpeech2ConformerLayerNorm(size)

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = FastSpeech2ConformerLayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            # for the CNN module
            self.norm_conv = FastSpeech2ConformerLayerNorm(size)

            # for the final output of the block
            self.norm_final = FastSpeech2ConformerLayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, hidden_states, pos_emb, mask):
        """
        Compute encoded features.

        Args:
            hidden_states (torch.Tensor): Tensor (batch, time, size).
            pos_emb (torch.Tensor): Tensor (1, time, size).
            mask (torch.Tensor): Mask tensor for the input (batch, time).

        Returns:
            torch.Tensor: Output tensor (batch, time, size). torch.Tensor: Mask tensor (batch, time).

        """
        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.norm_ff_macaron(hidden_states)
            hidden_states = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(hidden_states))
            if not self.normalize_before:
                hidden_states = self.norm_ff_macaron(hidden_states)

        # multi-headed self-attention module
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.norm_mha(hidden_states)

        if pos_emb is not None:
            attention_output, attention_scores = self.self_attn(
                hidden_states, hidden_states, hidden_states, pos_emb, mask
            )
        else:
            attention_output, attention_scores = self.self_attn(hidden_states, hidden_states, hidden_states, mask)

        if self.concat_after:
            x_concat = torch.cat((hidden_states, attention_output), dim=-1)
            hidden_states = residual + self.concat_linear(x_concat)
        else:
            hidden_states = residual + self.dropout(attention_output)
        if not self.normalize_before:
            hidden_states = self.norm_mha(hidden_states)

        # convolution module
        if self.conv_module is not None:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.norm_conv(hidden_states)
            hidden_states = residual + self.dropout(self.conv_module(hidden_states))
            if not self.normalize_before:
                hidden_states = self.norm_conv(hidden_states)

        # feed forward module
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.norm_ff(hidden_states)
        hidden_states = residual + self.ff_scale * self.dropout(self.feed_forward(hidden_states))
        if not self.normalize_before:
            hidden_states = self.norm_ff(hidden_states)

        if self.conv_module is not None:
            hidden_states = self.norm_final(hidden_states)

        return hidden_states, attention_scores, pos_emb, mask


class FastSpeech2ConformerMultiLayeredConv1d(nn.Module):
    """
    Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed to replace positionwise feed-forward network in Transformer
    block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, dropout_rate):
        """
        Initialize FastSpeech2ConformerMultiLayeredConv1d module.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.conv2 = torch.nn.Conv1d(
            hidden_channels,
            input_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        """
        Calculate forward propagation.

        Args:
            hidden_states (torch.Tensor): Batch of input tensors (batch_size, T, input_channels).

        Returns:
            torch.Tensor: Batch of output tensors (batch_size, T, hidden_channels).
        """
        hidden_states = torch.relu(self.conv1(hidden_states.transpose(-1, 1))).transpose(-1, 1)
        return self.conv2(self.dropout(hidden_states).transpose(-1, 1)).transpose(-1, 1)


class FastSpeech2ConformerRelPositionalEncoding(nn.Module):
    """
    Args:
    Relative positional encoding module (new implementation). Details can be found in
    https://github.com/espnet/espnet/pull/2816. See : Appendix Batch in https://arxiv.org/abs/1901.02860
        d_model (int): Embedding dimension. dropout_rate (float): Dropout rate. max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """
        Construct an PositionalEncoding object.
        """
        super().__init__()
        self.d_model = d_model
        self.input_scale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, feature_representation):
        """
        Args:
        Add positional encoding.
            feature_representation (torch.Tensor): Input tensor (batch_size, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch_size, time, `*`).
        """
        self.extend_pe(feature_representation)
        hidden_states = feature_representation * self.input_scale
        center_idx = self.pe.size(1) // 2
        pos_emb = self.pe[:, center_idx - hidden_states.size(1) + 1 : center_idx + hidden_states.size(1)]
        return self.dropout(hidden_states), self.dropout(pos_emb)


class FastSpeech2ConformerEncoder(nn.Module):
    """
    FastSpeech2ConformerEncoder encoder module.

    Args:
        input_dim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x))) if False, no additional
            linear will be applied. i.e. x -> x + att(x)
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        activation_type (str): Conformer activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernel size of convolution module.

    """

    def __init__(
        self,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer=None,
        normalize_before=True,
        concat_after=False,
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        use_cnn_module=False,
        cnn_module_kernel=31,
        utt_embed=None,
        lang_embs=None,
    ):
        super().__init__()

        self.embed = input_layer
        self.pos_enc = FastSpeech2ConformerRelPositionalEncoding(attention_dim, positional_dropout_rate)

        self.utt_embed = utt_embed
        if utt_embed is not None:
            self.hs_emb_projection = torch.nn.Linear(attention_dim + utt_embed, attention_dim)
        if lang_embs is not None:
            self.language_embedding = torch.nn.Embedding(num_embeddings=lang_embs, embedding_dim=attention_dim)

        # self-attention module definition
        encoder_selfattn_layer = FastSpeech2ConformerRelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)

        # feed-forward module definition
        positionwise_layer = FastSpeech2ConformerMultiLayeredConv1d
        positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate)

        # convolution module definition
        convolution_layer = FastSpeech2ConformerConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        self.conformer_layers = nn.ModuleList(
            [
                FastSpeech2ConformerEncoderLayer(
                    attention_dim,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        input_tensor,
        masks,
        utterance_embedding=None,
        lang_ids=None,
        output_hidden_states=None,
        output_attentions=None,
    ):
        """
        Args:
        Encode input sequence.
            input_tensor (torch.Tensor): Input tensor (batch, time, input_dim). masks (torch.Tensor): Mask tensor
            (batch, time). utterance_embedding: embedding containing lots of conditioning signals lang_ids: ids of the
            languages per sample in the batch output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        Returns:
            torch.Tensor: Output tensor (batch, time, attention_dim). torch.Tensor: Mask tensor (batch, time).
        """
        feature_representation = input_tensor
        if self.embed is not None:
            feature_representation = self.embed(feature_representation)
        if lang_ids is not None:
            lang_embs = self.language_embedding(lang_ids)
            # offset phoneme representation by language specific offset
            feature_representation = feature_representation + lang_embs

        hidden_states, pos_emb = self.pos_enc(feature_representation)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for conformer_layer in self.conformer_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attention_output, pos_emb, masks = conformer_layer(hidden_states, pos_emb, masks)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attention_output,)

        if self.utt_embed:
            hidden_states = self._integrate_with_utt_embed(hidden_states, utterance_embedding)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )

    def _integrate_with_utt_embed(self, hidden_states, utt_embeddings):
        # concat hidden states with spk embeds and then apply projection
        embeddings_expanded = (
            torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        )
        hidden_states = self.hs_emb_projection(torch.cat([hidden_states, embeddings_expanded], dim=-1))
        return hidden_states


# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states

HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    """HiFi-GAN vocoder.""",
    HIFIGAN_START_DOCSTRING,
)
# Copied from transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan with SpeechT5->FastSpeech2Conformer
class FastSpeech2ConformerHifiGan(PreTrainedModel):
    config_class = FastSpeech2ConformerHifiGanConfig
    main_input_name = "spectrogram"

    def __init__(self, config: FastSpeech2ConformerHifiGanConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        if config.normalize_before:
            self.register_buffer("mean", torch.zeros(config.model_in_dim))
            self.register_buffer("scale", torch.ones(config.model_in_dim))

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)

    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        if self.config.normalize_before:
            spectrogram = (spectrogram - self.mean) / self.scale

        is_batched = spectrogram.dim() == 3
        if not is_batched:
            spectrogram = spectrogram.unsqueeze(0)

        hidden_states = spectrogram.transpose(2, 1)

        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # remove seq-len dim since this collapses to 1
            waveform = hidden_states.squeeze(1)

        return waveform
