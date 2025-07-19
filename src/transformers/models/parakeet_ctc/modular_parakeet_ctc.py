# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils.generic import can_return_tuple
from ..fastconformer import FastConformerEncoder
from .configuration_parakeet_ctc import ParakeetCTCConfig


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class ParakeetCTCDecoder(nn.Module):
    """
    CTC decoder for Parakeet models.

    This decoder implements Connectionist Temporal Classification (CTC) decoding
    for speech recognition. It consists of a linear projection layer that maps
    encoder hidden states to vocabulary logits.

    Args:
        config (ParakeetCTCConfig): Configuration containing decoder parameters
    """

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__()
        self.config = config

        # CTC head - linear projection from encoder hidden size to vocabulary size
        self.ctc_head = nn.Linear(config.encoder_config.hidden_size, config.vocab_size)

        # Store CTC-specific parameters for easy access
        self.blank_token_id = config.blank_token_id
        self.ctc_loss_reduction = config.ctc_loss_reduction
        self.ctc_zero_infinity = config.ctc_zero_infinity

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CTC decoder.

        Args:
            hidden_states: Encoder output of shape (batch_size, sequence_length, hidden_size)

        Returns:
            CTC logits of shape (batch_size, sequence_length, vocab_size)
        """
        return self.ctc_head(hidden_states)

    def compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            logits: Model predictions of shape (batch_size, sequence_length, vocab_size)
            labels: Target labels of shape (batch_size, max_label_length)
            input_lengths: Actual lengths of input sequences
            label_lengths: Actual lengths of label sequences

        Returns:
            CTC loss value
        """
        # Convert logits to log probabilities and transpose for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (sequence_length, batch_size, vocab_size)

        # Prepare targets by removing padding and special tokens
        targets = []
        for i, label_length in enumerate(label_lengths):
            label = labels[i, :label_length]
            label = label[label != -100]  # Remove padding tokens
            targets.append(label)

        targets = torch.cat(targets)

        # Compute CTC loss
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=label_lengths,
            blank=self.blank_token_id,
            reduction=self.ctc_loss_reduction,
            zero_infinity=self.ctc_zero_infinity,
        )

        return loss


class ParakeetCTCPreTrainedModel(PreTrainedModel):
    config_class = ParakeetCTCConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FastConformerBlock"]
    _skip_keys_device_placement = []

    def _init_weights(self, module):
        # Get initializer_range from the encoder config
        std = self.config.encoder_config.initializer_range

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


class ParakeetCTC(ParakeetCTCPreTrainedModel):
    """
    ParakeetCTC model for CTC-based speech recognition.

    This model follows an encoder-decoder architecture where:
    - Encoder: FastConformer model for processing audio features
    - Decoder: CTC decoder for speech recognition output

    Args:
        config (ParakeetCTCConfig): Model configuration
    """

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__(config)

        # Encoder: FastConformer for audio feature processing
        self.encoder = FastConformerEncoder(config.encoder_config)

        # Decoder: CTC decoder for speech recognition
        self.decoder = ParakeetCTCDecoder(config)

        # Initialize weights
        self.post_init()

    def get_encoder(self):
        """Get the encoder component."""
        return self.encoder

    def set_encoder(self, encoder):
        """Set the encoder component."""
        self.encoder = encoder

    def get_decoder(self):
        """Get the decoder component."""
        return self.decoder

    def set_decoder(self, decoder):
        """Set the decoder component."""
        self.decoder = decoder

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, CausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.encoder_config.use_return_dict

        # Forward through encoder
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Forward through decoder to get CTC logits
        logits = self.decoder(hidden_states)

        loss = None
        if labels is not None:
            # Calculate encoder output lengths
            if input_lengths is not None:
                encoder_lengths = calc_length(
                    input_lengths.float(),
                    all_paddings=2,
                    kernel_size=3,
                    stride=2,
                    ceil_mode=False,
                    repeat_num=int(math.log2(self.config.encoder_config.subsampling_factor)),
                )
                encoder_lengths = encoder_lengths.long()
            elif attention_mask is not None:
                input_lens = attention_mask.sum(-1).float()
                encoder_lengths = calc_length(
                    input_lens,
                    all_paddings=2,
                    kernel_size=3,
                    stride=2,
                    ceil_mode=False,
                    repeat_num=int(math.log2(self.config.encoder_config.subsampling_factor)),
                )
                encoder_lengths = encoder_lengths.long()
            else:
                encoder_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)

            # Calculate label lengths (excluding padding and blank tokens)
            label_lengths = torch.sum((labels != -100) & (labels != self.decoder.blank_token_id), dim=-1)

            # Compute CTC loss using the decoder
            loss = self.decoder.compute_ctc_loss(
                logits=logits,
                labels=labels,
                input_lengths=encoder_lengths,
                label_lengths=label_lengths,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        """
        Generate CTC decoded token sequences using greedy decoding.

        Args:
            input_features: Input mel-spectrogram features
            attention_mask: Attention mask
            input_lengths: Sequence lengths

        Returns:
            List of decoded token sequences (one per batch item)
        """
        with torch.no_grad():
            # Forward pass to get logits
            outputs = self.forward(
                input_features=input_features,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                return_dict=True,
            )

            logits = outputs.logits  # (batch, time, vocab)

            # Greedy CTC decoding
            predicted_ids = torch.argmax(logits, dim=-1)  # (batch, time)

            batch_size = predicted_ids.size(0)
            decoded_sequences = []

            for batch_idx in range(batch_size):
                sequence = predicted_ids[batch_idx]

                # Get actual sequence length if available
                if input_lengths is not None:
                    # Calculate the actual output length after subsampling
                    actual_length = calc_length(
                        input_lengths[batch_idx : batch_idx + 1].float(),
                        all_paddings=2,
                        kernel_size=3,
                        stride=2,
                        ceil_mode=False,
                        repeat_num=int(math.log2(self.config.encoder_config.subsampling_factor)),
                    ).item()
                    sequence = sequence[:actual_length]
                elif attention_mask is not None:
                    # Use attention mask to determine length
                    input_len = attention_mask[batch_idx].sum().float()
                    actual_length = calc_length(
                        input_len.unsqueeze(0),
                        all_paddings=2,
                        kernel_size=3,
                        stride=2,
                        ceil_mode=False,
                        repeat_num=int(math.log2(self.config.encoder_config.subsampling_factor)),
                    ).item()
                    sequence = sequence[:actual_length]

                # CTC collapse: remove blanks and repeated tokens
                decoded_tokens = []
                prev_token = None

                for token_id in sequence.tolist():
                    # Skip blank tokens (using the decoder's blank token ID)
                    if token_id == self.decoder.blank_token_id:
                        prev_token = token_id
                        continue

                    # Skip repeated tokens (CTC collapse)
                    if token_id != prev_token:
                        decoded_tokens.append(token_id)

                    prev_token = token_id

                decoded_sequences.append(decoded_tokens)

            return decoded_sequences


__all__ = ["ParakeetCTC", "ParakeetCTCDecoder", "ParakeetCTCPreTrainedModel"]
