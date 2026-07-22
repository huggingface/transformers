# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Modeling classes for MossTTSDelay."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    auto_docstring,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel
from ..qwen3 import Qwen3Config
from .processing_moss_tts_delay import AssistantMessage, MossTTSDelayProcessor, UserMessage


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MossTTSDelayConfig"


@auto_docstring(checkpoint="OpenMOSS-Team/MOSS-TTS-v1.5")
@strict
class MossTTSDelayConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MossTTSDelayModel`]. It is used to instantiate an
    MossTTSDelay model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MossTTSDelay [MossTTSDelay-8B](https://huggingface.co/OpenMOSS/mosstts-8b) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        language_config (`Union[Qwen3Config, dict]`, *optional*):
            Configuration for the backbone language model (Qwen3).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        n_codebooks (`int`, *optional*, defaults to 32):
            Number of additional VQ (Vector Quantization) heads/channels for audio.
            Determines the number of codebooks used in the audio representation.
        pad_token_id (`int`, *optional*, defaults to 151643):
            Padding token id for the text channel.
        im_start_token_id (`int`, *optional*, defaults to 151644):
            Token id used to mark the beginning of a chat message.
        im_end_token_id (`int`, *optional*, defaults to 151645):
            Token id used to mark the end of a chat message.
        codebook_size (`int`, *optional*, defaults to 1024):
            Vocabulary size for the audio tokens (codebooks 1 to N).
        audio_user_slot_token_id (`int`, *optional*, defaults to 151654):
            The specific token ID used as a placeholder/slot for user-side audio inputs in the prompt.
        audio_assistant_gen_slot_token_id (`int`, *optional*, defaults to 151656):
            The specific token ID representing the generation slot for the assistant's audio output.
            Acting as the trigger for the TTS generation process.
        audio_assistant_delay_slot_token_id (`int`, *optional*, defaults to 151662):
            The token ID used in the 'Delay Pattern' paradigm to represent the delayed/offset positions
            between different VQ channels.
        audio_start_token_id (`int`, *optional*, defaults to 151652):
            Special token ID used to denote the start of an audio sequence in the stream.
        audio_end_token_id (`int`, *optional*, defaults to 151653):
            Special token ID used to denote the end of an audio sequence (EOS for audio).
        codebook_pad_token_id (`int`, *optional*, defaults to 1024):
            The padding value used within the audio VQ codebooks. Typically equals `codebook_size`.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Audio sampling rate used by the processor and audio tokenizer.
    """

    model_type = "moss_tts_delay"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"language_config": Qwen3Config}

    def __init__(
        self,
        language_config: Qwen3Config | dict | None = None,
        initializer_range: float = 0.02,
        n_codebooks: int = 32,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        codebook_size: int = 1024,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_gen_slot_token_id: int = 151656,
        audio_assistant_delay_slot_token_id: int = 151662,
        audio_start_token_id: int = 151652,
        audio_end_token_id: int = 151653,
        codebook_pad_token_id: int = 1024,
        sampling_rate: int = 24000,
        **kwargs,
    ):
        r"""
        language_config (`Qwen3Config` or `dict`, *optional*):
            Configuration for the backbone language model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation used to initialize weights.
        n_codebooks (`int`, *optional*, defaults to 32):
            Number of audio VQ codebook channels.
        pad_token_id (`int`, *optional*, defaults to 151643):
            Padding token id for the text channel.
        im_start_token_id (`int`, *optional*, defaults to 151644):
            Token id used to mark the beginning of a chat message.
        im_end_token_id (`int`, *optional*, defaults to 151645):
            Token id used to mark the end of a chat message.
        codebook_size (`int`, *optional*, defaults to 1024):
            Vocabulary size for each audio codebook.
        audio_user_slot_token_id (`int`, *optional*, defaults to 151654):
            Placeholder token id for user-side audio inputs.
        audio_assistant_gen_slot_token_id (`int`, *optional*, defaults to 151656):
            Placeholder token id that triggers assistant audio generation.
        audio_assistant_delay_slot_token_id (`int`, *optional*, defaults to 151662):
            Token id used for delayed audio-code positions.
        audio_start_token_id (`int`, *optional*, defaults to 151652):
            Token id that marks the start of an audio segment.
        audio_end_token_id (`int`, *optional*, defaults to 151653):
            Token id that marks the end of an audio segment.
        codebook_pad_token_id (`int`, *optional*, defaults to 1024):
            Padding code used in audio VQ channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Audio sampling rate used by the processor and audio tokenizer.
        """
        if isinstance(language_config, dict):
            self.language_config = Qwen3Config(**language_config)
        elif language_config is None:
            self.language_config = Qwen3Config()
        else:
            self.language_config = language_config

        self.initializer_range = initializer_range
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_assistant_gen_slot_token_id = audio_assistant_gen_slot_token_id
        self.audio_assistant_delay_slot_token_id = audio_assistant_delay_slot_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.codebook_pad_token_id = codebook_pad_token_id
        self.sampling_rate = sampling_rate

        self.hidden_size = self.language_config.hidden_size
        self.vocab_size = self.language_config.vocab_size
        self.pad_token_id = pad_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        if hasattr(self.language_config, "to_dict"):
            output["language_config"] = self.language_config.to_dict()
        else:
            output["language_config"] = self.language_config
        return output


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    top_k = min(top_k, vocab_size)
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    filtered_logits = torch.full_like(logits, float("-inf"))
    batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(-1)
    filtered_logits[batch_indices, top_k_indices] = top_k_values
    return filtered_logits


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits[indices_to_remove] = float("-inf")
    return logits


def _apply_repetition_penalty(
    logits: torch.Tensor,
    prev_tokens: torch.LongTensor | None,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0 or prev_tokens is None:
        return logits

    if logits.dim() == 2:
        unique_tokens = torch.unique(prev_tokens.reshape(-1))
        token_logits = logits[:, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, unique_tokens] = token_logits
        return logits

    for head_idx in range(logits.shape[1]):
        unique_tokens = torch.unique(prev_tokens[..., head_idx].reshape(-1))
        if unique_tokens.numel() == 0:
            continue

        token_logits = logits[:, head_idx, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, head_idx, unique_tokens] = token_logits

    return logits


def _sample_token(
    logits: torch.Tensor,
    prev_tokens: torch.LongTensor | None = None,
    repetition_penalty: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    do_sample: bool = True,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    logits = _apply_repetition_penalty(logits, prev_tokens, repetition_penalty)

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    original_shape = logits.shape
    reshaped_logits = logits.view(-1, vocab_size)

    if top_k is not None and top_k > 0:
        reshaped_logits = _apply_top_k(reshaped_logits, top_k)

    if top_p is not None and top_p < 1.0:
        reshaped_logits = _apply_top_p(reshaped_logits, top_p)

    probs = F.softmax(reshaped_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens.view(original_shape[:-1])


def _find_last_equal(tensor: torch.Tensor, value: int) -> torch.Tensor:
    mask = (tensor == value).int()
    flipped_indices = mask.flip(dims=[1]).argmax(dim=1)
    last_indices = (tensor.shape[1] - 1) - flipped_indices
    no_match = tensor[torch.arange(tensor.shape[0], device=tensor.device), last_indices] != value
    last_indices[no_match] = -1
    return last_indices


@dataclass
class MossTTSDelayOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Weighted sum of channel losses.
        all_sum_losses (`torch.FloatTensor` of shape `(batch_size, n_codebooks + 1)`, *optional*):
            Sum of losses for each sample and each channel before averaging.
        all_token_nums (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of non-masked tokens per sample.
        sample_losses (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Loss per sample.
        channel_losses (`torch.FloatTensor` of shape `(n_codebooks + 1,)`, *optional*):
            Loss per channel (text head + vq heads).
        logits (`List[torch.FloatTensor]`, *optional*):
            List of prediction scores from each head.
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer).
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of torch.FloatTensor (one for each layer) of the attention weights.
    """

    loss: torch.FloatTensor | None = None
    all_sum_losses: torch.FloatTensor | None = None
    all_token_nums: torch.LongTensor | None = None
    sample_losses: torch.FloatTensor | None = None
    channel_losses: torch.FloatTensor | None = None
    logits: list[torch.FloatTensor] | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


class MossTTSDelayPreTrainedModel(PreTrainedModel):
    config_class = MossTTSDelayConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """
        Transformers 5.0+ safe init:
        - MUST use transformers.initialization helpers
        - MUST respect param._is_hf_initialized to avoid overwriting ckpt-loaded params
        """
        # Let HF handle its standard modules first (LayerNorm, Linear, Embedding, etc.)
        super()._init_weights(module)

        # Pick a std consistent with HF conventions
        # Prefer model/text config initializer_range if present.
        std = None
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        elif hasattr(self.config, "language_config") and hasattr(self.config.language_config, "initializer_range"):
            std = self.config.language_config.initializer_range
        else:
            std = 0.02

        # Initialize extra audio embeddings
        if isinstance(module, nn.Embedding):
            # Only touch our extra embeddings (avoid double touching LM's embeddings if not desired)
            # If you prefer, you can skip this check and rely on super()._init_weights for all embeddings.
            if getattr(module, "num_embeddings", None) == self.config.codebook_size + 1:
                init.normal_(module.weight, mean=0.0, std=std)
                # If you later set padding_idx, you must explicitly zero it (and respect _is_hf_initialized!)
                # init.zeros_ will internally check param flags, but slicing needs manual care.

        # Initialize multi-head projections you added
        if isinstance(module, nn.Linear):
            # For your lm_heads, super()._init_weights already covers typical Linear.
            # This block is only needed if you have custom Linear variants later.
            pass


MOSSTTS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MossTTSDelayConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The MossTTSDelay Model architecture tailored for Text-to-Speech generation with multi-head VQ prediction.",
    MOSSTTS_START_DOCSTRING,
)
class MossTTSDelayModel(MossTTSDelayPreTrainedModel):
    UserMessage = UserMessage
    AssistantMessage = AssistantMessage
    Processor = MossTTSDelayProcessor

    def __init__(self, config: MossTTSDelayConfig):
        super().__init__(config)
        self.config = config

        config.language_config.dtype = config.dtype

        self.language_model = AutoModel.from_config(config.language_config)

        # Audio VQ Embeddings (Extra channels)
        # Note: input_ids[..., 0] uses Qwen's embedding.
        # input_ids[..., 1:] use these extensions.
        self.emb_ext = nn.ModuleList()
        for vq_idx in range(self.config.n_codebooks):
            # Add +1 for potential padding/special tokens logic if strictly required by upstream data prep
            self.emb_ext.append(
                nn.Embedding(self.config.codebook_size + 1, config.language_config.hidden_size, padding_idx=None)
            )

        # Multi-Head Prediction Layers
        # Head 0: Main language head
        # Head 1..N: Audio VQ heads
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.language_config.hidden_size, config.language_config.vocab_size, bias=False)]
        )
        for vq_idx in range(self.config.n_codebooks):
            self.lm_heads.append(
                nn.Linear(config.language_config.hidden_size, self.config.codebook_size + 1, bias=False)
            )

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def get_input_embeddings(self, input_ids: torch.LongTensor | None = None) -> torch.Tensor:
        """
        Computes the combined embeddings from text and multiple audio VQ channels.

        Args:
            input_ids: Shape (Batch, Seq_Len, 1 + n_codebooks)
        """
        if input_ids is None:
            return self.language_model.get_input_embeddings()

        # Base Text/Content Embedding
        # input_ids[..., 0] is standard text or semantic tokens
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids[..., 0])

        # Add VQ Embeddings
        for i, embed_layer in enumerate(self.emb_ext):
            # i corresponds to channel i+1 in input_ids
            # We assume the data pipeline ensures indices are within range
            inputs_embeds = inputs_embeds + embed_layer(input_ids[..., i + 1])

        return inputs_embeds

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        # Returning a list of heads might break some HF utilities expecting a single head.
        # However, for custom models, this is acceptable.
        return self.lm_heads

    @add_start_docstrings_to_model_forward(MOSSTTS_START_DOCSTRING)
    @replace_return_docstrings(output_type=MossTTSDelayOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        hidden_out_layers: list[int] | None = None,
        channelwise_loss_weight: list[float] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MossTTSDelayOutputWithPast:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, 1 + n_codebooks)`):
                Indices of input sequence tokens in the vocabulary.
                Dimension 2 contains: [Text/Semantics, VQ_0, VQ_1, ..., VQ_N].
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, 1 + n_codebooks)`, *optional*):
                Labels for computing the masked language modeling loss.
            channelwise_loss_weight (`List[float]`, *optional*):
                Manual weights for summing losses across different heads (Text vs Audio channels).

        Returns:
        """

        if len(input_ids.shape) != 3 or input_ids.shape[-1] != self.config.n_codebooks + 1:
            raise ValueError("`Input_ids`'s shape should be exactly (batch_size, sequence_length, 1 + n_codebooks).")

        # 1. Prepare Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        # 2. Backbone Forward
        # Qwen3Model outputs standard CausalLMOutputWithPast or similar
        outputs = self.language_model(
            input_ids=None,  # Passed via inputs_embeds
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=True,  # Always need hidden states for multi-head projection
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        # 3. Handle specific layer outputs if requested (Delay Pattern often requires features from specific layers)
        last_hidden_state = outputs.last_hidden_state
        if hidden_out_layers is None:
            # Default to using the last layer for all heads
            # In some architectures (like MusicGen), different codebooks come from different transformer layers.
            # Here we default to the final layer as per original code behavior [-1] * (n + 1).
            hidden_states_for_heads = [last_hidden_state] * (len(self.lm_heads))
        else:
            # If hidden_out_layers is provided (e.g. [-1, -2, -3...]), fetch them from all_hidden_states
            # Note: outputs.hidden_states includes embedding output at index 0 usually.
            all_hs = outputs.hidden_states
            hidden_states_for_heads = [all_hs[idx] for idx in hidden_out_layers]

        # 4. Project to Logits (Multi-Head)
        layer_logits = []
        for i, (hs, head) in enumerate(zip(hidden_states_for_heads, self.lm_heads)):
            logits = head(hs)
            # Original code logic: Mask the last token index for audio heads (indices > 0)
            # This implies the vocab size is (N+1) but the model shouldn't predict the (N+1)-th token
            # (perhaps reserved for padding in the input but invalid for prediction).
            if i > 0:
                logits[..., -1] = float("-inf")
            layer_logits.append(logits)

        # 5. Loss Calculation
        loss = None
        all_sum_losses = None
        all_token_nums = None
        sample_losses = None
        channel_losses = None

        if labels is not None:
            # Ensure labels match input shape rank (B, S, C)
            if labels.dim() != 3:
                raise ValueError(f"Labels must have rank 3 (B, S, C), got {labels.shape}")

            batch_size = labels.size(0)
            n_heads = len(layer_logits)

            # Container for per-sample, per-channel losses
            # Shape: [Batch, n_heads]
            all_sum_losses_list = []

            # Count valid tokens (not -100) per sample.
            # Note: Assuming mask is consistent across channels or we take sum over dim 1 (seq)
            # Usually strict masking means checking one channel or all.
            # Original code: torch.sum(labels != -100, dim=1) -> [B, C]
            all_token_nums = torch.sum(labels != -100, dim=1)

            for i, logits in enumerate(layer_logits):
                # logits: [B, S, V]
                # cur_labels: [B, S]
                cur_labels = labels[..., i]

                # Flatten for CrossEntropy
                # logits: [B*S, V], labels: [B*S]
                loss_fct = CrossEntropyLoss(reduction="none")
                vocab_size = logits.size(-1)

                reshaped_logits = logits.view(-1, vocab_size)
                reshaped_labels = cur_labels.contiguous().view(-1)

                # Calculate loss per token
                per_token_loss = loss_fct(reshaped_logits, reshaped_labels)

                # Reshape back to [B, S] and sum over Sequence dimension to get per-sample loss
                per_token_loss = per_token_loss.view(batch_size, -1)
                per_sample_loss = torch.sum(per_token_loss, dim=-1)  # [B]

                all_sum_losses_list.append(per_sample_loss)

            # Stack to [B, n_heads]
            all_sum_losses = torch.stack(all_sum_losses_list, dim=1)

            # Weighted Loss Aggregation
            if channelwise_loss_weight is not None:
                if len(channelwise_loss_weight) != n_heads:
                    raise ValueError(f"channelwise_loss_weight length {len(channelwise_loss_weight)} != {n_heads}")

                w_tensor = torch.tensor(
                    channelwise_loss_weight, device=all_sum_losses.device, dtype=all_sum_losses.dtype
                )

                # Sample losses: Weighted sum over channels per sample / Total weight
                # Normalize by token count per channel
                # Avoid division by zero with epsilon or mask
                token_counts_safe = all_token_nums.float().clamp(min=1.0)

                normalized_losses = all_sum_losses / token_counts_safe
                sample_losses = (normalized_losses * w_tensor).sum(dim=1) / w_tensor.sum()

                # Channel losses: Sum over batch / Sum tokens over batch
                total_loss_per_channel = all_sum_losses.sum(dim=0)
                total_tokens_per_channel = all_token_nums.sum(dim=0).float().clamp(min=1.0)
                channel_losses = total_loss_per_channel / total_tokens_per_channel

                # Final scalar loss
                loss = (channel_losses * w_tensor).sum() / w_tensor.sum()
            else:
                # Default average if no weights provided
                total_tokens = all_token_nums.sum().float().clamp(min=1.0)
                loss = all_sum_losses.sum() / total_tokens
                channel_losses = all_sum_losses.sum(dim=0) / all_token_nums.sum(dim=0).clamp(min=1.0)

        return MossTTSDelayOutputWithPast(
            loss=loss,
            all_sum_losses=all_sum_losses,
            all_token_nums=all_token_nums,
            sample_losses=sample_losses,
            channel_losses=channel_losses,
            logits=layer_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1000,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
    ):
        if text_temperature > 0:
            text_do_sample = True
        else:
            text_temperature = 1
            text_do_sample = False
        if audio_temperature > 0:
            audio_do_sample = True
        else:
            audio_temperature = 1
            audio_do_sample = False

        past_key_values = None
        device = input_ids.device
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        batch_size, seq_len, n_codebooks = input_ids.shape
        n_codebooks -= 1

        generation_ids = input_ids[:]
        is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)

        audio_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        torch_int64_max = torch.iinfo(torch.int64).max
        delayed_lengths = torch.full((batch_size,), torch_int64_max, dtype=torch.int64, device=device)

        is_continuation = (input_ids[:, -1, 0] == self.config.audio_start_token_id) | (
            input_ids[:, -1, 0] == self.config.audio_assistant_gen_slot_token_id
        )
        audio_start_indices = _find_last_equal(input_ids[..., 0], self.config.audio_start_token_id)
        audio_start_mask = is_continuation & (audio_start_indices != -1)
        audio_lengths[audio_start_mask] = seq_len - audio_start_indices[audio_start_mask]

        is_audio = audio_start_mask.clone()

        pre_exclude_mask0 = torch.tensor(
            [
                self.config.pad_token_id,
                self.config.audio_assistant_gen_slot_token_id,
                self.config.audio_assistant_delay_slot_token_id,
                self.config.audio_end_token_id,
            ],
            device=device,
        )
        pre_exclude_mask1 = torch.ones(self.config.language_config.vocab_size, device=device).bool()
        pre_exclude_mask1[
            [self.config.audio_assistant_gen_slot_token_id, self.config.audio_assistant_delay_slot_token_id]
        ] = False

        for time_step in tqdm(range(max_new_tokens), desc=f"Generating bs{batch_size} ..."):
            outputs = self(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            next_token_logits = [
                logit[:, -1, :] / text_temperature if logit_idx == 0 else logit[:, -1, :] / audio_temperature
                for logit_idx, logit in enumerate(outputs.logits)
            ]  # List, len=n_codebooks+1, [batch_size, 1, vocab_size];
            next_token_logits[0] = next_token_logits[0].clone()
            next_text_token = torch.full((batch_size,), self.config.pad_token_id, device=device)
            next_text_token[~is_stopping & (delayed_lengths < n_codebooks)] = (
                self.config.audio_assistant_delay_slot_token_id
            )
            is_audio_eos = ~is_stopping & (delayed_lengths == n_codebooks)
            next_text_token[is_audio_eos] = self.config.audio_end_token_id
            is_audio[is_audio_eos] = False
            sampling_text_mask = ~is_stopping & (delayed_lengths > n_codebooks)
            next_token_logits[0][~is_audio] = next_token_logits[0][~is_audio].index_fill(
                -1, pre_exclude_mask0, float("-inf")
            )
            next_token_logits[0][is_audio] = next_token_logits[0][is_audio].masked_fill(
                pre_exclude_mask1, float("-inf")
            )
            if time_step == 0:
                next_token_logits[0][..., 151662] = float("-inf")
            if time_step <= n_codebooks:
                next_token_logits[0][..., self.config.im_end_token_id] = float("-inf")

            next_text_token[sampling_text_mask] = _sample_token(
                logits=next_token_logits[0][sampling_text_mask],
                top_p=text_top_p,
                top_k=text_top_k,
                do_sample=text_do_sample,
            )
            is_audio[next_text_token == self.config.audio_start_token_id] = True
            is_stopping[next_text_token == self.config.im_end_token_id] = True

            next_audio_tokens = torch.full((batch_size, n_codebooks), self.config.codebook_pad_token_id, device=device)

            pre_audio_mask = audio_lengths.unsqueeze(1) > torch.arange(n_codebooks, dtype=int, device=device).expand(
                batch_size, n_codebooks
            )
            post_audio_mask = (
                torch.arange(n_codebooks, dtype=int, device=device).expand(batch_size, n_codebooks)
                > delayed_lengths.unsqueeze(1) - 1
            )
            post_audio_mask[delayed_lengths == torch_int64_max] = True
            sampling_audio_mask = pre_audio_mask & post_audio_mask
            next_audio_tokens[~sampling_audio_mask] = self.config.codebook_pad_token_id

            if sampling_audio_mask.sum() > 0:
                audio_ch0_logits = next_token_logits[1][sampling_audio_mask[:, 0]]
                audio_logits = torch.stack(next_token_logits[2:], dim=1)[sampling_audio_mask[:, 1:]]
                audio_ch0_logits[..., self.config.codebook_pad_token_id] = float("-inf")
                audio_logits[..., self.config.codebook_pad_token_id] = float("-inf")
                next_audio_tokens[:, 0][sampling_audio_mask[:, 0]] = _sample_token(
                    logits=audio_ch0_logits,
                    prev_tokens=generation_ids[:, :, 1],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                    do_sample=audio_do_sample,
                )
                next_audio_tokens[:, 1:][sampling_audio_mask[:, 1:]] = _sample_token(
                    logits=audio_logits,
                    prev_tokens=generation_ids[:, :, 2:],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p,
                    top_k=audio_top_k,
                    do_sample=audio_do_sample,
                )

            audio_lengths[
                (next_text_token == self.config.audio_start_token_id)
                | (next_text_token == self.config.audio_assistant_gen_slot_token_id)
                | (next_text_token == self.config.audio_assistant_delay_slot_token_id)
            ] += 1
            audio_lengths[next_text_token == self.config.audio_end_token_id] = 0
            delayed_lengths[
                (delayed_lengths == torch_int64_max)
                & (next_text_token == self.config.audio_assistant_delay_slot_token_id)
            ] = 0
            delayed_lengths[delayed_lengths != torch_int64_max] += 1
            delayed_lengths[delayed_lengths > n_codebooks] = torch_int64_max

            current_input_ids = torch.cat([next_text_token[:, None, None], next_audio_tokens[:, None, :]], dim=2)
            current_attention_mask = torch.cat([current_attention_mask, (~is_stopping).unsqueeze(-1)], dim=-1)
            generation_ids = torch.cat([generation_ids, current_input_ids], dim=1)

            if is_stopping.sum() == batch_size:
                break

        start_indices = _find_last_equal(input_ids[..., 0], self.config.im_start_token_id) + 3
        start_lengths = seq_len - start_indices

        output = []
        for start_idx, start_length, cur_generation_ids in zip(start_indices, start_lengths, generation_ids):
            output.append((start_length, cur_generation_ids[start_idx:]))

        return output


__all__ = ["MossTTSDelayConfig", "MossTTSDelayModel", "MossTTSDelayOutputWithPast", "MossTTSDelayPreTrainedModel"]
