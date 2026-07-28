# Copyright 2026 The SwissAI Initiative, The Emu team, BAAI, and The HuggingFace Inc. team. All rights reserved.
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
"""Apertus 1.5: a discrete-token early-fusion multimodal model (image + audio + text -> text).

The model composes an Apertus language backbone with two frozen, encode-only discrete tokenizers whose codes are
mapped into the shared text vocabulary by plain offsets: a bundled vision tokenizer (`Apertus1p5VisionTokenizerModel`) and the
standalone WavTokenizer audio codec (`AutoModel`-resolved from the `audio_tokenizer_config`).

The vision tokenizer is an encode-only port of the EMU3.5 Vision Tokenizer by BAAI
(*Emu3.5: Native Multimodal Models are World Learners*, https://huggingface.co/papers/2510.26583; weights and
original code at https://huggingface.co/BAAI/Emu3.5-VisionTokenizer, Apache-2.0). See the `Apertus1p5VisionTokenizerModel`
docstring for the port's scope (inference-only, no decoder) and its float32 requirement.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ..apertus.configuration_apertus import ApertusConfig
from ..apertus.modeling_apertus import ApertusForCausalLM, ApertusModel, ApertusPreTrainedModel
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel


def _pad_logits_to_vocab_size(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Pad a physically pruned LM-head output with non-generatable input-only token scores."""
    padding = vocab_size - logits.shape[-1]
    if padding == 0:
        return logits
    # `finfo.min` rather than `-inf`: the tail still softmaxes to exactly 0 probability, but stays finite under
    # score arithmetic (classifier-free guidance subtracts two score sets, and `-inf - -inf` is NaN)
    return F.pad(logits, (0, padding), value=torch.finfo(logits.dtype).min)


def _pruned_output_vocab_size(config: PreTrainedConfig) -> int | None:
    """Return the physical LM-head width when it is pruned below the vocabulary, else `None`."""
    text_config = config.get_text_config()
    output_vocab_size = getattr(text_config, "output_vocab_size", None)
    if output_vocab_size is not None and output_vocab_size != text_config.vocab_size:
        return output_vocab_size
    return None


def _check_pruned_head_resize(config: PreTrainedConfig) -> None:
    """Reject embedding resizes for pruned heads: the generic resize forces the head to the embedding size."""
    if _pruned_output_vocab_size(config) is not None:
        raise NotImplementedError(
            "Resizing token embeddings is not supported for a pruned LM head (`output_vocab_size` is "
            "set): resize the unpruned checkpoint and prune it again."
        )


def _check_pruned_head_tie(config: PreTrainedConfig) -> None:
    """Reject weight tying for pruned heads: the generic tying installs the full-width embeddings as the head."""
    text_config = config.get_text_config()
    tied = getattr(config, "tie_word_embeddings", False) or getattr(text_config, "tie_word_embeddings", False)
    if tied and _pruned_output_vocab_size(config) is not None:
        raise ValueError(
            "Cannot tie a pruned LM head (`output_vocab_size` smaller than `vocab_size`) to the input "
            "embeddings; keep `tie_word_embeddings=False`."
        )


def _check_pruned_head_labels(labels: torch.Tensor, out_features: int, vocab_size: int) -> None:
    """Reject labels beyond a pruned head: cross entropy would fail with a bare index error otherwise."""
    if out_features < vocab_size:
        torch_compilable_check(
            (labels < out_features).all(),
            f"`labels` must be smaller than `output_vocab_size` ({out_features}) or equal to -100; with a pruned "
            "LM head, positions holding multimodal or other input-only ids must be masked with -100.",
        )


@auto_docstring(checkpoint="swiss-ai/Apertus-1.5-8B")
@strict
class Apertus1p5VisionTokenizerConfig(PreTrainedConfig):
    r"""
    codebook_size (`int`, *optional*, defaults to 131072):
        Number of entries in the IBQ codebook.
    embed_dim (`int`, *optional*, defaults to 256):
        Dimension of the codebook vectors (the quantizer input after `quant_conv`).
    latent_channels (`int`, *optional*, defaults to 256):
        Number of channels output by the encoder (before `quant_conv`).
    in_channels (`int`, *optional*, defaults to 3):
        Number of input image channels.
    base_channels (`int`, *optional*, defaults to 256):
        Base channel count of the encoder; per-stage widths are `base_channels * channel_multiplier[stage]`.
    channel_multiplier (`list[int]`, *optional*, defaults to `(1, 1, 2, 2, 4)`):
        Channel scaling factor per encoder stage. The number of stages is `len(channel_multiplier)`, giving a
        spatial downsampling factor of `2**(len(channel_multiplier) - 1)` (16 by default).
    num_res_blocks (`int`, *optional*, defaults to 4):
        Number of residual blocks per encoder stage.
    attn_resolutions (`list[int]`, *optional*, defaults to `(16,)`):
        Feature-map resolutions (relative to a `resolution`-sized input) at which attention blocks are inserted.
    resolution (`int`, *optional*, defaults to 256):
        Reference input resolution used (statically, together with `attn_resolutions`) to decide which encoder
        stages carry attention blocks. Does not restrict the actual input size.
    dropout (`float`, *optional*, defaults to 0.0):
        Dropout inside the residual blocks.

    Example:

    ```python
    >>> from transformers import Apertus1p5VisionTokenizerConfig, Apertus1p5VisionTokenizerModel

    >>> # Initializing a vision tokenizer configuration (Apertus 1.5 defaults)
    >>> configuration = Apertus1p5VisionTokenizerConfig()

    >>> # Initializing the encode-only tokenizer (with random weights) from the configuration
    >>> model = Apertus1p5VisionTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "apertus1p5_vision_tokenizer"
    base_config_key = "vision_tokenizer_config"

    codebook_size: int = 131072
    embed_dim: int = 256
    latent_channels: int = 256
    in_channels: int = 3
    base_channels: int = 256
    channel_multiplier: list[int] | tuple[int, ...] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 4
    attn_resolutions: list[int] | tuple[int, ...] = (16,)
    resolution: int = 256
    dropout: float = 0.0

    @property
    def spatial_scale_factor(self) -> int:
        return 2 ** (len(self.channel_multiplier) - 1)


@auto_docstring(checkpoint="swiss-ai/Apertus-1.5-8B")
@strict
class Apertus1p5TextConfig(ApertusConfig):
    r"""
    output_vocab_size (`int`, *optional*):
        Number of LM-head rows kept after pruning the multimodal token rows from the output projection; the
        retained output ids are `0..output_vocab_size - 1`. `None` means the head is unpruned (`vocab_size`
        rows). Input embeddings always use `vocab_size`, and logits returned without `labels` are padded to that
        logical width with `torch.finfo(dtype).min` scores for the input-only tail. A pruned head cannot be
        tied to the input embeddings.

    Example:

    ```python
    >>> from transformers import Apertus1p5TextConfig, Apertus1p5TextForCausalLM

    >>> # Initializing an Apertus 1.5 text configuration (extended vocabulary, optionally pruned output head)
    >>> configuration = Apertus1p5TextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Apertus1p5TextForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "apertus1p5_text"
    base_config_key = "text_config"

    # the extended vocabulary covers text plus the visual and audio code ranges; all other hyperparameters
    # are read from the checkpoint config and keep the Apertus defaults otherwise
    vocab_size: int = 266752
    output_vocab_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.output_vocab_size is not None:
            if not 1 <= self.output_vocab_size <= self.vocab_size:
                raise ValueError(
                    f"`output_vocab_size` ({self.output_vocab_size}) must be in `[1, vocab_size]` "
                    f"(vocab_size is {self.vocab_size})."
                )
            if self.output_vocab_size == self.vocab_size:
                # a full-width head is unpruned; normalizing keeps resize/tie semantics consistent
                self.output_vocab_size = None
            elif self.tie_word_embeddings:
                raise ValueError(
                    "A pruned LM head (`output_vocab_size` smaller than `vocab_size`) cannot be tied to the "
                    "input embeddings; set `tie_word_embeddings=False`."
                )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="swiss-ai/Apertus-1.5-8B")
@strict
class Apertus1p5Config(PreTrainedConfig):
    r"""
    text_config (`Union[dict, PreTrainedConfig]`, *optional*):
        Configuration of the Apertus language backbone. The extended vocabulary (text + visual + audio tokens)
        lives in `text_config.vocab_size`, which sizes the input embedding table. The LM head uses
        `text_config.output_vocab_size` physical rows when set, otherwise it uses the full
        `text_config.vocab_size`. Model outputs have `text_config.vocab_size` logits (loss-only calls with
        `labels` keep the physical width); input-only ids beyond the physical LM head are padded with
        `torch.finfo(dtype).min` scores and cannot be selected by unconstrained generation.
    vision_tokenizer_config (`Union[dict, Apertus1p5VisionTokenizerConfig]`, *optional*):
        Configuration of the bundled EMU3.5-derived vision tokenizer.
    audio_tokenizer_config (`Union[dict, PreTrainedConfig]`, *optional*):
        Configuration of the WavTokenizer audio codec.
    image_token_id (`int`, *optional*, defaults to 131079):
        Id of the `<|image|>` placeholder token that image code positions carry in `input_ids`.
    audio_token_id (`int`, *optional*, defaults to 131085):
        Id of the `<|audio|>` placeholder token that audio code positions carry in `input_ids`.
    image_token_offset (`int`, *optional*, defaults to 131272):
        Offset mapping a visual codebook index `i` to the vocabulary id of `<|visual token i|>`.
    audio_token_offset (`int`, *optional*, defaults to 262344):
        Offset mapping an audio codebook index `i` to the vocabulary id of `<|audio token i|>`.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to tie the LM head weights to the input embeddings.

    Example:

    ```python
    >>> from transformers import Apertus1p5Config, Apertus1p5ForConditionalGeneration

    >>> # Initializing a configuration (Apertus backbone + vision/audio tokenizer defaults)
    >>> configuration = Apertus1p5Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Apertus1p5ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "apertus1p5"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "text_config": AutoConfig,
        "vision_tokenizer_config": Apertus1p5VisionTokenizerConfig,
        "audio_tokenizer_config": AutoConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_tokenizer_config: dict | Apertus1p5VisionTokenizerConfig | None = None
    audio_tokenizer_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 131079
    audio_token_id: int = 131085
    image_token_offset: int = 131272
    audio_token_offset: int = 262344
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "apertus1p5_text")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            # the extended vocabulary covers the visual and audio token ranges
            self.text_config = Apertus1p5TextConfig()

        if isinstance(self.vision_tokenizer_config, dict):
            self.vision_tokenizer_config = Apertus1p5VisionTokenizerConfig(**self.vision_tokenizer_config)
        elif self.vision_tokenizer_config is None:
            self.vision_tokenizer_config = Apertus1p5VisionTokenizerConfig()

        if isinstance(self.audio_tokenizer_config, dict):
            self.audio_tokenizer_config["model_type"] = self.audio_tokenizer_config.get("model_type", "wavtokenizer")
            self.audio_tokenizer_config = CONFIG_MAPPING[self.audio_tokenizer_config["model_type"]](
                **self.audio_tokenizer_config
            )
        elif self.audio_tokenizer_config is None:
            self.audio_tokenizer_config = CONFIG_MAPPING["wavtokenizer"]()

        if self.image_token_offset + self.vision_tokenizer_config.codebook_size != self.audio_token_offset:
            raise ValueError(
                "The visual token range must end where the audio token range begins: expected "
                f"`image_token_offset + codebook_size == audio_token_offset`, got {self.image_token_offset} + "
                f"{self.vision_tokenizer_config.codebook_size} != {self.audio_token_offset}."
            )
        audio_vocab_end = self.audio_token_offset + self.audio_tokenizer_config.codebook_size
        if audio_vocab_end > self.text_config.vocab_size:
            raise ValueError(
                f"The audio token range ends at {audio_vocab_end}, beyond the vocabulary "
                f"(`text_config.vocab_size` = {self.text_config.vocab_size})."
            )
        output_vocab_size = getattr(self.text_config, "output_vocab_size", None)
        if (
            self.tie_word_embeddings
            and output_vocab_size is not None
            and output_vocab_size != self.text_config.vocab_size
        ):
            raise ValueError(
                "A pruned LM head (`text_config.output_vocab_size` set) cannot be tied to the input "
                "embeddings; set `tie_word_embeddings=False`."
            )
        super().__post_init__(**kwargs)


@auto_docstring
@dataclass
class Apertus1p5VisionTokenizerModelOutput(BaseModelOutputWithPooling):
    r"""
    pooler_output (`tuple[torch.FloatTensor]`):
        One `(num_codes, hidden_size)` tensor of embedded image codes per input image.
    image_tokens (`list[torch.LongTensor]`):
        One grid of discrete codebook indices per input image, of shape `(height // factor, width // factor)`
        where `factor` is `config.spatial_scale_factor` (16 by default).
    """

    image_tokens: list[torch.LongTensor] | None = None


@auto_docstring
@dataclass
class Apertus1p5AudioTokenizerModelOutput(BaseModelOutputWithPooling):
    r"""
    pooler_output (`tuple[torch.FloatTensor]`):
        One `(num_codes, hidden_size)` tensor of embedded audio codes per input clip.
    audio_tokens (`list[torch.LongTensor]`):
        One 1D tensor of `ceil(length / hop)` discrete codebook indices per input clip.
    """

    audio_tokens: list[torch.LongTensor] | None = None


class Apertus1p5VisionTokenizerResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states


class Apertus1p5VisionTokenizerAttnBlock(nn.Module):
    """Single-head self-attention over spatial positions with 1x1 convolutions.

    The manual `bmm` attention (instead of the library attention interface) preserves bitwise parity with the
    original tokenizer, whose argmax code assignment is sensitive to tiny numeric differences.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        batch_size, channels, height, width = query.shape
        query = query.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        key = key.reshape(batch_size, channels, height * width)
        attn_weights = torch.bmm(query, key) * channels**-0.5
        attn_weights = F.softmax(attn_weights, dim=2)

        value = value.reshape(batch_size, channels, height * width)
        hidden_states = torch.bmm(value, attn_weights.permute(0, 2, 1))
        hidden_states = hidden_states.reshape(batch_size, channels, height, width)
        return residual + self.proj_out(hidden_states)


class Apertus1p5VisionTokenizerDownsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # asymmetric right/bottom padding, as in the original VQGAN-style tokenizer
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Apertus1p5VisionTokenizerEncoder(nn.Module):
    def __init__(self, config: Apertus1p5VisionTokenizerConfig):
        super().__init__()
        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks

        self.conv_in = nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, stride=1, padding=1)

        # attention placement is decided statically from the reference `resolution`, as in the original
        current_resolution = config.resolution
        in_channel_multiplier = (1,) + tuple(config.channel_multiplier)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.base_channels * in_channel_multiplier[i_level]
            block_out = config.base_channels * config.channel_multiplier[i_level]
            for _ in range(self.num_res_blocks):
                block.append(Apertus1p5VisionTokenizerResnetBlock(block_in, block_out, dropout=config.dropout))
                block_in = block_out
                if current_resolution in config.attn_resolutions:
                    attn.append(Apertus1p5VisionTokenizerAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Apertus1p5VisionTokenizerDownsample(block_in)
                current_resolution = current_resolution // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = Apertus1p5VisionTokenizerResnetBlock(block_in, block_in, dropout=config.dropout)
        self.mid.attn_1 = Apertus1p5VisionTokenizerAttnBlock(block_in)
        self.mid.block_2 = Apertus1p5VisionTokenizerResnetBlock(block_in, block_in, dropout=config.dropout)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, config.latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(pixel_values)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hidden_states = self.down[i_level].block[i_block](hidden_states)
                if len(self.down[i_level].attn) > 0:
                    hidden_states = self.down[i_level].attn[i_block](hidden_states)
            if i_level != self.num_resolutions - 1:
                hidden_states = self.down[i_level].downsample(hidden_states)

        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        return self.conv_out(hidden_states)


class Apertus1p5VisionTokenizerVectorQuantizer(nn.Module):
    """
    IBQ codebook lookup, from *Scalable Image Tokenization with Index Backpropagation Quantization*
    (https://huggingface.co/papers/2412.02692). IBQ differs from plain VQ in how the codebook is trained; at
    inference, quantization reduces to a dot-product similarity argmax over the codebook. This class implements
    only that inference path, not IBQ's differentiable index-backpropagation training path.
    """

    def __init__(self, config: Apertus1p5VisionTokenizerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.LongTensor:
        # hidden_states: (batch_size, embed_dim, height, width)
        logits = torch.einsum("bdhw,nd->bnhw", hidden_states, self.embedding.weight)
        return logits.argmax(dim=1)  # (batch_size, height, width)


@auto_docstring(
    custom_intro="""
    The Apertus 1.5 vision tokenizer: an encode-only port of the EMU3.5 Vision Tokenizer by BAAI
    (*Emu3.5: Native Multimodal Models are World Learners*, https://huggingface.co/papers/2510.26583; weights and
    original code at https://huggingface.co/BAAI/Emu3.5-VisionTokenizer, Apache-2.0), an IBQ image tokenizer with
    a 131k codebook and 16x spatial downsampling.

    The port is inference-only: Apertus 1.5 generates text only, so it omits IBQ's differentiable training path,
    tokenizer losses, and decoder, and `encode` returns hard argmax indices that stop gradients (calling
    `train()` does not restore the omitted components). Run the tokenizer in `float32`: code assignment is an
    argmax over codebook logits, and half precision flips ~8% of codes (bf16); the weights are kept fp32
    on half-precision `from_pretrained` loads via `_keep_in_fp32_modules_strict`.
    """
)
class Apertus1p5VisionTokenizerModel(PreTrainedModel):
    config: Apertus1p5VisionTokenizerConfig
    base_model_prefix = "vision_tokenizer"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["Apertus1p5VisionTokenizerResnetBlock", "Apertus1p5VisionTokenizerAttnBlock"]
    # code assignment is an argmax over codebook logits: half precision flips ~8% of codes (bf16, 131k codebook),
    # so the tokenizer is kept in fp32 even when the model is loaded in fp16/bf16
    _keep_in_fp32_modules_strict = ["encoder", "quant_conv", "quantize"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.GroupNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight)

    def __init__(self, config: Apertus1p5VisionTokenizerConfig):
        super().__init__(config)
        self.config = config

        self.encoder = Apertus1p5VisionTokenizerEncoder(config)
        self.quantize = Apertus1p5VisionTokenizerVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, kernel_size=1)

        self.vision_spatial_factor = config.spatial_scale_factor
        # The pretrained tokenizer starts in evaluation mode. This does not freeze parameters or add `no_grad`.
        self.eval()

        self.post_init()

    def encode(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        """
        Tokenizes images into a grid of discrete codebook indices.

        The method respects the caller's gradient mode, as public Transformers model methods normally do; use
        `torch.no_grad()` or `torch.inference_mode()` when calling it for standalone inference.

        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, channels, height, width)`):
                Input images, normalized to `[-1, 1]` with sides that are multiples of the spatial factor (16).

        Returns:
            `torch.LongTensor` of shape `(batch_size, height // factor, width // factor)` with values in
            `[0, codebook_size)`, where `factor` is `config.spatial_scale_factor` (16 by default).
        """
        # the tokenizer runs in fp32 even inside a half-precision model (`_keep_in_fp32_modules_strict`):
        # cast the input to the encoder's dtype so callers can pass fp16/bf16 pixel values
        pixel_values = pixel_values.to(self.encoder.conv_in.weight.dtype)
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        return self.quantize(hidden_states)


@auto_docstring
class Apertus1p5PreTrainedModel(PreTrainedModel):
    config: Apertus1p5Config
    base_model_prefix = "model"
    input_modalities = ("image", "audio", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # per-image / per-clip tokenizer loops are data-dependent control flow
    _can_compile_fullgraph = False
    _supports_flex_attn = True
    _supports_attention_backend = True
    _no_split_modules = [
        "Apertus1p5TextDecoderLayer",
        "ApertusDecoderLayer",  # plain `apertus` text configs remain accepted
        "Apertus1p5VisionTokenizerResnetBlock",
        "Apertus1p5VisionTokenizerAttnBlock",
    ]

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        # the no-op getter call (no arguments) stays allowed
        if new_num_tokens is not None or pad_to_multiple_of is not None:
            _check_pruned_head_resize(self.config)
        return super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True):
        # the config forbids constructing a tied pruned head, but `tie_word_embeddings` can be flipped on the
        # live config afterwards, and the generic tying would then silently install the full-width input
        # embeddings as `lm_head.weight`
        _check_pruned_head_tie(self.config)
        return super().tie_weights(missing_keys, recompute_mapping)


class Apertus1p5TextPreTrainedModel(ApertusPreTrainedModel):
    config: Apertus1p5TextConfig

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        # the no-op getter call (no arguments) stays allowed
        if new_num_tokens is not None or pad_to_multiple_of is not None:
            _check_pruned_head_resize(self.config)
        return PreTrainedModel.resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of, mean_resizing)

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True):
        # Mirrors Apertus1p5PreTrainedModel: post-hoc `tie_word_embeddings` flips must not tie a pruned head.
        # The duplication is forced: sharing the guards via a mixin or dual inheritance is not converter-safe
        # (the modular converter reorders the generated bases, which either breaks the MRO or silently shadows
        # the mixin methods).
        _check_pruned_head_tie(self.config)
        return PreTrainedModel.tie_weights(self, missing_keys, recompute_mapping)


class Apertus1p5TextModel(ApertusModel):
    config: Apertus1p5TextConfig


class Apertus1p5TextForCausalLM(ApertusForCausalLM, Apertus1p5PreTrainedModel):
    config: Apertus1p5TextConfig
    # the second base only contributes the pruned-head resize/tie guards; override its composite metadata
    input_modalities = ("text",)
    _can_compile_fullgraph = True
    _no_split_modules = ["Apertus1p5TextDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        # the LM head may be pruned to the text-only prefix of the vocabulary; input embeddings keep the
        # full vocabulary, and retained output ids equal their token ids
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size or config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.output_vocab_size or config.vocab_size)` or -100; tokens with indices set to `-100` are
            ignored (masked). With a pruned head, positions holding multimodal ids (`>= output_vocab_size`)
            must be masked with `-100`, exactly like any other id outside the output range. Calls with `labels`
            return logits of the physical head width; calls without `labels` return logits padded to
            `config.vocab_size`, with `torch.finfo(dtype).min` scores for the input-only tail.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        projected_logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            _check_pruned_head_labels(labels, self.lm_head.out_features, self.config.vocab_size)
            # Compute loss on the physical text-only projection. The logical input-only tail would contribute zero
            # probability and needlessly increase loss memory.
            loss = self.loss_function(
                logits=projected_logits, labels=labels, vocab_size=self.lm_head.out_features, **kwargs
            )
            # Loss-only calls keep the compact physical width: padding would materialize a full-vocabulary copy
            # of the logits for every position on every training step.
            logits = projected_logits
        else:
            # Generic generation assumes that prompt ids and returned score indices share `config.vocab_size`.
            # Preserve the compact physical head while exposing that logical width; the multimodal tail remains
            # non-generatable.
            logits = _pad_logits_to_vocab_size(projected_logits, self.config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Apertus 1.5 model: an Apertus language backbone whose input embeddings are shared by text tokens and
    the discrete image/audio codes produced by the bundled vision tokenizer and audio codec.

    Media tensors are flattened over all prompts, independently for images and audio, in batch-sample order and
    then left-to-right placeholder order within each sample. The model encodes tensor rows in that order and
    scatters their concatenated features into the corresponding placeholder positions in row-major order. Images
    and audio may be interleaved arbitrarily because each modality uses its own placeholder mask. The model checks
    the total feature/placeholder count for each modality, but cannot infer per-sample ownership; callers that
    bypass `Apertus1p5Processor` must preserve this ordering themselves.
    """
)
class Apertus1p5Model(Apertus1p5PreTrainedModel):
    # both tokenizers assign codes by argmax, which half precision perturbs (see Apertus1p5VisionTokenizerModel)
    _keep_in_fp32_modules_strict = ["vision_tokenizer", "audio_tokenizer"]

    def __init__(self, config: Apertus1p5Config):
        super().__init__(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.vision_tokenizer = Apertus1p5VisionTokenizerModel(config.vision_tokenizer_config)
        self.audio_tokenizer = AutoModel.from_config(config.audio_tokenizer_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @staticmethod
    def _validate_image_inputs(pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor | None) -> None:
        if pixel_values.shape[0] == 0:
            raise ValueError("`pixel_values` contains no images; pass `pixel_values=None` when there are no images.")
        if image_sizes is None:
            raise ValueError("`image_sizes` must be provided when `pixel_values` are provided.")
        if pixel_values.shape[0] != image_sizes.shape[0]:
            raise ValueError(
                "The number of images in `pixel_values` must match the number of entries in `image_sizes`, but got "
                f"{pixel_values.shape[0]} images and {image_sizes.shape[0]} size entries."
            )

    @staticmethod
    def _validate_audio_inputs(input_features: torch.FloatTensor, feature_attention_mask: torch.Tensor | None) -> None:
        if input_features.shape[0] == 0:
            raise ValueError(
                "`input_features` contains no audio clips; pass `input_features=None` when there is no audio."
            )
        if feature_attention_mask is None:
            raise ValueError("`feature_attention_mask` must be provided when `input_features` are provided.")
        if input_features.shape[0] != feature_attention_mask.shape[0]:
            raise ValueError(
                "The number of audio clips in `input_features` must match the number of rows in `feature_attention_mask`, "
                f"but got {input_features.shape[0]} clips and {feature_attention_mask.shape[0]} mask rows."
            )

    def get_image_tokens(self, pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor) -> torch.LongTensor:
        """
        Tokenizes images into discrete codes and maps them into the shared text vocabulary via
        `config.image_token_offset`. Each image is cropped to its true size and encoded individually: the encoder
        contains global attention, so batch padding would perturb the codes.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(num_images, num_channels, height, width)`):
                The tensors corresponding to the input images, padded to a common size.
            image_sizes (`torch.LongTensor` of shape `(num_images, 2)`):
                The true size of each image, being (height, width).

        Returns:
            `torch.LongTensor`: flat vocabulary ids of all image codes (`sum_i (h_i // 16) * (w_i // 16)` items).
        """
        self._validate_image_inputs(pixel_values, image_sizes)
        vocab_ids_list = []
        for image, size in zip(pixel_values, image_sizes):
            image = image[None, :, : int(size[0]), : int(size[1])]
            codes = self.vision_tokenizer.encode(image)[0]
            vocab_ids_list.append(codes.flatten() + self.config.image_token_offset)
        return torch.cat(vocab_ids_list)

    @can_return_tuple
    @auto_docstring(
        custom_intro="Tokenizes images into discrete tokens with the vision tokenizer and embeds them with the text embeddings layer"
    )
    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor
    ) -> tuple | Apertus1p5VisionTokenizerModelOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_images, num_channels, height, width)`):
            The tensors corresponding to the input images, padded to a common size.
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`):
            The true size of each image, being (height, width).
        """
        vocab_ids = self.get_image_tokens(pixel_values, image_sizes)
        factor = self.vision_tokenizer.vision_spatial_factor
        grid_sizes = [(int(size[0]) // factor, int(size[1]) // factor) for size in image_sizes]
        split_sizes = [height * width for height, width in grid_sizes]
        image_embeddings = self.get_input_embeddings()(vocab_ids)
        image_features = torch.split(image_embeddings, split_sizes)
        image_tokens = [
            (ids - self.config.image_token_offset).reshape(grid)
            for ids, grid in zip(torch.split(vocab_ids, split_sizes), grid_sizes)
        ]
        return Apertus1p5VisionTokenizerModelOutput(image_tokens=image_tokens, pooler_output=image_features)

    def get_audio_tokens(
        self, input_features: torch.FloatTensor, feature_attention_mask: torch.Tensor
    ) -> torch.LongTensor:
        """
        Tokenizes audio clips into discrete codes and maps them into the shared text vocabulary via
        `config.audio_token_offset`. Each clip is sliced to its true length and encoded individually so the codes
        are independent of batch padding.

        Args:
            input_features (`torch.FloatTensor` of shape `(num_clips, 1, max_length)`):
                Mono 24 kHz waveforms, padded to a common length.
            feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`):
                Mask with 1 for valid samples and 0 for padding, as returned by the feature extractor.

        Returns:
            `torch.LongTensor`: flat vocabulary ids of all audio codes (`sum_i ceil(length_i / hop)` items).
        """
        self._validate_audio_inputs(input_features, feature_attention_mask)
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(1)
        audio_lengths = feature_attention_mask.sum(dim=-1)
        if (audio_lengths == 0).any():
            raise ValueError("`feature_attention_mask` marks an audio clip with zero valid samples.")
        # a 0 -> 1 transition along a row means the mask is not right-padded
        if (feature_attention_mask.long().diff(dim=-1) > 0).any():
            raise ValueError(
                "`feature_attention_mask` must be right-padded (valid samples first), as returned by the feature "
                "extractor."
            )
        vocab_ids_list = []
        for clip, length in zip(input_features, audio_lengths):
            clip = clip[None, :, : int(length)].to(self.audio_tokenizer.dtype)
            codes = self.audio_tokenizer.encode(clip).audio_codes
            vocab_ids_list.append(codes.flatten() + self.config.audio_token_offset)
        return torch.cat(vocab_ids_list)

    @can_return_tuple
    @auto_docstring(
        custom_intro="Tokenizes audio clips into discrete tokens with the audio codec and embeds them with the text embeddings layer"
    )
    def get_audio_features(
        self, input_features: torch.FloatTensor, feature_attention_mask: torch.Tensor
    ) -> tuple | Apertus1p5AudioTokenizerModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, 1, max_length)`):
            Mono 24 kHz waveforms, padded to a common length.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`):
            Mask with 1 for valid samples and 0 for padding, as returned by the feature extractor.
        """
        vocab_ids = self.get_audio_tokens(input_features, feature_attention_mask)
        hop_length = self.audio_tokenizer.config.hop_length
        split_sizes = [-(-int(length) // hop_length) for length in feature_attention_mask.sum(dim=-1)]
        audio_embeddings = self.get_input_embeddings()(vocab_ids)
        audio_features = torch.split(audio_embeddings, split_sizes)
        audio_tokens = [ids - self.config.audio_token_offset for ids in torch.split(vocab_ids, split_sizes)]
        return Apertus1p5AudioTokenizerModelOutput(audio_tokens=audio_tokens, pooler_output=audio_features)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        audio_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder masks from `input_ids` or `inputs_embeds`, and checks that each
        placeholder token count equals the length of the corresponding multimodal features. If the lengths are
        different, an error is raised.
        """
        if input_ids is None:
            embedder = self.get_input_embeddings()
            image_token_embed = embedder(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token_embed).all(-1)
            audio_token_embed = embedder(
                torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_audio_mask = (inputs_embeds == audio_token_embed).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        if image_features is not None:
            n_image_tokens = special_image_mask.sum()
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                f"features: {image_features.shape[0]}",
            )
        if audio_features is not None:
            n_audio_tokens = special_audio_mask.sum()
            torch_compilable_check(
                n_audio_tokens * inputs_embeds.shape[-1] == audio_features.numel(),
                f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, "
                f"features: {audio_features.shape[0]}",
            )

        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        special_audio_mask = special_audio_mask.unsqueeze(-1).to(inputs_embeds.device)
        return special_image_mask, special_audio_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_images, num_channels, max_height, max_width)`, *optional*):
            Images from all prompts, padded to a common size. Rows must follow image-placeholder order across the
            text batch: batch-sample order, then left-to-right within each sample.
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            The true `(height, width)` of each image in `pixel_values`, in the same row order, used to crop away
            batch padding.
        input_features (`torch.FloatTensor` of shape `(num_clips, 1, max_length)`, *optional*):
            Mono 24 kHz audio waveforms from all prompts, padded to a common length. Rows must independently follow
            audio-placeholder order across the text batch: batch-sample order, then left-to-right within each
            sample.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`, *optional*):
            Mask with 1 for valid samples and 0 for padding, in the same row order as `input_features`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_sizes).pooler_output
            image_features = torch.cat(image_features, dim=0)
        audio_features = None
        if input_features is not None:
            audio_features = self.get_audio_features(input_features, feature_attention_mask).pooler_output
            audio_features = torch.cat(audio_features, dim=0)

        if image_features is not None or audio_features is not None:
            special_image_mask, special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features, audio_features=audio_features
            )
            if image_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            if audio_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return outputs


@auto_docstring(
    custom_intro="""
    The Apertus 1.5 model with a language-modeling head, for text generation conditioned on interleaved
    image, audio and text inputs.
    """
)
class Apertus1p5ForConditionalGeneration(Apertus1p5PreTrainedModel, GenerationMixin):
    output_modalities = ("text",)
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # both tokenizers assign codes by argmax, which half precision perturbs (see Apertus1p5VisionTokenizerModel)
    _keep_in_fp32_modules_strict = ["vision_tokenizer", "audio_tokenizer"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Apertus1p5Model(config)
        # the LM head may be pruned to the text-only prefix of the vocabulary (see Apertus1p5TextConfig)
        output_vocab_size = getattr(config.text_config, "output_vocab_size", None)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, output_vocab_size or config.text_config.vocab_size, bias=False
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            The true size of each image, being (height, width), used to crop away batch padding.
        input_features (`torch.FloatTensor` of shape `(num_clips, 1, max_length)`, *optional*):
            Mono 24 kHz audio waveforms, padded to a common length.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, max_length)`, *optional*):
            Mask with 1 for valid samples and 0 for padding, as returned by the feature extractor.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.output_vocab_size or text_config.vocab_size)` or -100; tokens with indices set to
            `-100` are ignored (masked). With a pruned head, positions holding multimodal ids
            (`>= output_vocab_size`) must be masked with `-100`, exactly like any other id outside the
            output range. Calls with `labels` return logits of the physical head width; calls without `labels`
            return logits padded to `text_config.vocab_size`, with `torch.finfo(dtype).min` scores for the
            input-only tail.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        projected_logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            _check_pruned_head_labels(labels, self.lm_head.out_features, self.config.text_config.vocab_size)
            # Compute loss on the physical text-only projection. The logical input-only tail would contribute zero
            # probability and needlessly increase loss memory.
            loss = self.loss_function(
                logits=projected_logits, labels=labels, vocab_size=self.lm_head.out_features, **kwargs
            )
            # Loss-only calls keep the compact physical width: padding would materialize a full-vocabulary copy
            # of the logits for every position on every training step.
            logits = projected_logits
        else:
            # Generic generation assumes that prompt ids and returned score indices share `text_config.vocab_size`.
            # Preserve the compact physical head while exposing that logical width; the multimodal tail stays
            # non-generatable.
            logits = _pad_logits_to_vocab_size(projected_logits, self.config.text_config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        input_features=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- multimodal inputs should only be forwarded on the first generation step

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            input_features=input_features,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["input_features"] = None

        return model_inputs

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        """
        Overwritten -- the media tensors (`pixel_values`/`image_sizes`, `input_features`/`feature_attention_mask`)
        are flattened over all media items in the batch, so the default row-wise `repeat_interleave` would separate
        the items belonging to one prompt. Placeholder counts per sample are instead matched against per-item token
        counts to split the flattened tensors into per-prompt groups, and each complete group is repeated: with two
        beams and a prompt containing `[img0, img1]`, row-wise expansion would produce `[img0, img0, img1, img1]`,
        grouped expansion produces `[img0, img1, img0, img1]`, so every beam receives the complete media group of
        its source prompt in original order. All batch-shaped tensors keep the ordinary `repeat_interleave`.
        """
        if expand_size == 1:
            return input_ids, model_kwargs

        def _get_placeholder_counts(token_id: int) -> list[int]:
            inputs_embeds = model_kwargs.get("inputs_embeds")
            if inputs_embeds is not None:
                token_embed = self.get_input_embeddings()(
                    torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                placeholder_mask = (inputs_embeds == token_embed).all(dim=-1)
            elif input_ids is not None:
                placeholder_mask = input_ids == token_id
            else:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided for multimodal expansion.")
            return placeholder_mask.sum(dim=-1).tolist()

        def _get_media_nums(placeholder_counts: list[int], media_token_counts: list[int], modality: str) -> list[int]:
            media_nums = []
            media_idx = 0
            for sample_idx, expected_tokens in enumerate(placeholder_counts):
                sample_media_idx = media_idx
                actual_tokens = 0
                while actual_tokens < expected_tokens and media_idx < len(media_token_counts):
                    actual_tokens += media_token_counts[media_idx]
                    media_idx += 1
                if actual_tokens != expected_tokens:
                    raise ValueError(
                        f"Cannot assign flattened {modality} inputs to sample {sample_idx}: found "
                        f"{expected_tokens} placeholder tokens, but the corresponding media inputs produce "
                        f"{actual_tokens} tokens."
                    )
                media_nums.append(media_idx - sample_media_idx)

            if media_idx != len(media_token_counts):
                raise ValueError(
                    f"Cannot assign all flattened {modality} inputs to the batch: {len(media_token_counts) - media_idx} "
                    f"media inputs remain after matching the per-sample placeholder counts."
                )
            return media_nums

        image_nums = None
        pixel_values = model_kwargs.get("pixel_values")
        image_sizes = model_kwargs.get("image_sizes")
        if pixel_values is not None:
            self.model._validate_image_inputs(pixel_values, image_sizes)
            factor = self.model.vision_tokenizer.vision_spatial_factor
            image_token_counts = (image_sizes // factor).prod(dim=-1).tolist()
            image_nums = _get_media_nums(
                _get_placeholder_counts(self.config.image_token_id), image_token_counts, "image"
            )

        audio_nums = None
        input_features = model_kwargs.get("input_features")
        feature_attention_mask = model_kwargs.get("feature_attention_mask")
        if input_features is not None:
            self.model._validate_audio_inputs(input_features, feature_attention_mask)
            hop_length = self.model.audio_tokenizer.config.hop_length
            audio_lengths = feature_attention_mask.sum(dim=-1)
            audio_token_counts = ((audio_lengths + hop_length - 1) // hop_length).tolist()
            audio_nums = _get_media_nums(
                _get_placeholder_counts(self.config.audio_token_id), audio_token_counts, "audio"
            )

        def _repeat_interleave_samples(tensor: torch.Tensor, lengths: list[int]) -> torch.Tensor:
            samples = torch.split(tensor, lengths)
            repeat_args = [expand_size] + [1] * (tensor.dim() - 1)
            return torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)

        image_keys = {"pixel_values", "image_sizes"}
        audio_keys = {"input_features", "feature_attention_mask"}

        def _expand_dict_for_generation(dict_to_expand):
            for key, value in dict_to_expand.items():
                if value is None or not isinstance(value, torch.Tensor):
                    continue
                if key in image_keys and image_nums is not None:
                    dict_to_expand[key] = _repeat_interleave_samples(value, image_nums)
                elif key in audio_keys and audio_nums is not None:
                    dict_to_expand[key] = _repeat_interleave_samples(value, audio_nums)
                else:
                    dict_to_expand[key] = value.repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "Apertus1p5Config",
    "Apertus1p5TextConfig",
    "Apertus1p5VisionTokenizerConfig",
    "Apertus1p5ForConditionalGeneration",
    "Apertus1p5PreTrainedModel",
    "Apertus1p5TextForCausalLM",
    "Apertus1p5TextModel",
    "Apertus1p5TextPreTrainedModel",
    "Apertus1p5VisionTokenizerModel",
    "Apertus1p5Model",
]
