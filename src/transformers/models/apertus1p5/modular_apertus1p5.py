# Copyright 2026 The Emu team, BAAI, The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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

The bundled vision tokenizer (`Apertus1p5VQVAE`) is an encode-only port of the EMU3.5 Vision Tokenizer by BAAI
(*Emu3.5: Native Multimodal Models are World Learners*, https://huggingface.co/papers/2510.26583; weights and
original code at https://huggingface.co/BAAI/Emu3.5-VisionTokenizer, Apache-2.0). It uses IBQ
(*Scalable Image Tokenization with Index Backpropagation Quantization*, https://huggingface.co/papers/2412.02692):
at inference the quantizer reduces to a dot-product similarity argmax over a 131k codebook, at a 16x spatial
downsample. This port does not support training the vision tokenizer: it omits the differentiable IBQ training
path, tokenizer losses, and decoder, and returns only hard indices whose argmax stops gradients. The public
`encode` method deliberately respects the caller's gradient mode instead of forcing `torch.no_grad`, following
the convention for public Transformers model methods; this API choice does not make the tokenizer trainable.
"""

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..emu3.configuration_emu3 import Emu3Config, Emu3TextConfig
from ..emu3.image_processing_emu3 import Emu3ImageProcessor, Emu3ImageProcessorKwargs
from ..emu3.modeling_emu3 import (
    Emu3Attention,
    Emu3DecoderLayer,
    Emu3ForCausalLM,
    Emu3ForConditionalGeneration,
    Emu3ImageVocabularyMapping,
    Emu3MLP,
    Emu3Model,
    Emu3PreTrainedModel,
    Emu3RMSNorm,
    Emu3RotaryEmbedding,
    Emu3TextModel,
    Emu3VQVAEModelOutput,
)
from ..emu3.processing_emu3 import Emu3Processor, Emu3ProcessorKwargs, Emu3TextKwargs


@auto_docstring(checkpoint="swiss-ai/Apertus-1.5-8B")
@strict
class Apertus1p5VQVAEConfig(PreTrainedConfig):
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
    channel_multiplier (`list[int]`, *optional*, defaults to `[1, 1, 2, 2, 4]`):
        Channel scaling factor per encoder stage. The number of stages is `len(channel_multiplier)`, giving a
        spatial downsampling factor of `2**(len(channel_multiplier) - 1)` (16 by default).
    num_res_blocks (`int`, *optional*, defaults to 4):
        Number of residual blocks per encoder stage.
    attn_resolutions (`list[int]`, *optional*, defaults to `[16]`):
        Feature-map resolutions (relative to a `resolution`-sized input) at which attention blocks are inserted.
    resolution (`int`, *optional*, defaults to 256):
        Reference input resolution used (statically, together with `attn_resolutions`) to decide which encoder
        stages carry attention blocks. Does not restrict the actual input size.
    dropout (`float`, *optional*, defaults to 0.0):
        Dropout inside the residual blocks.
    """

    model_type = "apertus1p5_vqgan"
    base_config_key = "vq_config"

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


class Apertus1p5TextConfig(Emu3TextConfig):
    pass


class Apertus1p5Config(Emu3Config):
    pass


class Apertus1p5VQVAEModelOutput(Emu3VQVAEModelOutput):
    pass


class Apertus1p5Attention(Emu3Attention):
    pass


class Apertus1p5RMSNorm(Emu3RMSNorm):
    pass


class Apertus1p5MLP(Emu3MLP):
    pass


class Apertus1p5DecoderLayer(Emu3DecoderLayer):
    pass


class Apertus1p5VQVAEResnetBlock(nn.Module):
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


class Apertus1p5VQVAEAttnBlock(nn.Module):
    """Single-head self-attention over spatial positions with 1x1 convolutions."""

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


class Apertus1p5VQVAEDownsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # asymmetric right/bottom padding, as in the original VQGAN-style tokenizer
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Apertus1p5VQVAEEncoder(nn.Module):
    def __init__(self, config: Apertus1p5VQVAEConfig):
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
                block.append(Apertus1p5VQVAEResnetBlock(block_in, block_out, dropout=config.dropout))
                block_in = block_out
                if current_resolution in config.attn_resolutions:
                    attn.append(Apertus1p5VQVAEAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Apertus1p5VQVAEDownsample(block_in)
                current_resolution = current_resolution // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = Apertus1p5VQVAEResnetBlock(block_in, block_in, dropout=config.dropout)
        self.mid.attn_1 = Apertus1p5VQVAEAttnBlock(block_in)
        self.mid.block_2 = Apertus1p5VQVAEResnetBlock(block_in, block_in, dropout=config.dropout)

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


class Apertus1p5VQVAEVectorQuantizer(nn.Module):
    """
    IBQ codebook lookup, from *Scalable Image Tokenization with Index Backpropagation Quantization*
    (https://huggingface.co/papers/2412.02692). IBQ differs from plain VQ in how the codebook is trained; at
    inference, quantization reduces to a dot-product similarity argmax over the codebook. This class implements
    only that inference path, not IBQ's differentiable index-backpropagation training path.
    """

    def __init__(self, config: Apertus1p5VQVAEConfig):
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
    a 131k codebook and 16x spatial downsampling. Apertus 1.5 generates text only, so the original decoder is not
    ported. Run the tokenizer in `float32`: code assignment is an argmax over codebook logits, and half precision
    perturbs a few percent of codes.

    This is an inference-only tokenizer port and does not support tokenizer training. It omits IBQ's
    differentiable index-backpropagation path, training losses, and decoder, while `encode` returns hard indices
    whose argmax stops gradients. Calling `train()` does not restore those components. The `encode` method is not
    decorated with `torch.no_grad`: public Transformers model methods conventionally respect the caller's
    gradient mode. Callers may use `torch.no_grad()` or `torch.inference_mode()` for standalone inference; the
    absence of a decorator does not imply that this tokenizer supports training.
    """
)
class Apertus1p5VQVAE(PreTrainedModel):
    config: Apertus1p5VQVAEConfig
    base_model_prefix = "visionvq"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["Apertus1p5VQVAEResnetBlock", "Apertus1p5VQVAEAttnBlock"]
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

    def __init__(self, config: Apertus1p5VQVAEConfig):
        super().__init__(config)
        self.config = config

        self.encoder = Apertus1p5VQVAEEncoder(config)
        self.quantize = Apertus1p5VQVAEVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, kernel_size=1)

        self.vision_spatial_factor = config.spatial_scale_factor
        # The pretrained tokenizer starts in evaluation mode. This does not freeze parameters or add `no_grad`.
        self.eval()

        self.post_init()

    def encode(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        """
        Tokenizes images into a grid of discrete codebook indices.

        This method implements inference tokenization only. Its hard argmax output is not differentiable, and
        this port does not include the IBQ training path or losses needed to train the tokenizer. It deliberately
        respects the caller's gradient mode, as public Transformers model methods normally do; use
        `torch.no_grad()` or `torch.inference_mode()` when calling it for standalone inference.

        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, channels, height, width)`):
                Input images, normalized to `[-1, 1]` with sides that are multiples of the spatial factor (16).

        Returns:
            `torch.LongTensor` of shape `(batch_size, height // 16, width // 16)` with values in
            `[0, codebook_size)`.
        """
        # the tokenizer runs in fp32 even inside a half-precision model (`_keep_in_fp32_modules_strict`):
        # cast the input to the encoder's dtype so callers can pass fp16/bf16 pixel values
        pixel_values = pixel_values.to(self.encoder.conv_in.weight.dtype)
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        return self.quantize(hidden_states)


class Apertus1p5ImageVocabularyMapping(Emu3ImageVocabularyMapping):
    pass


class Apertus1p5PreTrainedModel(Emu3PreTrainedModel):
    pass


class Apertus1p5RotaryEmbedding(Emu3RotaryEmbedding):
    pass


class Apertus1p5TextModel(Emu3TextModel):
    pass


class Apertus1p5ForCausalLM(Emu3ForCausalLM):
    pass


class Apertus1p5Model(Emu3Model):
    # keep the vision tokenizer in fp32 when the model is loaded in fp16/bf16: its code assignment is an argmax
    # over codebook logits and half precision flips a significant fraction of codes
    _keep_in_fp32_modules_strict = ["vqmodel"]

    @staticmethod
    def _validate_image_inputs(pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor | None) -> None:
        if image_sizes is None:
            raise ValueError("`image_sizes` must be provided when `pixel_values` are provided.")
        if pixel_values.shape[0] != image_sizes.shape[0]:
            raise ValueError(
                "The number of images in `pixel_values` must match the number of entries in `image_sizes`, but got "
                f"{pixel_values.shape[0]} images and {image_sizes.shape[0]} size entries."
            )

    def get_image_tokens(self, pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor) -> torch.LongTensor:
        """
        Tokenizes images into discrete tokens with the vision tokenizer and converts them to BPE tokens.
        Each image is cropped to its true size and encoded individually: the encoder contains global attention,
        so batch padding would perturb the codes.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`):
                The sizes of the images in the batch, being (height, width) for each image.
        """
        self._validate_image_inputs(pixel_values, image_sizes)
        image_tokens_list = []
        for image, size in zip(pixel_values, image_sizes):
            image = image[None, :, : int(size[0]), : int(size[1])]
            image_tokens_list.append(self.vqmodel.encode(image)[0])
        bpe_tokens_list = [self.vocabulary_mapping.convert_img2bpe(tokens).flatten() for tokens in image_tokens_list]
        return torch.cat(bpe_tokens_list)

    @can_return_tuple
    @auto_docstring(
        custom_intro="Tokenizes images into discrete tokens with the vision tokenizer and embeds them with the text embeddings layer"
    )
    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | Apertus1p5VQVAEModelOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images.
        """
        self._validate_image_inputs(pixel_values, image_sizes)
        image_tokens_list = []
        for image, size in zip(pixel_values, image_sizes):
            image = image[None, :, : int(size[0]), : int(size[1])]
            image_tokens_list.append(self.vqmodel.encode(image)[0])
        split_sizes = [
            (height // self.vqmodel.vision_spatial_factor) * (width // self.vqmodel.vision_spatial_factor + 1)
            for height, width in image_sizes
        ]
        bpe_tokens_list = [self.vocabulary_mapping.convert_img2bpe(tokens).flatten() for tokens in image_tokens_list]
        bpe_tokens = torch.cat(bpe_tokens_list)
        image_embeddings = self.get_input_embeddings()(bpe_tokens)
        image_features = torch.split(image_embeddings, split_sizes)
        return Apertus1p5VQVAEModelOutput(image_tokens=image_tokens_list, pooler_output=image_features)

    def decode_image_tokens(self, image_tokens: torch.LongTensor, height: int, width: int):
        raise AttributeError(
            "Apertus 1.5 generates text only; its vision tokenizer is encode-only and has no decoder."
        )


class Apertus1p5ForConditionalGeneration(Emu3ForConditionalGeneration):
    output_modalities = ("text",)
    # keep the vision tokenizer in fp32 when the model is loaded in fp16/bf16 (argmax code assignment)
    _keep_in_fp32_modules_strict = ["vqmodel"]

    def decode_image_tokens(self, **kwargs):
        raise AttributeError(
            "Apertus 1.5 generates text only; its vision tokenizer is encode-only and has no decoder."
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.Tensor | None = None,
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
        image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`):
            The sizes of the images in the batch, being (height, width) for each image.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
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
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Apertus1p5ImageProcessorKwargs(Emu3ImageProcessorKwargs):
    pass


class Apertus1p5ImageProcessor(Emu3ImageProcessor):
    pass


class Apertus1p5TextKwargs(Emu3TextKwargs):
    pass


class Apertus1p5ProcessorKwargs(Emu3ProcessorKwargs):
    pass


class Apertus1p5Processor(Emu3Processor):
    pass


__all__ = [
    "Apertus1p5Config",
    "Apertus1p5TextConfig",
    "Apertus1p5VQVAEConfig",
    "Apertus1p5ForConditionalGeneration",
    "Apertus1p5ForCausalLM",
    "Apertus1p5TextModel",
    "Apertus1p5PreTrainedModel",
    "Apertus1p5VQVAE",
    "Apertus1p5Model",
    "Apertus1p5ImageProcessor",
    "Apertus1p5Processor",
]
