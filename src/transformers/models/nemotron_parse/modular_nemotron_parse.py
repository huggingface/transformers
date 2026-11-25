# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ImageInput, PILImageResampling, SizeDict
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPooling
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig, AutoModel
from ..donut.image_processing_donut import DonutImageProcessor, DonutImageProcessorKwargs
from ..mbart.configuration_mbart import MBartConfig
from ..pp_formulanet.modeling_pp_formulanet import (
    PPFormulaNetForConditionalGeneration,
    PPFormulaNetModel,
    PPFormulaNetPreTrainedModel,
    PPFormulaNetTextModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-Parse-2.0")
@strict
class NemotronParseTextConfig(MBartConfig):
    r"""
    Example:

    ```python
    >>> from transformers import NemotronParseTextConfig, NemotronParseTextModel

    >>> configuration = NemotronParseTextConfig()
    >>> model = NemotronParseTextModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "nemotron_parse_text"
    base_config_key = "text_config"
    attribute_map = {
        "num_attention_heads": "decoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "decoder_layers",
    }
    vocab_size: int = 72256
    decoder_layers: int = 10
    scale_embedding: bool = True
    decoder_start_token_id: int | None = 2
    max_position_embeddings = AttributeError()
    encoder_layers = AttributeError()
    encoder_ffn_dim = AttributeError()
    encoder_attention_heads = AttributeError()
    encoder_layerdrop = AttributeError()
    classifier_dropout = AttributeError()
    is_decoder = AttributeError()


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-Parse-2.0")
@strict
class NemotronParseConfig(PreTrainedConfig):
    model_type = "nemotron_parse"
    sub_configs_defaults = {
        "text_config": NemotronParseTextConfig,
        # C-RADIOv2-H: one cls token per teacher, padded with registers to 8 prefix tokens
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="radio",
            init_kwargs={"num_cls_tokens": 4, "num_registers": 4, "summary_idxs": [0, 1, 2]},
        ),
    }

    text_config: dict | NemotronParseTextConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # The released checkpoints store the sub-configs as `encoder`/`decoder`. Their encoder has an empty
        # `model_type` and forces eager attention, which would materialize 16 x 13k x 13k attention weights per
        # layer for a page, and their decoder carries a stale `num_hidden_layers` next to the real `decoder_layers`.
        if (encoder := kwargs.pop("encoder", None)) is not None:
            encoder = {k: v for k, v in encoder.items() if k not in ("model_type", "attn_implementation", "auto_map")}
            self.vision_config = {**self.sub_configs_defaults["vision_config"].init_kwargs, **encoder}
        if (decoder := kwargs.pop("decoder", None)) is not None:
            self.text_config = {k: v for k, v in decoder.items() if k != "num_hidden_layers"}
        super().__post_init__(**kwargs)


class NemotronParseImageProcessorKwargs(DonutImageProcessorKwargs):
    pass


@auto_docstring
class NemotronParseImageProcessor(DonutImageProcessor):
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 2048, "width": 1664}
    do_resize = False
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[NemotronParseImageProcessorKwargs]):
        # the released `preprocessor_config.json` files also store the size as a redundant `longest_edge`
        if isinstance(size := kwargs.get("size"), dict):
            kwargs["size"] = {key: value for key, value in size.items() if key != "longest_edge"}
        super().__init__(**kwargs)

    def thumbnail(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Shrinks the image to fit into `size`, clamping the height first and then the width."""
        height, width = image.shape[-2:]
        aspect_ratio = width / height
        new_height, new_width = height, width
        if new_height > size.height:
            new_height, new_width = size.height, int(size.height * aspect_ratio)
        if new_width > size.width:
            new_height, new_width = int(size.width / aspect_ratio), size.width
        if (new_height, new_width) == (height, width):
            return image
        return self.resize(image, SizeDict(height=new_height, width=new_width), resample, **kwargs)

    def pad_image(self, image: "torch.Tensor", size: SizeDict, random_padding: bool = False) -> "torch.Tensor":
        """Centers the image on a white canvas of `size`."""
        delta_height, delta_width = size.height - image.shape[-2], size.width - image.shape[-1]
        pad_top, pad_left = delta_height // 2, delta_width // 2
        return tvF.pad(image, [pad_left, pad_top, delta_width - pad_left, delta_height - pad_top], fill=255)


@auto_docstring
class NemotronParseProcessor(ProcessorMixin):
    text_kwargs = {"add_special_tokens": False}

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        r"""
        text (`str`, `list[str]`, *optional*):
            The task prompt(s), e.g. `"</s><s><predict_bbox><predict_classes><output_markdown>"`. They are tokenized
            into `decoder_input_ids`.
        """
        model_inputs = super().__call__(images=images, text=text, **kwargs)
        if text is not None:
            model_inputs["decoder_input_ids"] = model_inputs.pop("input_ids")
            model_inputs["decoder_attention_mask"] = model_inputs.pop("attention_mask")
        return model_inputs

    @property
    def model_input_names(self):
        return self.image_processor.model_input_names + ["decoder_input_ids", "decoder_attention_mask"]


@auto_docstring
class NemotronParsePreTrainedModel(PPFormulaNetPreTrainedModel):
    # splitting the vision encoder would leave the direct `summary_idxs` buffer of `RadioModel` without a device
    _no_split_modules = ["NemotronParseVisionEncoder", "NemotronParseDecoderLayer"]
    _keep_in_fp32_modules_strict = AttributeError()
    _supports_sdpa = True
    input_modalities = ("image", "text")
    # the image processor normalizes the pixel values, so the RADIO input conditioner is not used
    _keys_to_ignore_on_load_unexpected = [r"input_conditioner\."]

    def _init_weights(self, module):
        raise AttributeError("Not needed for NemotronParse")


@auto_docstring
class NemotronParseVisionEncoder(NemotronParsePreTrainedModel):
    def __init__(self, config: NemotronParseConfig):
        super().__init__(config)
        vision_config = config.vision_config
        hidden_size = config.text_config.d_model
        self.vision_model = AutoModel.from_config(vision_config)
        self.vision_model.make_preprocessor_external()
        self.conv1 = nn.Conv1d(vision_config.hidden_size, hidden_size, kernel_size=1)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 4), stride=(1, 4), bias=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.sum_proj = nn.Linear(len(vision_config.summary_idxs) * vision_config.hidden_size, hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutputWithPooling:
        vision_outputs = self.vision_model(pixel_values.to(self.dtype), **kwargs)
        patch_size = self.config.vision_config.patch_size
        grid = (pixel_values.shape[-2] // patch_size, pixel_values.shape[-1] // patch_size)

        features = self.layer_norm1(self.conv1(vision_outputs.features.transpose(1, 2)).transpose(1, 2))
        features = self.conv2(features.transpose(1, 2).unflatten(2, grid))
        features = self.layer_norm2(features.flatten(2).transpose(1, 2))
        summary = self.layer_norm3(self.sum_proj(vision_outputs.summary))

        return BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=torch.cat([features, summary.unsqueeze(1)], dim=1),
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )


@auto_docstring
class NemotronParseTextModel(PPFormulaNetTextModel):
    """MBart decoder without position embeddings."""

    def __init__(self, config: NemotronParseTextConfig):
        super().__init__(config)
        del self.max_target_positions
        del self.embed_positions

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions:
        r"""
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the vision encoder, used in the cross-attention of the decoder.
        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding tokens of the encoder outputs.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # initialize `cache`
        if use_cache and cache is None:
            cache = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        batch_size, seq_length = inputs_embeds.size()[:-1]
        cache_length = cache.get_seq_length() if cache is not None else 0

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = cache_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        self_attn_cache = cache.self_attention_cache if isinstance(cache, EncoderDecoderCache) else cache

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache=self_attn_cache,
        )
        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = self.layernorm_embedding(inputs_embeds)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if self.training and self.layerdrop > 0.0:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            hidden_states = decoder_layer(
                hidden_states,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                cache=cache,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            cache=cache,
        )


@auto_docstring
class NemotronParseModel(PPFormulaNetModel):
    def __init__(self, config: NemotronParseConfig):
        super().__init__(config)
        self.encoder = NemotronParseVisionEncoder(config)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value


@auto_docstring
class NemotronParseForConditionalGeneration(PPFormulaNetForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}
    # the decoder has no TP plan, so its embedding is never sharded and neither may the tied head be
    _tp_plan = AttributeError()

    def forward(**super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, NemotronParseForConditionalGeneration
        >>> from transformers.image_utils import load_image

        >>> model = NemotronParseForConditionalGeneration.from_pretrained("nvidia/NVIDIA-Nemotron-Parse-2.0", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("nvidia/NVIDIA-Nemotron-Parse-2.0")

        >>> image = load_image("https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/paddle_doc_test.jpg")
        >>> prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        >>> generated_ids = model.generate(**inputs, max_new_tokens=32)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        '<x_0.3018><y_0.2719># SCANNED DOCUMENT<x_0.6426><y_0.2953><class_Title>\n\n<x_0.3008><y_0.318>Bill To: Example Customer<x_0.5146><y_0.3328><class_Text>\n\n<x_0.3018><y_0.3406>In'
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "NemotronParseConfig",
    "NemotronParseTextConfig",
    "NemotronParseImageProcessor",
    "NemotronParseProcessor",
    "NemotronParsePreTrainedModel",
    "NemotronParseTextModel",
    "NemotronParseModel",
    "NemotronParseForConditionalGeneration",
]
