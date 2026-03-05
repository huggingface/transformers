# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GraniteDoclingHybrid model."""

from itertools import accumulate
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING
from ..got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from ..granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from ..granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache
from ..idefics3.configuration_idefics3 import Idefics3VisionConfig
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3CausalLMOutputWithPast,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
)
from ..idefics3.processing_idefics3 import (
    Idefics3Processor,
    get_image_prompt_string,
    is_image_or_image_url,
    is_url,
    load_image,
)
from .configuration_granite_docling_hybrid import GraniteDoclingHybridConfig


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput


logger = logging.get_logger(__name__)


## Configuration ##


class GraniteDoclingHybridVisionConfig(Idefics3VisionConfig):
    pass


class GraniteDoclingHybridGraniteMoeHybridConfig(GraniteMoeHybridConfig):
    pass


class GraniteDoclingHybridConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GraniteDoclingHybridModel`]. It is used to instantiate a
    GraniteDoclingHybrid model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Idefics3 model architecture,
    but with a GraniteMoeHybrid text model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`GraniteDoclingHybridVisionConfig` or `dict`, *optional*, defaults to `GraniteDoclingHybridVisionConfig`):
            Custom vision config or dict for the vision tower
        text_config (`PretrainedConfig` or `dict`, *optional*, defaults to `GraniteMoeHybridConfig`):
            Custom text config or dict for the text model
        scale_factor (`int`, *optional*, defaults to 2):
            The scale factor for the image encoder.

    Example:
    ```python
    >>> from transformers import GraniteDoclingHybridModel, GraniteDoclingHybridConfig
    >>> # Initializing configuration
    >>> configuration = GraniteDoclingHybridConfig()
    >>> # Initializing a model from the configuration
    >>> model = GraniteDoclingHybridModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_docling_hybrid"
    sub_configs = {"text_config": CONFIG_MAPPING, "vision_config": GraniteDoclingHybridVisionConfig}

    def __init__(
        self,
        image_token_id=128257,
        tie_word_embeddings=False,
        vision_config=None,
        text_config=None,
        scale_factor=2,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings

        if vision_config is None:
            self.vision_config = GraniteDoclingHybridVisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = GraniteDoclingHybridVisionConfig(**vision_config)
        elif isinstance(vision_config, GraniteDoclingHybridVisionConfig):
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "granitemoehybrid")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, using default GraniteMoeHybrid text config")
            text_config = CONFIG_MAPPING["granitemoehybrid"]()

        self.text_config = text_config
        self.scale_factor = scale_factor

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


## Processing ##


class GraniteDoclingHybridProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class GraniteDoclingHybridProcessor(Idefics3Processor):

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        audio=None,
        videos=None,
        image_seq_len: int | None = None,
        **kwargs: Unpack[GraniteDoclingHybridProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        This method extends the Idefics3Processor to handle GotOcr2ImageProcessor specifics.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `list[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<row_x_col_y>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            image_seq_len (`int`, *optional*):
                The length of the image sequence. If not provided, the default value of self.image_seq_len is used.
                image_seq_len should be equal to int(((image_size // patch_size) ** 2) / (scale_factor**2))
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            GraniteDoclingHybridProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        n_images_in_text = []
        n_images_in_images = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {self.image_token} tokens and {len(images)} images."
                        )
                    # Reorganize the images to match the prompts
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                else:
                    images = [images]
            elif (
                not isinstance(images, (list, tuple))
                and not isinstance(images[0], (list, tuple))
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )
            n_images_in_images = [len(sample) for sample in images]

            # Load images if they are URLs
            images = [[load_image(im) if is_url(im) else im for im in sample] for sample in images]

            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

                # GotOcr2ImageProcessor doesn't return rows/cols, compute them
                image_rows = []
                image_cols = []
                for sample_images in images:
                    sample_image_rows = []
                    sample_image_cols = []
                    for img in sample_images:
                        width, height = img.size
                        n_cols, n_rows = get_optimal_tiled_canvas(
                            (height, width),
                            (
                                self.image_processor.size["height"],
                                self.image_processor.size["width"],
                            ),
                            self.image_processor.min_patches,
                            self.image_processor.max_patches,
                        )
                        sample_image_rows.append(n_rows)
                        sample_image_cols.append(n_cols)
                    image_rows.append(sample_image_rows)
                    image_cols.append(sample_image_cols)

                # Post-process inputs for GotOcr2ImageProcessor
                inputs.pop("num_patches", None)  # Not needed downstream
                pixel_values = inputs.get("pixel_values")
                if pixel_values is not None and len(pixel_values.shape) == 4:
                    # Make 5D to match Idefics3 expected format: (batch, num_images, num_channels, height, width)
                    inputs["pixel_values"] = pixel_values.unsqueeze(0)

                fake_image_token = self.fake_image_token
                image_token = self.image_token
                global_img_token = self.global_image_tag

                prompt_strings = []
                batch_image_seq_lengths = []
                for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                    # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
                    image_prompt_strings = []
                    image_seq_lengths = []
                    for n_rows, n_cols in zip(sample_rows, sample_cols):
                        image_prompt_string = get_image_prompt_string(
                            n_rows,
                            n_cols,
                            image_seq_len,
                            image_token=image_token,
                            fake_token_around_image=fake_image_token,
                            global_img_token=global_img_token,
                        )
                        # Add +2 and +3 for special BOI/EOI/fake_image_wrapper tokens
                        row_length = (self.image_seq_len + 2) * n_cols + 1
                        image_seq_lengths.append((self.image_seq_len + 3) + row_length * n_rows)
                        image_prompt_strings.append(image_prompt_string)

                    batch_image_seq_lengths.append(image_seq_lengths)
                    split_sample = sample.split(image_token)
                    if len(split_sample) == 0:
                        raise ValueError("The image token should be present in the text.")

                    # Place in the image prompt strings where the image tokens are
                    sample = split_sample[0]
                    for i, image_prompt_string in enumerate(image_prompt_strings):
                        sample += image_prompt_string + split_sample[i + 1]
                    prompt_strings.append(sample)

                text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
                self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])
                inputs.update(text_inputs)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        if return_mm_token_type_ids:
            array_ids = np.array(inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            for i, seq_lengths in enumerate(batch_image_seq_lengths):
                image_start_positions = np.where(array_ids[i] == self.fake_image_token_id)[0]
                j = 0
                for seq_len in seq_lengths:
                    if j >= len(image_start_positions):
                        break
                    start = image_start_positions[j]
                    end = start + seq_len
                    mm_token_type_ids[i, start:end] = 1
                    j = np.searchsorted(image_start_positions, end)

            inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data=inputs, tensor_type=return_tensors)

## Modeling ##

class GraniteDoclingHybridBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    pass


class GraniteDoclingHybridCausalLMOutputWithPast(Idefics3CausalLMOutputWithPast):
    pass


class HybridMambaAttentionDynamicCache(HybridMambaAttentionDynamicCache):
    pass


class GraniteDoclingHybridPreTrainedModel(Idefics3PreTrainedModel):
    config_class = GraniteDoclingHybridConfig


class GraniteDoclingHybridModel(Idefics3Model):
    config_class = GraniteDoclingHybridConfig

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | GraniteDoclingHybridBaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
            the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
            max_num_images is the maximum number of images among the batch_size samples in the batch.
            Padding images are not needed beyond padding the pixel_values at the entrance of the model.
            For efficiency, we only pass through the vision_model's forward the real images by
            discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
            image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask).pooler_output
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        return GraniteDoclingHybridBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The GraniteDoclingHybrid Model with a language modeling head. It is made up of a SigLIP vision encoder,
    with a GraniteMoeHybrid language model on top.
    """
)
class GraniteDoclingHybridForConditionalGeneration(Idefics3ForConditionalGeneration):
    config_class = GraniteDoclingHybridConfig

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteDoclingHybridCausalLMOutputWithPast:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `GraniteDoclingHybridForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
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

        return GraniteDoclingHybridCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten to handle HybridMambaAttentionDynamicCache initialization

        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Initialize HybridMambaAttentionDynamicCache if needed
        if model_inputs.get("use_cache", True) and not isinstance(
            model_inputs.get("past_key_values"), HybridMambaAttentionDynamicCache
        ):
            cache_source = model_inputs.get("inputs_embeds")
            if cache_source is None:
                cache_source = model_inputs.get("decoder_inputs_embeds")
            if cache_source is not None:
                batch_size = cache_source.shape[0]
                dtype = cache_source.dtype
                device = cache_source.device
            else:
                input_tensor = model_inputs.get("input_ids")
                if input_tensor is None:
                    input_tensor = model_inputs.get("decoder_input_ids")
                if input_tensor is None:
                    input_tensor = input_ids
                if input_tensor is None:
                    raise ValueError("Unable to determine batch size for GraniteMoeHybrid cache initialization.")
                batch_size = input_tensor.shape[0]
                dtype = self.model.text_model.get_input_embeddings().weight.dtype
                device = input_tensor.device

            model_inputs["past_key_values"] = HybridMambaAttentionDynamicCache(
                self.model.text_model.config,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )

        cache_position = model_inputs.get("cache_position", cache_position)
        if cache_position is None:
            cache_position = torch.zeros(1, dtype=torch.long, device=self.device)

        if image_hidden_states is not None or cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


__all__ = [
    "GraniteDoclingHybridConfig",
    "GraniteDoclingHybridForConditionalGeneration",
    "GraniteDoclingHybridModel",
    "GraniteDoclingHybridPreTrainedModel",
]
