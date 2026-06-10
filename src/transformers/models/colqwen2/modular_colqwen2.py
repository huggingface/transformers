# Copyright 2025 The HuggingFace Inc. team.
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

from dataclasses import dataclass

from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import ModelOutput, auto_docstring, can_return_tuple, is_torch_available, logging
from ..colpali.modeling_colpali import ColPaliForRetrieval, ColPaliPreTrainedModel
from ..colpali.processing_colpali import ColPaliProcessor
from .configuration_colqwen2 import ColQwen2Config


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class ColQwen2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
            "return_mm_token_type_ids": False,
            "return_text_replacement_offsets": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ColQwen2Processor(ColPaliProcessor):
    valid_processor_kwargs = ColQwen2ProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        visual_prompt_prefix: str | None = None,
        query_prefix: str | None = None,
        **kwargs,
    ):
        r"""
        visual_prompt_prefix (`str`, *optional*, defaults to `"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"`):
            A string that gets tokenized and prepended to the image tokens.
        query_prefix (`str`, *optional*, defaults to `"Query: "`):
            A prefix to be used for the query.
        """
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template)
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token

        self.visual_prompt_prefix = visual_prompt_prefix or (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
        )
        self.query_prefix = query_prefix or "Query: "
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if images is not None:
            if is_valid_image(images):
                images = [images]
            elif not (isinstance(images, list) and (is_valid_image(images[0]) or (isinstance(images[0], list) and is_valid_image(images[0][0])))):
                raise ValueError("images must be an image, list of images or list of list of images")
            images, _, videos, audio = super().prepare_inputs_layout(images=images, **kwargs)
            text = [self.visual_prompt_prefix] * len(images)
            return images, text, videos, audio
        return super().prepare_inputs_layout(images=images, text=text, videos=videos, audio=audio, **kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        merge_length = self.image_processor.merge_size**2
        return self.image_token * (int(image_inputs["image_grid_thw"][image_idx].prod()) // merge_length)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None and text is not None:
            raise ValueError("Only one of text or images can be processed at a time")
        suffix = kwargs.pop("suffix", None)
        if images is None:
            # Query mode: augment text before base class tokenizes it
            return_token_type_ids = suffix is not None
            if suffix is None:
                suffix = self.query_augmentation_token * 10
            if isinstance(text, str):
                text = [text]
            text = [f"{self.query_prefix}{q}{suffix}" for q in text]
            kwargs["return_token_type_ids"] = return_token_type_ids
        return_data = super().__call__(images=images, text=text, **kwargs)
        if images is not None:
            # DDP-aware pixel_values re-padding
            offsets = return_data["image_grid_thw"][:, 1] * return_data["image_grid_thw"][:, 2]
            pixel_values = list(torch.split(return_data["pixel_values"], offsets.tolist()))
            return_data["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True)
        return return_data

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = ColQwen2ProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        # ColQwen doesn't process videos. Make a copy of list when removing
        # otherwise `self.feature_extractor.model_input_names` is also modified
        image_processor_input_names = [
            name for name in image_processor_input_names if name not in ["pixel_values_videos", "video_grid_thw"]
        ]
        return tokenizer_input_names + image_processor_input_names


class ColQwen2PreTrainedModel(ColPaliPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    Base class for ColQwen2 embeddings output.
    """
)
@dataclass
class ColQwen2ForRetrievalOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        The embeddings of the model.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: torch.FloatTensor | None = None
    embeddings: torch.Tensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    Following the ColPali approach, ColQwen2 leverages VLMs to construct efficient multi-vector embeddings directly
    from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
    between these document embeddings and the corresponding query embeddings, using the late interaction method
    introduced in ColBERT.

    Using ColQwen2 removes the need for potentially complex and brittle layout recognition and OCR pipelines with
    a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

    ColQwen2 is part of the ColVision model family, which was introduced with ColPali in the following paper:
    [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://huggingface.co/papers/2407.01449).
    """
)
class ColQwen2ForRetrieval(ColPaliForRetrieval):
    def __init__(self, config: ColQwen2Config):
        super().__init__(config)
        del self._tied_weights_keys

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ) -> ColQwen2ForRetrievalOutput:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if pixel_values is not None and image_grid_thw is not None:
            # NOTE: image_grid_thw: (batch_size, 3) where image_grid_thw[i] = (num_patches_h, num_patches_w, temporal_patch_size)
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            arange = torch.arange(pixel_values.shape[1], device=offsets.device)  # (max_len,)
            mask = arange.unsqueeze(0) < offsets.unsqueeze(1)  # (batch_size, max_len)
            pixel_values = pixel_values[mask]  # (total_valid_patches, channels, height, width)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Custom data preparation to fix an issue with the gradient flow when training with multiple GPUs.
        if inputs_embeds is None:
            inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                image_embeds = self.vlm.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True).pooler_output
                image_mask = (
                    (input_ids == self.config.vlm_config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        vlm_output = self.vlm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vlm_hidden_states = vlm_output.hidden_states if output_hidden_states else None

        last_hidden_states = vlm_output[0]  # (batch_size, sequence_length, hidden_size)
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(last_hidden_states.to(proj_dtype))  # (batch_size, sequence_length, dim)

        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return ColQwen2ForRetrievalOutput(
            embeddings=embeddings,
            past_key_values=vlm_output.past_key_values,
            hidden_states=vlm_hidden_states,
            attentions=vlm_output.attentions,
        )


__all__ = [
    "ColQwen2ForRetrieval",
    "ColQwen2PreTrainedModel",
    "ColQwen2Processor",
]
