# Copyright 2024 The Rhymes-AI Teams Authors and The HuggingFace Inc. team. All rights reserved.
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

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature, get_patch_output_size, select_best_resolution
from ...image_transforms import divide_to_patches
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_python import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_available,
    is_torchvision_available,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoTokenizer
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
)


logger = logging.get_logger(__name__)


def sequential_experts_gemm(token_states, expert_weights, tokens_per_expert):
    """
    Compute the matrix multiplication (GEMM) for each expert sequentially. This approach is computationally inefficient, especially when dealing with a large number of experts.

    Args:
        token_states (torch.Tensor): Input tensor of shape (num_tokens, in_features).
        expert_weights (torch.Tensor): Weight tensor of shape (num_experts, in_features, out_features).
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

    Returns:
        torch.Tensor: Output tensor of shape (num_tokens, out_features).
    """
    num_tokens = token_states.shape[0]
    out_features = expert_weights.shape[-1]
    output = torch.zeros(num_tokens, out_features, dtype=token_states.dtype, device=token_states.device)

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the beginning for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    for expert_num in range(expert_weights.shape[0]):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        tokens = token_states[start:end]

        out = torch.matmul(tokens, expert_weights[expert_num])
        output[start:end] = out
    return output


@auto_docstring(checkpoint="rhymes-ai/Aria")
@strict
class AriaTextConfig(LlamaConfig):
    r"""
    moe_num_experts (`int`, *optional*, defaults to 8):
        The number of experts in the MoE layer.
    moe_topk (`int`, *optional*, defaults to 2):
        The number of top experts to route to for each token.
    moe_num_shared_experts (`int`, *optional*, defaults to 2):
        The number of shared experts.
    """

    model_type = "aria_text"
    base_config_key = "text_config"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }

    intermediate_size: int = 4096
    moe_num_experts: int = 8
    moe_topk: int = 2
    moe_num_shared_experts: int = 2
    pad_token_id: int | None = 2


@auto_docstring(checkpoint="rhymes-ai/Aria")
@strict
class AriaConfig(PreTrainedConfig):
    r"""
    projector_patch_to_query_dict (`dict`, *optional*):
        Mapping of patch sizes to query dimensions.
    """

    model_type = "aria"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AriaTextConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | AriaTextConfig | None = None
    vision_feature_layer: int | list[int] = -1
    projector_patch_to_query_dict: dict | None = None
    image_token_index: int = 9
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # Convert the keys and values of projector_patch_to_query_dict to integers
        # This ensures consistency even if they were provided as strings
        if self.projector_patch_to_query_dict is None:
            self.projector_patch_to_query_dict = {
                1225: 128,
                4900: 256,
            }
        self.projector_patch_to_query_dict = {int(k): int(v) for k, v in self.projector_patch_to_query_dict.items()}
        self.max_value_projector_patch_to_query_dict = max(self.projector_patch_to_query_dict.values())

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = "idefics3_vision"
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["idefics3_vision"]()

        if isinstance(self.text_config, dict) and "model_type" in self.text_config:
            self.text_config = AriaTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = AriaTextConfig()

        super().__post_init__(**kwargs)


class AriaTextRMSNorm(LlamaRMSNorm):
    pass


class AriaProjectorMLP(nn.Module):
    """
    Feed-Forward Network module for the Aria Projector.

    Args:
        in_features (`int`):
            Input embedding dimension.
        hidden_features (`int`):
            Hidden dimension of the feed-forward network.
        output_dim (`int`):
            Output dimension.
    """

    def __init__(self, in_features, hidden_features, output_dim):
        super().__init__()
        self.linear_in = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_out = nn.Linear(hidden_features, output_dim, bias=False)
        self.act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_states = self.act(self.linear_in(hidden_states))
        hidden_states = self.linear_out(hidden_states)
        return hidden_states


class AriaCrossAttention(nn.Module):
    """
    Aria Cross-Attention module.

    Args:
        config (`AriaConfig`):
            The configuration to use.
    """

    def __init__(self, config: AriaConfig, dropout_rate: float = 0):
        super().__init__()
        hidden_size = config.vision_config.hidden_size
        num_heads = config.vision_config.num_attention_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Original code here: https://github.com/rhymes-ai/Aria/blob/719ff4e52b727443cba3793b0e27fe64e0244fe1/aria/model/projector.py#L48
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer_norm_kv = nn.LayerNorm(hidden_size)

    def forward(self, key_value_states, hidden_states, attn_mask=None):
        """
        Forward pass of the AriaCrossAttention module.

        Args:
            key_value_states (`torch.Tensor`):
                Input tensor for key and value.
            hidden_states (`torch.Tensor`):
                Input tensor for query.
            attn_mask (`torch.Tensor`, *optional*, defaults to None):
                Attention mask.

        Returns:
            torch.Tensor:
                Output tensor after cross-attention.
        """
        query = self.q_proj(self.layer_norm(hidden_states))

        key_value_states = self.layer_norm_kv(key_value_states)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)

        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)

        attn_output = self.dropout(self.linear(attn_output))

        return attn_output


class AriaProjector(nn.Module):
    """
    Aria Projector module.

    This module projects vision features into the language model's embedding space, enabling interaction between vision and language components.

    Args:
        config (`AriaConfig`):
            Configuration object for the model.
    """

    def __init__(
        self,
        config: AriaConfig,
    ):
        super().__init__()

        self.patch_to_query_dict = config.projector_patch_to_query_dict
        self.in_features = config.vision_config.hidden_size
        self.num_heads = config.vision_config.num_attention_heads
        self.kv_dim = config.vision_config.hidden_size
        self.hidden_features = config.text_config.hidden_size
        self.output_dim = config.text_config.hidden_size

        self.query = nn.Parameter(torch.zeros(config.max_value_projector_patch_to_query_dict, self.in_features))

        self.cross_attn = AriaCrossAttention(config)

        self.layer_norm = nn.LayerNorm(self.in_features)
        self.feed_forward = AriaProjectorMLP(self.in_features, self.hidden_features, self.output_dim)

    def forward(self, key_value_states: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        Forward pass of the Projector module.

        Args:
            key_value_states (`torch.Tensor`):
                Input tensor of shape (batch_size, num_patches, kv_dim).
            attn_mask (`torch.Tensor`, *optional*, default is None):
                Attention mask.

        Returns:
            `torch.Tensor`: Output tensor of shape (batch_size, query_number, output_dim).
        """
        batch_size, num_patches = key_value_states.shape[0], key_value_states.shape[1]

        if num_patches not in self.patch_to_query_dict:
            raise KeyError(
                f"Number of patches {num_patches} not found in patch_to_query_dict amongst possible values {self.patch_to_query_dict.keys()}."
            )
        query_num = self.patch_to_query_dict[num_patches]

        queries = self.query[:query_num].unsqueeze(0).repeat(batch_size, 1, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1)

        attention_out = self.cross_attn(key_value_states, queries, attn_mask=attn_mask)

        out = self.feed_forward(self.layer_norm(attention_out))

        return out


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


class AriaImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    max_image_size (`int`, *optional*, defaults to `self.max_image_size`):
        Maximum image size. Must be either 490 or 980.
    min_image_size (`int`, *optional*, defaults to `self.min_image_size`):
        Minimum image size. Images smaller than this in any dimension will be scaled up.
    split_resolutions (`list[list[int]]`, *optional*, defaults to `self.split_resolutions`):
        A list of possible resolutions as (height, width) pairs for splitting high-resolution images into patches.
    split_image (`bool`, *optional*, defaults to `self.split_image`):
        Whether to split the image into patches using the best matching resolution from `split_resolutions`.
    """

    max_image_size: int
    min_image_size: int
    split_resolutions: list[list[int]]
    split_image: bool


@auto_docstring
class AriaImageProcessor(TorchvisionBackend):
    model_input_names = ["pixel_values", "pixel_mask", "num_crops"]
    valid_kwargs = AriaImageProcessorKwargs

    resample = PILImageResampling.BICUBIC
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    max_image_size = 980
    min_image_size = 336
    split_image = False
    split_resolutions = None
    do_convert_rgb = True
    do_rescale = True
    do_normalize = True

    def __init__(self, **kwargs: Unpack[AriaImageProcessorKwargs]):
        if kwargs.get("split_resolutions") is None:
            default_resolutions = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (6, 1), (7, 1), (8, 1)]  # fmt: skip
            kwargs["split_resolutions"] = [[el[0] * 490, el[1] * 490] for el in default_resolutions]
        super().__init__(**kwargs)

    def _get_padding_size(self, original_resolution: tuple, target_resolution: tuple) -> list[int]:
        """Get padding size for patching, returns [left, top, right, bottom] for tvF.pad."""
        original_height, original_width = original_resolution
        target_height, target_width = target_resolution
        paste_x, r_x = divmod(target_width - original_width, 2)
        paste_y, r_y = divmod(target_height - original_height, 2)
        return [paste_x, paste_y, paste_x + r_x, paste_y + r_y]

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    ) -> "torch.Tensor":
        """Resize an image to a target resolution while maintaining aspect ratio."""
        new_height, new_width = get_patch_output_size(
            image, target_resolution, input_data_format=ChannelDimension.FIRST
        )
        return self.resize(image, SizeDict(height=new_height, width=new_width), resample)

    def _pad_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
    ) -> "torch.Tensor":
        """Pad an image to a target resolution while maintaining aspect ratio."""
        new_resolution = get_patch_output_size(image, target_resolution, input_data_format=ChannelDimension.FIRST)
        padding = self._get_padding_size(new_resolution, target_resolution)
        return tvF.pad(image, padding=padding)

    def get_image_patches(
        self,
        image: "torch.Tensor",
        grid_pinpoints: list[list[int]],
        patch_size: int,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    ) -> list["torch.Tensor"]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (`torch.Tensor`):
                The input image to be processed (channels-first format).
            grid_pinpoints (`list[list[int]]`):
                A list of possible resolutions as (height, width) pairs.
            patch_size (`int`):
                Size of each square patch to divide the image into.
            resample (`PILImageResampling | tvF.InterpolationMode | int | None`):
                Resampling filter to use when resizing.

        Returns:
            `list[torch.Tensor]`: A list of image patches in channels-first format.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        resized_image = self._resize_for_patching(image, best_resolution, resample)
        padded_image = self._pad_for_patching(resized_image, best_resolution)
        patches = divide_to_patches(padded_image, patch_size=patch_size)
        return patches

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        max_image_size: int = 980,
        min_image_size: int = 336,
        split_resolutions: list[list[int]] | None = None,
        split_image: bool = False,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> BatchFeature:
        if max_image_size not in [490, 980]:
            raise ValueError("max_image_size must be either 490 or 980")

        pixel_masks = []
        processed_crops = []
        num_crops = None

        for image in images:
            if split_image:
                crop_images = self.get_image_patches(image, split_resolutions, max_image_size, resample)
            else:
                crop_images = [image]

            if num_crops is None or len(crop_images) > num_crops:
                num_crops = len(crop_images)

            for crop_image in crop_images:
                h, w = crop_image.shape[-2], crop_image.shape[-1]
                scale = max_image_size / max(h, w)
                if w >= h:
                    new_h = max(int(h * scale), min_image_size)
                    new_w = max_image_size
                else:
                    new_h = max_image_size
                    new_w = max(int(w * scale), min_image_size)

                crop_image = self.resize(crop_image, SizeDict(height=new_h, width=new_w), resample)

                padding_bottom = max_image_size - new_h
                padding_right = max_image_size - new_w
                crop_image = tvF.pad(crop_image, [0, 0, padding_right, padding_bottom])

                pixel_mask = torch.zeros((max_image_size, max_image_size), dtype=torch.bool)
                pixel_mask[:new_h, :new_w] = True
                pixel_masks.append(pixel_mask)
                processed_crops.append(crop_image)

        stacked_images = torch.stack(processed_crops, dim=0)
        stacked_images = self.rescale_and_normalize(
            stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )
        stacked_masks = torch.stack(pixel_masks, dim=0)

        return BatchFeature(
            data={
                "pixel_values": stacked_images,
                "pixel_mask": stacked_masks,
                "num_crops": num_crops,
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*):
                Any kwargs to override defaults of the image processor.

        Returns:
            `int`: Number of patches per image.
        """
        split_image = images_kwargs.get("split_image", self.split_image)
        max_image_size = images_kwargs.get("max_image_size", self.max_image_size)

        resized_height, resized_width = select_best_resolution((height, width), self.split_resolutions)
        num_patches = 1 if not split_image else resized_height // max_image_size * resized_width // max_image_size
        return num_patches


class AriaImagesKwargs(ImagesKwargs, total=False):
    """
    split_image (`bool`, *optional*, defaults to `False`):
        Whether to split large images into multiple crops. When enabled, images exceeding the maximum size are
        divided into overlapping crops that are processed separately and then combined. This allows processing
        of very high-resolution images that exceed the model's input size limits.
    max_image_size (`int`, *optional*, defaults to `980`):
        Maximum image size (in pixels) for a single image crop. Images larger than this will be split into
        multiple crops when `split_image=True`, or resized if splitting is disabled. This parameter controls
        the maximum resolution of individual image patches processed by the model.
    min_image_size (`int`, *optional*):
        Minimum image size (in pixels) for a single image crop. Images smaller than this will be upscaled to
        meet the minimum requirement. If not specified, images are processed at their original size (subject
        to the maximum size constraint).
    """

    split_image: bool
    max_image_size: int
    min_image_size: int


class AriaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: AriaImagesKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "max_image_size": 980,
            "split_image": False,
        },
        "return_tensors": TensorType.PYTORCH,
    }


@auto_docstring
class AriaProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer: AutoTokenizer | str = None,
        chat_template: str | None = None,
        size_conversion: dict[float | int, int] | None = None,
    ):
        r"""
        size_conversion (`Dict`, *optional*):
            A dictionary indicating size conversions for images.
        """
        if size_conversion is None:
            size_conversion = {490: 128, 980: 256}
        self.size_conversion = {int(k): v for k, v in size_conversion.items()}

        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: ImageInput | None = None,
        **kwargs: Unpack[AriaProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
            `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_mask** -- Pixel mask to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            AriaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            # expand the image_token according to the num_crops and tokens per image
            tokens_per_image = self.size_conversion[image_inputs.pixel_values.shape[2]]
            prompt_strings = []
            num_crops = image_inputs.pop("num_crops") * tokens_per_image
            for sample in text:
                sample = sample.replace(self.tokenizer.image_token, self.tokenizer.image_token * num_crops)
                prompt_strings.append(sample)

        else:
            image_inputs = {}
            prompt_strings = text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

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
            images_kwargs = AriaProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            max_size = images_kwargs.get("max_image_size", None) or self.image_processor.max_image_size
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [self.size_conversion[max_size] * num_patches for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        # Remove `num_crops`, it is popped and used only when processing. Make a copy of list when removing
        # otherwise `self.image_processor.model_input_names` is also modified
        image_processor_input_names = [name for name in image_processor_input_names if name != "num_crops"]
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class AriaSharedExpertsMLP(LlamaMLP):
    """
    Shared Expert MLP for shared experts.

    Unlike routed experts, shared experts process all tokens without routing.
    This class reconfigures the intermediate size in comparison to the LlamaMLP.

    Args:
        config (`AriaTextConfig`): Configuration object for the Aria language model.
    """

    def __init__(self, config: AriaTextConfig):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size * config.moe_num_shared_experts


class AriaGroupedExpertsGemm(nn.Module):
    """
    Grouped GEMM (General Matrix Multiplication) module for efficient expert computation.
    This module utilizes the grouped_gemm library (https://github.com/fanshiqing/grouped_gemm)
    for optimized performance. If the grouped_gemm library is not installed, it gracefully
    falls back to a sequential GEMM implementation, which may be slower but ensures
    functionality.

    Args:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        groups (`int`):
            Number of expert groups.
    """

    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, in_features, out_features))

    def forward(self, input, tokens_per_expert):
        """
        Perform grouped matrix multiplication.

        Args:
            input (`torch.Tensor`):
                Input tensor of shape (num_tokens, in_features).
            tokens_per_expert (`torch.Tensor`):
                Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor of shape (num_tokens, out_features).
        """
        return sequential_experts_gemm(
            input,
            self.weight,
            tokens_per_expert.cpu(),
        )


class AriaExperts(nn.Module):
    def __init__(self, config: AriaTextConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = AriaGroupedExpertsGemm(config.hidden_size, config.intermediate_size * 2, config.moe_num_experts)
        self.fc2 = AriaGroupedExpertsGemm(config.intermediate_size, config.hidden_size, config.moe_num_experts)

    def route_tokens_to_experts(self, router_logits):
        top_logits, top_indices = torch.topk(router_logits, k=self.config.moe_topk, dim=1)
        scores = nn.functional.softmax(top_logits, dim=-1)
        return top_indices, scores

    def forward(self, hidden_states, router_logits) -> torch.Tensor:
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        original_dtype = top_k_index.dtype
        tokens_per_expert = torch.histc(
            top_k_index.flatten().to(torch.float32),
            bins=self.config.moe_num_experts,
            min=0,
            max=self.config.moe_num_experts - 1,
        ).to(original_dtype)
        indices = top_k_index

        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices)
        permuted_tokens = hidden_states.index_select(0, sorted_indices // self.config.moe_topk)

        fc1_output = self.fc1(permuted_tokens, tokens_per_expert)
        projection, gate = torch.chunk(fc1_output, 2, dim=-1)
        fc1_output = nn.functional.silu(projection) * gate
        expert_output = self.fc2(fc1_output, tokens_per_expert)

        unpermuted_tokens = torch.zeros(
            (top_k_weights.shape[0] * self.config.moe_topk, expert_output.size(1)),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        unpermuted_tokens.index_copy_(0, sorted_indices, expert_output)
        unpermuted_tokens = unpermuted_tokens.view(-1, self.config.moe_topk, expert_output.size(1))

        output = (unpermuted_tokens * top_k_weights.unsqueeze(-1)).sum(dim=1)
        return output


class AriaTextMoELayer(nn.Module):
    def __init__(self, config: AriaTextConfig):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.experts = AriaExperts(config)
        self.shared_experts = AriaSharedExpertsMLP(config)
        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        router_logits = self.router(hidden_states)
        expert_output = self.experts(hidden_states, router_logits).view(original_shape)
        shared_expert_output = self.shared_experts(hidden_states.view(original_shape))
        return expert_output + shared_expert_output


class AriaTextAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""


class AriaTextDecoderLayer(LlamaDecoderLayer):
    """
    Aria Text Decoder Layer.

    This class defines a single decoder layer in the language model, incorporating self-attention and Mixture of Experts (MoE) feed-forward network.

    Args:
        config (`AriaTextConfig`):
            Configuration object for the text component of the model.
        layer_idx (`int`):
            Index of the layer.
    """

    def __init__(self, config: AriaTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = AriaTextMoELayer(config)


@auto_docstring
class AriaTextPreTrainedModel(PreTrainedModel):
    config: AriaTextConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    _no_split_modules = ["AriaTextDecoderLayer", "AriaGroupedExpertsGemm"]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": AriaTextDecoderLayer,
        "attentions": AriaTextAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AriaGroupedExpertsGemm):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class AriaPreTrainedModel(LlamaPreTrainedModel):
    config: AriaConfig
    base_model_prefix = "model"
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (dynamic slicing)
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, AriaProjector):
            init.trunc_normal_(module.query, std=self.config.initializer_range)


class AriaTextModel(LlamaModel):
    def __init__(self, config: AriaTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AriaTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.post_init()


class AriaTextForCausalLM(AriaTextPreTrainedModel, LlamaForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: AriaTextConfig):
        super().__init__(config)
        self.model = AriaTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(self, **super_kwargs):
        super().forward(self, **super_kwargs)


class AriaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class AriaModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class AriaModel(LlavaModel):
    def __init__(self, config: AriaConfig):
        super().__init__(config)
        self.multi_modal_projector = AriaProjector(config)

    def _create_patch_attention_mask(self, pixel_mask):
        if pixel_mask is None:
            return None

        patches_subgrid = pixel_mask.unfold(
            dimension=1,
            size=self.vision_tower.config.patch_size,
            step=self.vision_tower.config.patch_size,
        )
        patches_subgrid = patches_subgrid.unfold(
            dimension=2,
            size=self.vision_tower.config.patch_size,
            step=self.vision_tower.config.patch_size,
        )
        return (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] = -1,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        patch_attention_mask = self._create_patch_attention_mask(pixel_mask)
        image_outputs = self.vision_tower(
            pixel_values,
            patch_attention_mask=patch_attention_mask,
            output_hidden_states=True,  # Ignore arg on purpose
            return_dict=True,
            **kwargs,
        )
        image_attn_mask = None
        if patch_attention_mask is not None:
            flattened_mask = patch_attention_mask.flatten(1)
            image_attn_mask = torch.logical_not(flattened_mask)

        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        image_outputs.pooler_output = self.multi_modal_projector(selected_image_feature, attn_mask=image_attn_mask)

        return image_outputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_mask: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | AriaModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and inputs_embeds.shape[1] != 1:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                vision_feature_layer=self.config.vision_feature_layer,
                return_dict=True,
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return AriaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


@auto_docstring(
    custom_intro="""
    Aria model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language model
    to perform tasks that involve both image and text inputs.
    """
)
class AriaForConditionalGeneration(LlavaForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] = -1,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            vision_feature_layer=vision_feature_layer,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_mask: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | AriaCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `AriaForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> import httpx
        >>> from io import BytesIO
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModel
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("Rhymes-AI/Aria")
        >>> model = AutoModel.from_pretrained("Rhymes-AI/Aria", dtype=torch.bfloat16, device_map="auto")

        >>> # Create inputs
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image"},
        ...             {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
        ...             {"type": "image"},
        ...             {"type": "text", "text": "What can we see in this image?"},
        ...         ]
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image"},
        ...             {"type": "text", "text": "In which city is that bridge located?"},
        ...         ]
        ...     }
        ... ]

        >>> prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
        >>> images = [[image1, image2], [image3]]
        >>> inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts[0])
        Assistant: There are buildings, trees, lights, and water visible in this image.

        >>> print(generated_texts[1])
        Assistant: The bridge is in San Francisco.
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
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

        return AriaCausalLMOutputWithPast(
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
        inputs_embeds=None,
        pixel_values=None,
        pixel_mask=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            # Pixel values are used only in the first iteration if available
            # In subsequent iterations, they are already merged with text and cached
            # NOTE: first iteration doesn't have to be prefill, it can be the first
            # iteration with a question and cached system prompt (continue generate from cache)
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_mask"] = pixel_mask

        return model_inputs


__all__ = [
    "AriaConfig",
    "AriaTextConfig",
    "AriaImageProcessor",
    "AriaProcessor",
    "AriaForConditionalGeneration",
    "AriaPreTrainedModel",
    "AriaTextPreTrainedModel",
    "AriaTextModel",
    "AriaModel",
    "AriaTextForCausalLM",
]
