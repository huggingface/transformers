import inspect
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_processing_utils import BaseImageProcessor, select_best_resolution
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
    pad,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    to_numpy_array,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import (
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TruncationStrategy,
)
from ...utils import (
    logging,
)
from ...utils.import_utils import is_vision_available
from ..auto import CONFIG_MAPPING, AutoModel, AutoModelForCausalLM, AutoTokenizer
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ..llava.modeling_llava import LlavaCausalLMOutputWithPast


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image, ImageOps


def sequential_gemm(input, weight, tokens_per_expert):
    """
    Compute the matrix multiplication (GEMM) for each expert sequentially. This approach is computationally inefficient, especially when dealing with a large number of experts.

    Args:
        input (torch.Tensor): Input tensor of shape (num_tokens, in_features).
        weight (torch.Tensor): Weight tensor of shape (num_experts, in_features, out_features).
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

    Returns:
        torch.Tensor: Output tensor of shape (num_tokens, out_features).
    """
    num_tokens = input.shape[0]
    out_features = weight.shape[-1]
    output = torch.zeros(num_tokens, out_features, dtype=input.dtype, device=input.device)

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the begining for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    for expert_num in range(weight.shape[0]):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        tokens = input[start:end]

        out = torch.matmul(tokens, weight[expert_num])
        output[start:end] = out
    return output


try:
    from grouped_gemm.ops import gmm as experts_gemm

    if os.environ.get("USE_GROUPED_GEMM", "1") == "0":
        logger.warning("environment variable USE_GROUPED_GEMM is set to 0, using sequential GEMM instead.")
        experts_gemm = sequential_gemm
except ImportError as e:
    logger.warning("`grouped_gemm` is not installed, using sequential GEMM, which is slower.")
    experts_gemm = sequential_gemm


class IdentityOp(torch.nn.Module):
    """
    An identity operation that returns the input unchanged.

    This can be used as a placeholder or to maintain architectural consistency
    when a specific operation is not needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class AriaTextConfig(LlamaConfig):
    """
    Configuration class for Aria language model.

    This class extends the LlamaConfig to include additional parameters specific to the Mixture of Experts (MoE) architecture.

    Args:
        moe_intermediate_size (`int`): The intermediate size for MoE layers. Default is 4096.
        moe_num_experts (int): The number of experts in the MoE layer. Default is 8.
        moe_topk (int): The number of top experts to route to for each token. Default is 2.
        moe_z_loss_coeff (float): The coefficient for the auxiliary z-loss. Default is 1e-5.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary load balancing loss. Default is 1e-3.
        moe_num_shared_experts (int): The number of shared experts. Default is 2.
        **kwargs: Additional keyword arguments to be passed to the parent LlamaConfig.
    """

    model_type = "aria_text_model"

    def __init__(
        self,
        moe_intermediate_size: int = 4096,
        moe_num_experts: int = 8,
        moe_topk: int = 2,
        moe_z_loss_coeff: float = 1e-5,
        moe_aux_loss_coeff: float = 1e-3,
        moe_num_shared_experts: int = 2,
        pad_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_num_shared_experts = moe_num_shared_experts


class AriaConfig(PretrainedConfig):
    """
    Configuration class for Aria model.

    This class handles the configuration for both vision and text components of the Aria model,
    as well as additional parameters for image token handling and projector mapping.

    Args:
        vision_config (AriaVisionConfig or dict): Configuration for the vision component.
        text_config (AriaTextConfig or dict): Configuration for the text component.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query dimensions.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        **kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        model_type (str): Type of the model, set to "aria".
        is_composition (bool): Whether the model is a composition of multiple components.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query dimensions.
        vision_config (AriaVisionConfig): Configuration for the vision component.
        text_config (AriaTextConfig): Configuration for the text component.
    """

    model_type = "aria"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projector_patch_to_query_dict=None,
        ignore_index=-100,
        image_token_index=32000,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index

        # Convert the keys and values of projector_patch_to_query_dict to integers
        # This ensures consistency even if they were provided as strings
        if projector_patch_to_query_dict is None:
            projector_patch_to_query_dict = {
                1225: 128,
                4900: 256,
            }
        self.projector_patch_to_query_dict = projector_patch_to_query_dict.copy()

        if isinstance(vision_config, dict):
            vision_config["model_type"] = "idefics3_vision"
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["idefics3_vision"]()

        self.vision_config = vision_config
        self.initializer_range = initializer_range

        if isinstance(text_config, dict) and "model_type" in text_config:
            text_config = AriaTextConfig(**text_config)
        elif text_config is None:
            text_config = AriaTextConfig()

        self.text_config = text_config

        super().__init__(**kwargs)


class AriaRMSNorm(LlamaRMSNorm):
    pass


class AriaGeluDense(nn.Module):
    """
    Feed-Forward Network module.

    Args:
        in_features (int): Input embedding dimension.
        hidden_features (int): Hidden dimension of the feed-forward network.
        output_dim (int): Output dimension.
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
        config (AriaConfig): the configuration to use.
    """

    def __init__(self, config: AriaConfig, dropout_rate: float = 0):
        super().__init__()
        in_features = config.vision_config.hidden_size
        num_heads = config.vision_config.num_attention_heads
        kv_dim = config.vision_config.hidden_size
        self.num_heads = num_heads
        self.q_proj = nn.Linear(in_features, in_features, bias=False)
        self.k_proj = nn.Linear(kv_dim, in_features, bias=False)
        self.v_proj = nn.Linear(kv_dim, in_features, bias=False)

        # Use batch_first=True to simplify code by removing permutations compared to the original.
        # Original code here: https://github.com/rhymes-ai/Aria/blob/719ff4e52b727443cba3793b0e27fe64e0244fe1/aria/model/projector.py#L48
        self.multihead_attn = nn.MultiheadAttention(in_features, num_heads, batch_first=True)
        self.linear = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm = nn.LayerNorm(in_features)
        self.layer_norm_kv = nn.LayerNorm(kv_dim)

    def forward(self, x, hidden_states, attn_mask=None, add_residual=False):
        """
        Forward pass of the AriaCrossAttention module.

        Args:
            x (torch.Tensor): Input tensor for key and value.
            hidden_states (torch.Tensor): Input tensor for query.
            attn_mask (torch.Tensor, optional): Attention mask. Default is None.
            add_residual (bool): Whether to add residual connection. Default is False.

        Returns:
            torch.Tensor: Output tensor after cross-attention.
        """
        query = self.q_proj(self.layer_norm(hidden_states))

        x = self.layer_norm_kv(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)

        if add_residual:
            attn_output = hidden_states + self.dropout(self.linear(attn_output))
        else:
            attn_output = self.dropout(self.linear(attn_output))

        return attn_output


class AriaProjector(nn.Module):
    """
    A projection module with one cross-attention layer and one AriaGeluDense layer, which projects ViT's outputs into MoE's inputs.

    Args:
        config (AriaConfig): the configuration to use.

    Outputs:
        A tensor with the shape of (batch_size, query_number, output_dim)
    """

    def __init__(
        self,
        config: AriaConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_to_query_dict = config.projector_patch_to_query_dict
        self.in_features = config.vision_config.hidden_size
        self.num_heads = config.vision_config.num_attention_heads
        self.kv_dim = config.vision_config.hidden_size
        self.hidden_features = config.text_config.hidden_size
        self.output_dim = config.text_config.hidden_size

        self.query = nn.Parameter(torch.zeros(max(self.patch_to_query_dict.values()), self.in_features))

        trunc_normal_(self.query, std=0.02)

        self.cross_attn = AriaCrossAttention(config)

        self.layer_norm = nn.LayerNorm(self.in_features)
        self.feed_forward = AriaGeluDense(self.in_features, self.hidden_features, self.output_dim)  # TODO: Aria Projector MMLP
        # Removed weight inits compared to original:
        # https://github.com/rhymes-ai/Aria/blob/719ff4e52b727443cba3793b0e27fe64e0244fe1/aria/model/projector.py#L149

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None):
        """
        Forward pass of the Projector module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, kv_dim).
            attn_mask (torch.Tensor, optional): Attention mask. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_number, output_dim).
        """
        batch_size, num_patches = x.shape[0], x.shape[1]

        if num_patches not in self.patch_to_query_dict.keys():
            raise KeyError(f"Number of patches {num_patches} not found in patch_to_query_dict amongst possible values {self.patch_to_query_dict.keys()}.")
        query_num = self.patch_to_query_dict[num_patches]

        # Compared to original, simplify definition
        queries = self.query[:query_num].unsqueeze(0).repeat(batch_size, -1, -1)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1)

        attention_out = self.cross_attn(x, queries, attn_mask=attn_mask)

        out = self.feed_forward(self.layer_norm(attention_out))

        return out

# Copied from models.llava_next.image_processing_llava_next.py
def divide_to_patches(image: np.array, patch_size: int, input_data_format) -> List[np.array]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array`):
            The input image.
        patch_size (`int`):
            The size of each patch.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        list: A list of np.array representing the patches.
    """
    patches = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if input_data_format == ChannelDimension.LAST:
                patch = image[i : i + patch_size, j : j + patch_size]
            else:
                patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches


# Copied from transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = int(round(raw_size))

    if (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    elif width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = int(raw_size * height / width)
        else:
            oh = int(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = int(raw_size * width / height)
        else:
            ow = int(size * width / height)

    return (oh, ow)


class AriaImageProcessor(BaseImageProcessor):
    """
    A vision processor for the Aria model that handles image preprocessing.
    """

    def __init__(
        self,
        max_image_size=None,
        min_image_size=None,
        image_mean=None,
        image_std=None,
        split_ratio: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        Initialize the AriaImageProcessor.

        Args:
            max_image_size (int, optional): Maximum image size. Defaults to 980.
            min_image_size (int, optional): Minimum image size. Defaults to 336.
            image_mean (list, optional): Mean values for normalization. Defaults to [0.5, 0.5, 0.5].
            image_std (list, optional): Standard deviation values for normalization. Defaults to [0.5, 0.5, 0.5].
            split_ratio (list, optional): The ratio for splitting the image. Defaults to a list of common split ratios as tuples.
        """
        super().__init__(**kwargs)

        if image_mean is None:
            image_mean = [0.5, 0.5, 0.5]
        if image_std is None:
            image_std = [0.5, 0.5, 0.5]
        self.max_image_size = 980 if max_image_size is None else max_image_size
        self.min_image_size = 336 if min_image_size is None else min_image_size
        self.image_mean = image_mean
        self.image_std = image_std
        if split_ratio is None:
           self.split_ratio = [
               (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
               (1, 7), (1, 8), (2, 4), (2, 3), (2, 2),
               (2, 1), (3, 1), (3, 2), (4, 1), (4, 2),
               (5, 1), (6, 1), (7, 1), (8, 1),
           ]
       else:
           self.split_ratio = split_ratio

        # we make the transform a property so that it is lazily initialized,
        # this could avoid the error "TypeError: Object of type Normalize is not JSON serializable"
        # when we used save_pretrained or from_pretrained.
        self._transform = None
        self._set_processor_class("AriaProcessor")

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        max_image_size: int = 980,
        min_image_size: int = 336,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        split_image: Optional[bool] = False,
        do_convert_rgb: Optional[bool] = True,
        do_normalize: Optional[bool] = True,
    ):
        """
        Process a list of images.

        Args:
            images (ImageInput or list of ImageInput): The input image or a list of images.
            max_image_size (int, optional): Maximum image size. Defaults to `self.max_image_size` (980).
            min_image_size (int, optional): Minimum image size. Defaults to `self.min_image_size` (336).
            return_tensors (str or TensorType, optional): The type of tensor to return. Defaults to "pt".
            split_image (bool, optional): Whether to split the image. Defaults to False.
            do_convert_rgb (bool, optional): Whether to convert the image to RGB. Defaults to True.
            do_normalize (bool, optional): Whether to normalize the image. Defaults to True.

        Returns:
            BatchFeature: A BatchFeature object containing:
                - 'pixel_values': Tensor of processed image pixel values.
                - 'pixel_mask': Boolean pixel mask. This mask is a 2D tensor of shape (max_size, max_size) where:
                    - True (1) values indicate pixels that belong to the original resized image.
                    - False (0) values indicate pixels that are part of the padding.
                  The mask helps distinguish between actual image content and padded areas in subsequent processing steps.
                - 'num_crops': The maximum number of crops across all images.
        """
        max_size = self.max_image_size if max_image_size is None else max_image_size
        min_size = self.min_image_size if min_image_size is None else min_image_size

        if max_size not in [490, 980]:
            raise ValueError("max_image_size must be either 490 or 980")

        if not isinstance(images, list):
            images = [images]

        pixel_values = []
        pixel_masks = []
        num_crops = None

        for image in images:
            if do_convert_rgb:
                image = convert_to_rgb(image)
            image = to_numpy_array(image)
            if split_image:
                crop_images = self.get_image_patches(image, self.split_ratio, max_size, max_size)
            else:
                crop_images = [image]
            if num_crops is None or len(crop_images) > num_crops:
                num_crops = len(crop_images)
            for crop_image in crop_images:
                # At this point the scale is the rescaling factor that would bring the image to max_size in its larger dimension
                h, w = crop_image.shape[:2]
                scale = max_size / max(h, w)
                if w >= h:
                    new_size = (max(int(h * scale), min_size), max_size)  # h, w
                else:
                    new_size = (max_size, max(int(w * scale), min_size))  # h, w

                crop_image_resized = resize(crop_image, new_size, resample=Image.Resampling.BICUBIC)

                padding_bottom, padding_right = max_size - new_size[0], max_size - new_size[1]
                crop_image_padded = pad(crop_image_resized, ((0, padding_bottom), (0, padding_right)))

                # Create a pixel mask
                pixel_mask = torch.zeros(max_size, max_size, dtype=bool)
                pixel_mask[: new_size[0], : new_size[1]] = 1
                pixel_masks.append(pixel_mask)

                if do_normalize:
                    crop_image_padded = normalize(crop_image_padded, self.image_mean, self.image_std)
    
                # Switch to rgb channel first
                crop_image_padded = np.transpose(crop_image_padded, (2, 0, 1))
                pixel_values.append(crop_image_padded)
        return BatchFeature(
            data={
                "pixel_values": np.stack(pixel_values, axis=0),
                "pixel_mask": np.stack(pixel_masks, axis=0),
                "num_crops": num_crops,
            },
            tensor_type=return_tensors,
        )

    # Modified from models.llava_next.image_preprocessing_llava_next.LlavaNextImageProcessor.get_image_patches
    def get_image_patches(
        self,
        image: np.array,
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> List[np.array]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (np.array):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            List[np.array]: A list of NumPy arrays containing the processed image patches.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)

        patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]
        return patches


class AriaProcessor(ProcessorMixin):
    """
    AriaProcessor is a processor for the Aria model which wraps the Aria image preprocessor and the LLama slow tokenizer.
    Args:
        image_processor(AriaImageProcessor): The AriaImageProcessor to use for image preprocessing.
        tokenizer(AutoTokenizer): The AutoTokenizer to use for tokenizing the text.
        patch_size(int): The patch size to use for the image processor.
        chat_template(str): The chat template to use for the tokenizer.
        image_token(str): The image token to use for the tokenizer.
    """

    attributes = []
    valid_kwargs = ["chat_template", "patch_size", "image_token"]
    image_processor_class = None
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: AriaImageProcessor = None,
        tokenizer: Union[AutoTokenizer, str] = None,
        patch_size: int = 490,
        chat_template: str = None,
        image_token: str = "<|img|>",
        size_conversion: Optional[Dict] = None,
    ):
        super().__init__(chat_template=chat_template)
        if size_conversion is None:
            size_conversion = {490: 128, 980: 256}
        self.size_conversion = size_conversion

        if image_processor is None:
            self.image_processor = AriaImageProcessor(max_image_size=patch_size)
        else:
            self.image_processor = image_processor

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True, use_fast=False)
        else:
            self.tokenizer = tokenizer

        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_token = image_token

    # Modified from models.llava_next.processing_llave_next.LlavaNextProcessor.__call__
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_image_size: Optional[int] = 980,
        split_image: Optional[bool] = False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`ImageInput`, `np.ndarray`, `torch.Tensor`, `List[ImageInput]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            max_image_size (`int`, *optional*):
                Maximum size of the image to be processed.
            split_image (`bool`, *optional*):
                Whether to split the image into patches before processing.
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_mask** -- Pixel mask to be fed to a model. Returned when `images` is not `None`.
        """
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        if images is not None:
            image_inputs = self.image_processor(
                images,
                return_tensors=return_tensors,
                max_image_size=max_image_size,
                split_image=split_image,
            )
            # expand the image_token according to the num_crops and tokens per image
            tokens_per_image = self.size_conversion[image_inputs.pixel_values.shape[2]]

            prompt_strings = []
            num_crops = image_inputs.pop("num_crops") * tokens_per_image
            for sample in text:
                sample = sample.replace(self.image_token, self.image_token * num_crops)
                prompt_strings.append(sample)

        else:
            image_inputs = {}
            prompt_strings = text

        text_inputs = self.tokenizer(
            prompt_strings,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        return BatchFeature(data={**text_inputs, **image_inputs})

    @staticmethod
    def _extract_kwargs(func: callable, **kwargs) -> dict:
        """
        Extract the kwargs that are valid for the given function.
        """
        return {k: v for k, v in kwargs.items() if k in inspect.signature(func).parameters}

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save both the image processor and tokenizer.
        """
        if self.image_processor is not None:
            self.image_processor.save_pretrained(
                save_directory,
                **self._extract_kwargs(self.image_processor.save_pretrained, **kwargs),
            )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(
                save_directory,
                **self._extract_kwargs(self.tokenizer.save_pretrained, **kwargs),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        tokenizer_path=None,
        image_processor_path=None,
        **kwargs,
    ):
        """
        Load both the image processor and tokenizer from a pretrained model path.
        """
        tokenizer_path = tokenizer_path if tokenizer_path is not None else pretrained_model_name_or_path
        image_processor_path = (
            image_processor_path if image_processor_path is not None else pretrained_model_name_or_path
        )
        image_processor = AriaImageProcessor.from_pretrained(
            image_processor_path,
            **cls._extract_kwargs(AriaImageProcessor.from_pretrained, **kwargs),
        )
        if "use_fast" in kwargs:
            logger.warning("use_fast is not supported for AriaProcessor. Ignoring...")
            kwargs.pop("use_fast")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=False,
            **cls._extract_kwargs(AutoTokenizer.from_pretrained, **kwargs),
        )
        chat_template = tokenizer.chat_template

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class AriaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = AriaConfig
    base_model_prefix = "model"
    _no_split_modules = []
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA (Scaled Dot Product Attention) or not.
        """
        return self.language_model._supports_sdpa

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, AriaGroupedGEMM):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


# adapted from https://github.com/NVIDIA/Megatron-LM/blob/54f1f78529cbc2b9cddad313e7f9d96ac0420a27/megatron/core/transformer/moe/router.py#L96-L304
class AriaTopKRouter(nn.Module):
    """
    Top-K Router for Mixture of Experts (MoE) models.

    This router determines which experts should process each token based on the top-k scoring experts.
    It also applies auxiliary losses to encourage load balancing among experts.

    Args:
        config (AriaTextConfig): Configuration object containing MoE-related parameters.
    """

    def __init__(self, config: AriaTextConfig):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(torch.empty((self.config.moe_num_experts, self.config.hidden_size)))
        # FIXME: initialize the weight

    # Simplify code a lot compared to original, since we do not need training.
    # Original: https://github.com/rhymes-ai/Aria/blob/719ff4e52b727443cba3793b0e27fe64e0244fe1/aria/model/moe_lm.py#L170
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states, self.weight)
        top_logits, top_indices = torch.topk(logits, k=self.config.moe_topk, dim=1)
        scores = F.softmax(top_logits, dim=-1)

        original_dtype = top_indices.dtype

        tokens_per_expert = torch.histc(
            top_indices.flatten().to(torch.float32),
            bins=self.config.moe_num_experts,
            min=0,
            max=self.config.moe_num_experts - 1,
        )

        return scores, top_indices, tokens_per_expert.to(original_dtype)


class AriaSharedExpertsMLP(LlamaMLP):
    """
    Shared Expert MLP for shared experts.

    Unlike routed experts, shared experts process all tokens without routing.
    This class reconfigures the intermediate size in comparison to the LlamaMLP.

    Args:
        config (AriaTextConfig): Configuration object for the Aria language model.
    """

    def __init__(self, config: AriaTextConfig):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size * config.moe_num_shared_experts
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]


class AriaGroupedGEMM(nn.Module):
    """
    Grouped GEMM (General Matrix Multiplication) module for efficient expert computation.
    This module utilizes the grouped_gemm library (https://github.com/fanshiqing/grouped_gemm)
    for optimized performance. If the grouped_gemm library is not installed, it gracefully
    falls back to a sequential GEMM implementation, which may be slower but ensures
    functionality.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        groups (int): Number of expert groups.
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
            input (torch.Tensor): Input tensor of shape (num_tokens, in_features).
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor of shape (num_tokens, out_features).
        """
        tokens_per_expert = tokens_per_expert.cpu()

        # Ensure the CUDA device matches the input tensor's device.
        # This mismatch can occur when using `transformers.AutoModel.from_pretrained`
        # with `device_map="auto"` on a multi-GPU setup.
        if torch.cuda.is_available():
            torch.cuda.set_device(input.device)
        original_dtype = input.dtype
        return experts_gemm(input.to(torch.bfloat16), self.weight.to(torch.bfloat16), tokens_per_expert).to(
            original_dtype
        )


class AriaGroupedMLP(nn.Module):
    """
    Grouped MLP module for Mixture of Experts.

    Args:
        config (AriaTextConfig): Configuration object for the model.
    """

    def __init__(self, config: AriaTextConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = AriaGroupedGEMM(config.hidden_size, config.moe_intermediate_size * 2, config.moe_num_experts)
        self.fc2 = AriaGroupedGEMM(config.moe_intermediate_size, config.hidden_size, config.moe_num_experts)

    def forward(self, permuted_tokens, tokens_per_expert):
        """
        Forward pass of the Grouped MLP.

        Args:
            permuted_tokens (torch.Tensor): Permuted input tokens.
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        fc1_output = self.fc1(permuted_tokens, tokens_per_expert)
        x = torch.chunk(fc1_output, 2, dim=-1)
        fc1_output = F.silu(x[0]) * x[1]
        fc2_output = self.fc2(fc1_output, tokens_per_expert)
        return fc2_output


# Token permutation adapted from https://github.com/NVIDIA/Megatron-LM/blob/54f1f78529cbc2b9cddad313e7f9d96ac0420a27/megatron/core/transformer/moe/token_dispatcher.py#L291-L587
class AriaTextMoELayer(nn.Module):  # TODO: check naming convenstion for InstructBLIP, CLIP, etc
    """
    Mixture of Experts (MoE) Layer for the Aria model.

    This layer implements the MoE mechanism, which routes input tokens to different experts
    based on a routing algorithm, processes them through the experts, and then combines
    the outputs.

    Args:
        config (AriaTextConfig): Configuration object for the MoE layer.
    """

    def __init__(self, config: AriaTextConfig):
        super().__init__()

        self.router = AriaTopKRouter(config)
        self.experts = AriaGroupedMLP(config)
        self.shared_experts = AriaSharedExpertsMLP(config)
        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE Layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the MoE layer.

        Process:
        1. Route tokens to experts using the router.
        2. Permute tokens based on routing decisions.
        3. Process tokens through experts.
        4. Unpermute and combine expert outputs.
        5. Add shared expert output to the final result.
        """
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        scores, indices, tokens_per_expert = self.router(hidden_states)

        # Token permutation
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices)
        permuted_tokens = hidden_states.index_select(0, sorted_indices // self.config.moe_topk)

        # Process through experts
        expert_output = self.experts(permuted_tokens, tokens_per_expert)

        # Token unpermutation
        unpermuted_tokens = torch.zeros(
            (scores.shape[0] * self.config.moe_topk, expert_output.size(1)),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        unpermuted_tokens.index_copy_(0, sorted_indices, expert_output)
        unpermuted_tokens = unpermuted_tokens.view(-1, self.config.moe_topk, expert_output.size(1))

        output = (unpermuted_tokens * scores.unsqueeze(-1)).sum(dim=1).view(original_shape)

        # Add shared expert output
        shared_expert_output = self.shared_experts(hidden_states.view(original_shape))
        return output + shared_expert_output


class AriaDecoderLayer(LlamaDecoderLayer):
    """
    Custom Decoder Layer for the Aria model which modifies the standard `LlamaDecoderLayer` by
    replacing the traditional MLP with a Mixture of Experts (MoE) Layer.

    Args:
        config (LlamaConfig): Configuration object for the layer.
        layer_idx (int): Index of the current layer in the model.
    """

    def __init__(self, config: AriaTextConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = AriaTextMoELayer(config)
        self.input_layernorm = AriaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AriaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class AriaTextPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, AriaGroupedGEMM):
            module.weight.data.normal_(mean=0.0, std=std)


class AriaTextModel(LlamaModel, AriaTextPreTrainedModel):
    def __init__(self, config: AriaTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AriaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.post_init()


class AriaForCausalLM(AriaPreTrainedModel, LlamaForCausalLM):
    """
    Aria model for causal language modeling tasks.

    This class extends LlamaForCausalLM to incorporate the Mixture of Experts (MoE) approach,
    allowing for more efficient and scalable language modeling.

    Args:
        config (AriaTextConfig): Configuration object for the model.
    """

    _tied_weights_keys = ["lm_head.weight"]
    config_class = AriaTextConfig
    _no_split_modules = ["AriaDecoderLayer"]

    def __init__(self, config: AriaTextConfig):
        super().__init__(config)
        self.model = AriaTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class AriaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class AriaForConditionalGeneration(AriaPreTrainedModel, GenerationMixin):
    """
    Aria model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language model
    to perform tasks that involve both image and text inputs.

    Args:
        config (AriaConfig): Configuration object for the model.
    """

    _supports_sdpa = False

    def __init__(self, config: AriaConfig):
        super().__init__(config)

        self.vision_tower = AutoModel.from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.multi_modal_projector = AriaProjector(
            config
        )
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._use_flash_attention_2 = config.text_config._attn_implementation == "flash_attention_2"
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int,
    ):
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, AriaCausalLMOutputWithPast]:
        """
        Forward pass of the AriaForConditionalGeneration model.

        This method processes both text and image inputs, merges them if necessary,
        and generates output using the language model.

        Args:
            input_ids (torch.LongTensor, optional): Input token ids.
            pixel_values (torch.FloatTensor, optional): Pixel values of the images.
            pixel_mask (torch.LongTensor, optional): Mask for the pixel values.
            attention_mask (torch.Tensor, optional): Attention mask.
            position_ids (torch.LongTensor, optional): Position ids.
            past_key_values (List[torch.FloatTensor], optional): Past key values for efficient processing.
            inputs_embeds (torch.FloatTensor, optional): Input embeddings.
            labels (torch.LongTensor, optional): Labels for computing the language modeling loss.
            use_cache (bool, optional): Whether to use the model's cache mechanism.
            output_attentions (bool, optional): Whether to output attention weights.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a ModelOutput object.
            num_logits_to_keep (`int`, optional): Calculate logits for the last `num_logits_to_keep` tokens, or all `input_ids` if `0`.

        Returns:
            Union[Tuple, AriaCausalLMOutputWithPast]: Model outputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_feature_layer = -1

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                )
                n_image_tokens = (input_ids == self.config.image_token_index).sum(dim=-1)[0].item()
                n_image_features = image_features.shape[1]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                special_image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            # In case input_ids.shape[1] == 1 & pixel_values != None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors
                # such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, config=self.config)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AriaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
