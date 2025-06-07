import numbers
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as tvf

from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...activations import ACT2FN
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import to_numpy_array
from ...modeling_outputs import BaseModelOutput, dataclass
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, auto_docstring, logging
from .configuration_vjepa2 import VJEPA2Config, VJEPA2PredictorConfig

logger = logging.get_logger(__name__)

## Utility functions


class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def resize_clip(clip, size, interpolation="bilinear"):
    if isinstance(clip[0], np.ndarray) or isinstance(clip[0], torch.Tensor):
        if isinstance(size, numbers.Number):
            if clip[0].shape[-1] == 3:
                im_h, im_w, im_c = clip[0].shape
            else:
                im_c, im_h, im_w = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]

        if isinstance(clip[0], np.ndarray):
            if interpolation == "bilinear":
                np_inter = cv2.INTER_LINEAR
            else:
                np_inter = cv2.INTER_NEAREST
            scaled = [cv2.resize(img, size, interpolation=np_inter) for img in clip]
        else:  # isinstance(clip[0], torch.Tensor)
            if interpolation == "bilinear":
                np_inter = tvf.InterpolationMode.BILINEAR
            else:
                np_inter = tvf.InterpolationMode.NEAREST
            size = (
                size[1],
                size[0],
            )  # torchvision transformers expect the size in (h, w) order.
            scaled = [tvf.resize(img, size, interpolation=np_inter) for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == "bilinear":
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image or torch.Tensor"
            + "but got list of {0}".format(type(clip[0]))
        )
    return scaled


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation="nearest"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = resize_clip(clip, self.size, interpolation=self.interpolation)
        return resized


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray) or isinstance(clip[0], torch.Tensor):
        if clip[0].shape[-1] == 3:
            cropped = [img[min_h : min_h + h, min_w : min_w + w, :] for img in clip]
        else:
            cropped = [img[:, min_h : min_h + h, min_w : min_w + w] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]

    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image or torch.Tensor):"
            + "but got list of {0}".format(type(clip[0]))
        )
    return cropped


class CenterCrop(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray) or isinstance(clip[0], torch.Tensor):
            if clip[0].shape[-1] == 3:
                im_h, im_w, im_c = clip[0].shape
            else:
                im_c, im_h, im_w = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image or torch.Tensor"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.0))
        y1 = int(round((im_h - h) / 2.0))
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped


def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format"""
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        elif isinstance(clip[0], torch.Tensor):
            tensor_clip = torch.stack(clip)
            # Converting (T, C, H, W) -> (C, T, H, W) to match what `convert_img` followed by
            # `np_clip[:, img_idx, :, :] = img` does for other data types.
            tensor_clip = tensor_clip.permute(1, 0, 2, 3)
            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = torch.div(tensor_clip, 255)
            return tensor_clip
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image or torch.Tensor\
            but got list of {0}".format(
                    type(clip[0])
                )
            )

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image\
                but got list of {0}".format(
                        type(clip[0])
                    )
                )
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img

        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255.0
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = torch.div(tensor_clip, 255)
            return tensor_clip


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def fn_normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError("tensor is not a torch clip.")

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip


class Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor clip.
        """
        return fn_normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_3d_sincos_pos_embed(
    embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(
        grid_w, grid_h
    )  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


## Image processor (transforms)


class VJEPA2ImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        crop_size: int = 224,
        normalize=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        **kwargs,
    ):
        self.short_side_size = crop_size
        self.size = [crop_size]
        self.crop_size = crop_size
        self.crop_size_dict = {"height": crop_size, "width": crop_size}
        self.normalize = normalize

    def _transform(self, images):
        return Compose(
            [
                Resize(self.short_side_size, interpolation="bilinear"),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=self.normalize[0], std=self.normalize[1]),
            ]
        )(images)

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = to_numpy_array(images)  # H x W x 3
            images = np.expand_dims(images, axis=0)  # T=1 x H x W x C
        else:
            # to adapt video data
            images = np.concatenate(
                [np.expand_dims(to_numpy_array(image), axis=0) for image in images],
                axis=0,
            )  # T x H x W x C

        images = self._transform(images)  # C x T x H x W, where T=1 for image
        images = (
            images.permute(1, 0, 2, 3).unsqueeze(0).numpy()
        )  # add batch, B x T x C x H x W
        images = list(images)
        data = {"pixel_values": images}  # list of T x C x H x W

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.preprocess(*args, **kwds)


## Modules


class VJEPA2PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            config.in_chans,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    @staticmethod
    def num_patches(config):
        return (config.crop_size // config.patch_size) * (
            config.crop_size // config.patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VJEPA2PatchEmbeddings3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        config: VJEPA2Config,
    ):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size

        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=config.hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    @staticmethod
    def num_patches(config):
        return (
            (config.frames_per_clip // config.tubelet_size)
            * (config.crop_size // config.patch_size)
            * (config.crop_size // config.patch_size)
        )

    def forward(self, x, **kwargs):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VJEPA2Embeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings.
    """

    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__()

        self.config = config
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(config)

        self.num_patches = self.patch_embeddings.num_patches
        if not self.config.use_rope:
            # position embeddings
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_patches(config), config.hidden_size),
                requires_grad=False,
            )
            self._init_pos_embed(self.position_embeddings.data)  # sincos pos-embed

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def _init_pos_embed(self, pos_embed):
        grid_size = self.config.crop_size // self.config.patch_size
        grid_depth = self.config.frames_per_clip // self.config.tubelet_size
        sincos = get_3d_sincos_pos_embed(
            self.config.hidden_size,
            grid_size,
            grid_depth,
            cls_token=False,
            uniform_power=self.config.uniform_power,
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def interpolate_pos_encoding(self, x, pos_embed):
        _, N, dim = pos_embed.shape

        # If pos_embed already corret size, just return
        _, _, T, H, W = x.shape
        if (
            H == self.config.img_height
            and W == self.config.img_width
            and T == self.config.frames_per_clip
        ):
            return pos_embed

        # Convert depth, height, width of input to be measured in patches
        # instead of pixels/frames
        T = T // self.config.tubelet_size
        H = H // self.patch_size
        W = W // self.patch_size

        # Compute the initialized shape of the positional embedding measured
        # in patches
        N_t = self.config.frames_per_clip // self.config.tubelet_size
        N_h = self.config.img_height // self.patch_size
        N_w = self.config.img_width // self.patch_size

        # Compute scale factor for spatio-temporal interpolation
        scale_factor = (T / N_t, H / N_h, W / N_w)

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
            scale_factor=scale_factor,
            mode="trilinear",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, c, t, height, width = pixel_values.shape
        if t < self.config.tubelet_size:
            # For some cases, if the input vision (image/video) consists of num_frames < tubelet_size, then embedding lookup fails. In these cases, we duplicate the frames.
            pixel_values = pixel_values.repeat(1, 1, self.config.tubelet_size, 1, 1)

        target_dtype = self.patch_embeddings.proj.weight.dtype

        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # add positional encoding to each token
        # ignore if we are using Rope
        if not self.config.use_rope:
            embeddings = embeddings + self.interpolate_pos_encoding(
                pixel_values, self.position_embeddings
            )

        return embeddings


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Dinov2->VJEPA
class VJEPA2SelfAttention(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {(config.hidden_size,)} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.proj(context_layer.view(new_context_layer_shape))

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()

    # similar to inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # they are computing this every time. instead HF style is to compute the inv_freq once and store it
    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class VJEPA2RopeSelfAttention(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {(config.hidden_size,)} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        self.grid_size = self.config.crop_size // self.config.patch_size
        self.grid_depth = self.config.frames_per_clip // self.config.tubelet_size

        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(self.grid_size * self.grid_size)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        # Remove frame component from ids
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        # --
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, x, masks=None):
        device = x.device
        token_size = x.size(1)

        # Note: when masks is none, we use a 1d id instead of Bxnum_attention_heads mask, as 1d vector is broadcasted to the correct shapes.
        if masks is not None:
            ids = masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        else:
            ids = torch.arange(token_size, device=device)
        # change to allow for extrapolation
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        # --
        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk, pos_ids):
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        # Combine rotated dimension
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        else:
            qk = torch.cat([qkd, qkh, qkw], dim=-1)
        return qk

    def forward(
        self,
        hidden_states,
        position_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.proj(context_layer.view(new_context_layer_shape))

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Dinov2->VJEPA
class VJEPA2Attention(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__()
        self.attention = (
            VJEPA2RopeSelfAttention(config)
            if config.use_rope
            else VJEPA2SelfAttention(config)
        )
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attention.num_attention_heads,
            self.attention.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(
            heads
        )
        self.attention.all_head_size = (
            self.attention.attention_head_size * self.attention.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(
            hidden_states, position_mask, output_attentions, head_mask
        )

        attention_output = self_outputs[0]

        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# Copied from transformers.models.beit.modeling_dinov2.drop_path
def drop_path(
    input: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (
        input.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device
    )
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class VJEPA2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class VJEPA2MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class VJEPA2SwiGLUFFN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        if config.wide_SiLU:
            hidden_features = int(config.hidden_size * config.mlp_ratio)
            hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        else:
            hidden_features = int(config.hidden_size * config.mlp_ratio)

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


class VJEPA2Layer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self,
        config: Union[VJEPA2Config, VJEPA2PredictorConfig],
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA2Attention(config)

        self.drop_path = (
            VJEPA2DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_SiLU:
            self.mlp = VJEPA2SwiGLUFFN(config)
        else:
            self.mlp = VJEPA2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(
                hidden_states
            ),  # in Dinov2, layernorm is applied before self-attention
            position_mask=position_mask,  # position mask for context/target selection
            head_mask=head_mask,  # head mask is applied at F.scaled_dot_product_attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        # layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->Dinov2->VJEPA
class VJEPA2Encoder(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = VJEPA2Embeddings(config)
        dpr = [
            (
                config.drop_path_rate * i / (config.num_hidden_layers - 1)
                if config.num_hidden_layers > 1
                else 0.0
            )
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [VJEPA2Layer(config, dpr[i]) for i in range(config.num_hidden_layers)]
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = self.embeddings(pixel_values)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,  # Note: passing none position mask for now
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, None, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def apply_masks(x, masks, concat=True) -> torch.Tensor:
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat) -> torch.Tensor:
    N = len(x) // B
    x = torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )
    return x


class VJEPA2PredictorEmbeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings.
    """

    def __init__(self, config: VJEPA2PredictorConfig) -> None:
        super().__init__()

        # self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.config = config
        self.predictor_embeddings = nn.Linear(
            config.enc_hidden_size, config.hidden_size
        )
        if not self.config.use_rope:
            # position embeddings
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_patches(config), config.hidden_size),
                requires_grad=False,
            )
            self._init_pos_embed(self.position_embeddings.data)  # sincos pos-embed
        self.num_mask_tokens = 0
        self.zero_init_mask_tokens = config.zero_init_mask_tokens
        if config.use_mask_tokens:
            self.num_mask_tokens = config.num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, 1, config.hidden_size))
                    for i in range(self.num_mask_tokens)
                ]
            )

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    @staticmethod
    def num_patches(config):
        if config.frames_per_clip > 1:
            return (
                (config.frames_per_clip // config.tubelet_size)
                * (config.crop_size // config.patch_size)
                * (config.crop_size // config.patch_size)
            )
        else:
            return (config.crop_size // config.patch_size) * (
                config.crop_size // config.patch_size
            )

    def _init_pos_embed(self, pos_embed):
        grid_size = self.config.crop_size // self.config.patch_size
        grid_depth = self.config.frames_per_clip // self.config.tubelet_size
        sincos = get_3d_sincos_pos_embed(
            self.config.hidden_size,
            grid_size,
            grid_depth,
            cls_token=False,
            uniform_power=self.config.uniform_power,
        )

        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: List[torch.Tensor],
        target_mask: List[torch.Tensor],
        mask_index: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states : encoder outputs (context)
        context_mask: tokens of the context (outputs from the encoder)
        target_mask: tokens to predict
        mask_index: index of the target mask to choose (useful for multiclip?)
        """

        B = hidden_states.size(0)
        context = self.predictor_embeddings(hidden_states)

        # add positional encoding to each token
        # ignore if we are using Rope
        if not self.config.use_rope:
            pos_embed = self.position_embeddings.repeat(B, 1, 1)
            context = context + apply_masks(pos_embed, context_mask)

        # Make target tokens
        mask_index = mask_index % self.num_mask_tokens
        target = self.mask_tokens[mask_index]

        # Note: this is problematic if the config isn't initialized with the right frames_per_clip value, eg for scenarios if we want to run predictor for more tokens than in the config.
        # target = target.repeat(B, self.num_patches(self.config), 1)
        # Remedy: use the provided target mask to get the max patch num
        max_patch_num = (
            target_mask[0].max().item() + 1
        )  # one extra to include the last patch
        target = target.repeat(B, max_patch_num, 1)
        target = apply_masks(target, target_mask)

        if not self.config.use_rope:
            pos_embs = self.position_embeddings.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, target_mask)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(context_mask))
            target += pos_embs

        # Concatenate context & target tokens
        context = context.repeat(len(context_mask), 1, 1)
        embeddings = torch.cat([context, target], dim=1)

        # Positions of context & target tokens
        cm = torch.cat(context_mask, dim=0)
        tm = torch.cat(target_mask, dim=0)
        masks = torch.cat([cm, tm], dim=1)

        return embeddings, masks


class VJEPA2Predictor(nn.Module):
    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__()
        config: VJEPA2PredictorConfig = config.get_predictor_config()
        self.config = config
        self.embeddings = VJEPA2PredictorEmbeddings(config)
        dpr = [
            (
                config.drop_path_rate * i / (config.num_hidden_layers - 1)
                if config.num_hidden_layers > 1
                else 0.0
            )
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [VJEPA2Layer(config, dpr[i]) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.argsort: Any = None
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.enc_hidden_size, bias=True)

    def sort_tokens(self, hidden_states, position_masks, argsort, head_mask=None):
        position_masks = torch.gather(position_masks, dim=1, index=argsort)
        hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )
        if head_mask is not None and head_mask[0] is not None:
            head_mask = head_mask.permute(1, 0, 2, 3, 4)
            argsort_4d = (
                argsort.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, head_mask.size(1), head_mask.size(2), -1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, -1, head_mask.size(-1))
            )
            head_mask = torch.gather(head_mask, dim=3, index=argsort_4d)
            argsort_5d = (
                argsort.unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, head_mask.size(1), head_mask.size(2), head_mask.size(3), -1)
            )
            head_mask = torch.gather(head_mask, dim=4, index=argsort_5d)
            head_mask = head_mask.permute(1, 0, 2, 3, 4)
        return hidden_states, position_masks, head_mask

    def unsort_tokens(self, hidden_states, argsort):
        reverse_argsort = torch.argsort(argsort, dim=1)
        hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=reverse_argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )
        return hidden_states

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: List[torch.Tensor],
        target_mask: List[torch.Tensor],
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # mask out the encoder hidden states
        # this is implemented here as in VJEPA training a separate encoder is used for target
        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        _, N_ctxt, D = encoder_hidden_states.shape
        hidden_states, position_masks = self.embeddings(
            encoder_hidden_states, context_mask, target_mask
        )

        # Put tokens in sorted order
        argsort = torch.argsort(position_masks, dim=1)  # [B, N]
        hidden_states, position_masks, head_mask = self.sort_tokens(
            hidden_states, position_masks, argsort, head_mask
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    position_masks,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, position_masks, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.layernorm(hidden_states)
        # unsort and extract the predicted tokens
        hidden_states = self.unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, N_ctxt:]
        # projection
        hidden_states = self.proj(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class VJEPA2PreTrainedModel(PreTrainedModel):
    config_class = VJEPA2Config
    base_model_prefix = "vjepa"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VJEPA2SwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(
        self,
        module: Union[
            nn.Linear,
            nn.Conv2d,
            nn.LayerNorm,
            VJEPA2Embeddings,
            VJEPA2PredictorEmbeddings,
        ],
    ) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VJEPA2Embeddings):
            if not module.config.use_rope:
                module.position_embeddings.data = nn.init.trunc_normal_(
                    module.position_embeddings.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.position_embeddings.dtype)
        elif isinstance(module, VJEPA2PredictorEmbeddings):
            if not module.config.use_rope:
                module.position_embeddings.data = nn.init.trunc_normal_(
                    module.position_embeddings.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.position_embeddings.dtype)
            if not module.zero_init_mask_tokens:
                module.mask_token = nn.init.trunc_normal_(
                    module.mask_token.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.mask_token.dtype)


@dataclass
class VJEPA2PredictorOutputWithMaskedInput(ModelOutput):
    """
    VJEPA Predictor outputs that also contains the masked encoder outputs

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        masked_hidden_state (`torch.FloatTensor`), *optional*, returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        target_hidden_state (`torch.FloatTensor`), *optional*, returned when `target_mask` is provided which is applied on VJEPA2Encoder outputs
    """

    last_hidden_state: torch.FloatTensor = None
    masked_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class VJEPA2OutputWithMaskedInput(ModelOutput):
    """
    VJEPA outputs that also contains the masked encoder outputs
    Optionally contains the predictor outputs

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        masked_hidden_state (`torch.FloatTensor`), *optional*, returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        predictor_output (`VJEPA2PredictorOutputWithMaskedInput`), *optional* - returns the output from the predictor module
    """

    last_hidden_state: torch.FloatTensor = None
    masked_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    predictor_output: Optional[VJEPA2PredictorOutputWithMaskedInput] = None


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    """
    Inputs:
        - head_mask: bsz x seq_length x seq_length | None
    Returns
        - [num_hidden_layers x batch x num_heads x seq_length x seq_length] | [num_hidden_layers]
    """
    if head_mask is not None:
        head_mask = head_mask.unsqueeze(1).unsqueeze(0)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        # head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


@auto_docstring
class VJEPA2Model(VJEPA2PreTrainedModel):
    def __init__(self, config: VJEPA2Config):
        super().__init__(config)
        self.config = config

        self.encoder = VJEPA2Encoder(config)
        self.predictor = VJEPA2Predictor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(
        self,
    ) -> Union[VJEPA2PatchEmbeddings, VJEPA2PatchEmbeddings3D]:
        return self.encoder.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        context_head_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[List[torch.Tensor]] = None,
        target_head_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[List[torch.Tensor]] = None,
        skip_predictor: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VJEPA2OutputWithMaskedInput]:
        r"""
        context_head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the context.
        target_head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the target.
        context_mask (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*):
            The mask position ids indicating which encoder output patches are going to be exposed to the predictor. By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating full context available to the predictor.
        target_mask (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*):
            The mask position ids indicating which encoder output patches are going to be used as a prediction target for the predictor. By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating that the predictor should predict all encoder patches.
        skip_predictor (bool):
            flag to skip the predictor forward, useful if you just need the encoder outputs
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        context_head_mask = _convert_head_mask_to_5d(
            context_head_mask, self.config.num_hidden_layers
        )
        target_head_mask = _convert_head_mask_to_5d(
            target_head_mask, self.config.pred_num_hidden_layers
        )

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            head_mask=context_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]  # hidden_states

        if context_mask is None and target_mask is None:
            B = pixel_values.size(0)
            N = sequence_output.size(1)  # ensure we are using dynamic patch size
            context_mask = [
                torch.arange(N, device=pixel_values.device).unsqueeze(0).repeat((B, 1))
            ]
            target_mask = [
                torch.arange(N, device=pixel_values.device).unsqueeze(0).repeat((B, 1))
            ]

        if not skip_predictor:
            predictor_outputs = self.predictor(
                encoder_hidden_states=sequence_output,
                context_mask=context_mask,
                target_mask=target_mask,
                head_mask=target_head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            predictor_outputs = [None, None]
            predictor_output = None

        if not return_dict:
            enc_head_outputs = (
                sequence_output,
                apply_masks(sequence_output, context_mask),
            )
            pred_head_outputs = (
                predictor_outputs[0],
                apply_masks(sequence_output, target_mask),
            )
            pred_out = pred_head_outputs + predictor_outputs[1:]
            enc_out = enc_head_outputs + encoder_outputs[1:] + pred_out
            return enc_out

        encoder_output = VJEPA2OutputWithMaskedInput(
            last_hidden_state=sequence_output,
            masked_hidden_state=apply_masks(sequence_output, context_mask),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        predictor_output = VJEPA2PredictorOutputWithMaskedInput(
            last_hidden_state=predictor_outputs[0],
            target_hidden_state=apply_masks(sequence_output, target_mask),
            hidden_states=predictor_outputs.hidden_states,
            attentions=predictor_outputs.attentions,
        )
        encoder_output.predictor_output = predictor_output

        return encoder_output

    def get_vision_features(self, pixel_values) -> torch.Tensor:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
        encoder_outputs, _ = self.forward(pixel_values)
        return encoder_outputs.last_hidden_state


__all__ = ["VJEPA2Model", "VJEPA2PreTrainedModel", "VJEPA2ImageProcessor"]
