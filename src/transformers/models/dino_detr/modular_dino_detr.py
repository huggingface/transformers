import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

from torch.nn import MultiheadAttention

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.detr.image_processing_detr import center_to_corners_format
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

from ...activations import ACT2FN
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    is_scipy_available,
    is_timm_available,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)
from ...utils.generic import OutputRecorder, can_return_tuple, check_model_inputs
from ..deformable_detr.modeling_deformable_detr import (
    DeformableDetrConvEncoder,
    DeformableDetrEncoderLayer,
    DeformableDetrMultiscaleDeformableAttention,
)
from ..detr.image_processing_detr import DetrImageProcessor
from ..detr.modeling_detr import DetrConvModel, DetrFrozenBatchNorm2d, DetrMLPPredictionHead
from ..rt_detr.modeling_rt_detr import get_contrastive_denoising_training_group
from .configuration_dino_detr import DinoDetrConfig


logger = logging.get_logger(__name__)


class DinoDetrEncoderLayer(DeformableDetrEncoderLayer):
    pass


class DinoDetrFrozenBatchNorm2d(DetrFrozenBatchNorm2d):
    pass


class DinoDetrConvEncoder(DeformableDetrConvEncoder):
    pass


class DinoDetrConvModel(DetrConvModel):
    pass


class DinoDetrMLPPredictionHead(DetrMLPPredictionHead):
    pass


class DinoDetrMultiscaleDeformableAttention(DeformableDetrMultiscaleDeformableAttention):
    pass


if is_torch_available():
    import torch
    import torch.nn.functional as F
    from torch import nn

if is_torchvision_available():
    pass

if is_vision_available():
    pass

if is_scipy_available():
    pass

if is_timm_available():
    pass


_CONFIG_FOR_DOC = "DinoDetrConfig"
_CHECKPOINT_FOR_DOC = "kostaspitas/dino_detr"


@dataclass
class DinoDetrEncoderOutput(ModelOutput):
    """
    Base class for outputs of the DinoDetrEncoder. This class adds attributes specific to the encoder output.

    Args:
        output (`torch.FloatTensor`):
            Final output tensor of the encoder.
        intermediate_output (`torch.FloatTensor`, *optional*):
            Stacked intermediate hidden states (output of each layer of the encoder).
        intermediate_ref (`torch.FloatTensor`, *optional*):
            Stacked intermediate reference points (reference points of each layer of the encoder).
    """

    output: torch.FloatTensor
    intermediate_output: Optional[torch.FloatTensor] = None
    intermediate_ref: Optional[torch.FloatTensor] = None
    encoder_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class DinoDetrDecoderOutput(ModelOutput):
    """
    Base class for outputs of the DinoDetrDecoder. This class adds two attributes specific to the decoder output.

    Args:
        intermediate (`List[torch.FloatTensor]`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        ref_points (`Optional[List[torch.FloatTensor]]`, *optional*):
            Stacked intermediate reference points (reference points of each layer of the decoder).
    """

    intermediate: list[torch.FloatTensor]
    ref_points: Optional[list[torch.FloatTensor]] = None


@dataclass
class DinoDetrEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of the DinoDetrEncoderDecoder. This class adds attributes specific to the encoder-decoder output.

    Args:
        hidden_states (`torch.FloatTensor`):
            Final hidden states of the decoder.
        reference_points (`Optional[torch.FloatTensor]`, *optional*):
            Final reference points of the decoder.
        hidden_states_encoder (`Optional[torch.FloatTensor]`, *optional*):
            Final hidden states of the encoder.
        reference_points_encoder (`Optional[torch.FloatTensor]`, *optional*):
            Final reference points of the encoder.
    """

    hidden_states: torch.FloatTensor
    reference_points: Optional[torch.FloatTensor] = None
    hidden_states_encoder: Optional[torch.FloatTensor] = None
    reference_points_encoder: Optional[torch.FloatTensor] = None


@dataclass
class DinoDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the Dino DETR encoder-decoder model.

    Args:
        last_hidden_state (`torch.FloatTensor`):
            Sequence of hidden states at the output of the last layer of the decoder of the model.
        hidden_states (`Optional[list[torch.FloatTensor]]`, *optional*):
            List of hidden states at the output of each decoder layer.
        references (`Optional[list[torch.FloatTensor]]`, *optional*):
            List of reference points at the output of each decoder layer.
        encoder_last_hidden_state (`Optional[torch.FloatTensor]`, *optional*):
            Sequence of hidden states at the output of the last layer of the encoder of the model.
        encoder_reference (`Optional[torch.FloatTensor]`, *optional*):
            Reference points at the output of the encoder.
        denoising_meta (`Optional[dict]`, *optional*):
            Metadata related to denoising tasks.
        encoder_hidden_states (`Optional[Tuple[torch.FloatTensor]]`, *optional*):
            Tuple of hidden states of the encoder at the output of each layer.
        decoder_hidden_states (`Optional[Tuple[torch.FloatTensor]]`, *optional*):
            Tuple of hidden states of the decoder at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor
    hidden_states_model: Optional[list[torch.FloatTensor]] = None
    references: Optional[list[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_reference: Optional[torch.FloatTensor] = None
    denoising_meta: Optional[dict] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class DinoDetrObjectDetectionOutput(ModelOutput):
    """
    Output class for `DinoDetrForObjectDetection`.

    Args:
        last_hidden_state (`torch.FloatTensor`):
            Sequence of hidden states at the output of the last layer of the decoder of the model.
        reference (`Optional[torch.FloatTensor]`, *optional*):
            Final reference points of the decoder.
        encoder_last_hidden_state (`Optional[torch.FloatTensor]`, *optional*):
            Sequence of hidden states at the output of the last layer of the encoder of the model.
        encoder_reference (`Optional[torch.FloatTensor]`, *optional*):
            Final reference points of the encoder.
        loss (`Optional[torch.FloatTensor]`, *optional*):
            Total loss as a linear combination of a negative log-likelihood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Optional[Dict]`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`Optional[torch.FloatTensor]`, *optional*):
            Classification logits (including no-object) for all queries.
        pred_boxes (`Optional[torch.FloatTensor]`, *optional*):
            Normalized box coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding).
        auxiliary_outputs (`Optional[List[Dict]]`, *optional*):
            Optional, only returned when auxiliary losses are activated (i.e., `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the keys `logits` and `pred_boxes` for
            each decoder layer.
        denoising_meta (`Optional[dict]`, *optional*):
            Metadata related to denoising tasks.
        encoder_hidden_states (`Optional[Tuple[torch.FloatTensor]]`, *optional*):
            Tuple of hidden states of the encoder at the output of each layer.
        decoder_hidden_states (`Optional[Tuple[torch.FloatTensor]]`, *optional*):
            Tuple of hidden states of the decoder at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor
    reference: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_reference: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    auxiliary_outputs: Optional[list[dict]] = None
    denoising_meta: Optional[dict] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None


def _get_clones(module: torch.nn.Module, N: int, layer_share: bool = False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x: torch.FloatTensor, eps: float = 1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def denoising_post_process(
    outputs_class: torch.FloatTensor,
    outputs_coord: torch.FloatTensor,
    denoising_meta: dict,
    aux_loss: bool,
    _set_aux_loss: Callable,
):
    """
    Post-processes the outputs of the denoising task in the Conditional Denoising (Cdenoising) framework.

    This function separates the known (denoising) outputs from the unknown outputs and updates the metadata with
    the processed known outputs. It also handles auxiliary losses if required.

    Args:
        outputs_class (`torch.FloatTensor`):
            Classification logits of shape `(batch_size, num_queries, num_classes)`.
        outputs_coord (`torch.FloatTensor`):
            Predicted bounding box coordinates of shape `(batch_size, num_queries, 4)`.
        denoising_meta (`Dict`):
            Metadata dictionary containing information about the denoising task. Must include the key `"pad_size"`,
            which specifies the number of known (denoising) queries.
        aux_loss (`bool`):
            Whether to compute auxiliary losses for intermediate layers.
        _set_aux_loss (`Callable`):
            A callable function to compute auxiliary losses.

    Returns:
        `Tuple[torch.FloatTensor, torch.FloatTensor]`:
            A tuple containing:
            - `outputs_class` (`torch.FloatTensor`): Classification logits after removing the known queries.
            - `outputs_coord` (`torch.FloatTensor`): Bounding box coordinates after removing the known queries.
    """
    if denoising_meta and denoising_meta["dn_num_split"][0] > 0:
        output_known_class = outputs_class[:, :, : denoising_meta["dn_num_split"][0], :]
        output_known_coord = outputs_coord[:, :, : denoising_meta["dn_num_split"][0], :]
        outputs_class = outputs_class[:, :, denoising_meta["dn_num_split"][0] :, :]
        outputs_coord = outputs_coord[:, :, denoising_meta["dn_num_split"][0] :, :]
        out = {
            "logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        denoising_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord


class DinoDetrPositionEmbeddingSineHW(nn.Module):
    """
    This module implements a sine-based positional embedding for 2D images, inspired by the "Attention is All You Need" paper.

    The positional embedding is generalized to work on images by computing separate embeddings for the height and width dimensions.
    It supports optional normalization and scaling of the positional values.

    Args:
        num_pos_feats (`int`, *optional*, defaults to 64):
            The number of positional features (dimensions) for each spatial axis (height and width).
        temperatureH (`int`, *optional*, defaults to 10000):
            The temperature parameter for the height dimension, used to scale the sine and cosine functions.
        temperatureW (`int`, *optional*, defaults to 10000):
            The temperature parameter for the width dimension, used to scale the sine and cosine functions.
        normalize (`bool`, *optional*, defaults to `False`):
            Whether to normalize the positional values to the range [0, scale].
        scale (`float`, *optional*, defaults to `2 * math.pi`):
            The scale factor applied to the normalized positional values. Must be provided if `normalize` is `True`.

    Raises:
        `ValueError`: If `scale` is provided but `normalize` is set to `False`.

    Inputs:
        pixel_values (`torch.FloatTensor`):
            A tensor of shape `(batch_size, channels, height, width)` representing the input image.
        pixel_mask (`torch.LongTensor`):
            A binary mask of shape `(batch_size, height, width)` where 1 indicates valid pixels and 0 indicates padding.

    Returns:
        `torch.FloatTensor`: A tensor of shape `(batch_size, 2 * num_pos_feats, height, width)` containing the positional embeddings.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperatureH: int = 10000,
        temperatureW: int = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: torch.LongTensor):
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pixel_values.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pixel_values.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        position_embeddings = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return position_embeddings


class DinoDetrLearnedPositionEmbedding(nn.Module):
    """
    This module implements a learned absolute positional embedding for 2D images.

    The positional embeddings are learned independently for the height and width dimensions using embedding layers.
    These embeddings are combined to form the final positional encoding for the input image.

    Args:
        num_pos_feats (`int`, *optional*, defaults to 256):
            The number of positional features (dimensions) for each spatial axis (height and width).

    Inputs:
        pixel_values (`torch.FloatTensor`):
            A tensor of shape `(batch_size, channels, height, width)` representing the input image.
        pixel_mask (`torch.LongTensor`):
            A binary mask of shape `(batch_size, height, width)` where 1 indicates valid pixels and 0 indicates padding.
            **Note**: This argument is currently not used in the forward method.

    Returns:
        `torch.FloatTensor`: A tensor of shape `(batch_size, 2 * num_pos_feats, height, width)` containing the positional embeddings.
    """

    def __init__(self, num_pos_feats: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: torch.LongTensor):
        height, width = pixel_values.shape[-2:]
        width_indices = torch.arange(width, device=pixel_values.device)
        height_indices = torch.arange(height, device=pixel_values.device)
        x_embeddings = self.col_embed(width_indices)
        y_embeddings = self.row_embed(height_indices)
        position_embeddings = (
            torch.cat(
                [
                    x_embeddings.unsqueeze(0).repeat(height, 1, 1),
                    y_embeddings.unsqueeze(1).repeat(1, width, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(pixel_values.shape[0], 1, 1, 1)
        )
        return position_embeddings


def build_position_encoding(config):
    """
    Builds the positional encoding module for the Dino DETR model.

    This function creates a positional encoding module based on the configuration provided. It supports both sine-based
    and learned positional encodings.

    Args:
        config (`PretrainedConfig`):
            The configuration object containing model parameters. Must include the following attributes:
            - `d_model` (int): The hidden size of the model.
            - `pe_temperature_H` (int, *optional*): The temperature parameter for the height dimension in sine-based embeddings.
            - `pe_temperature_W` (int, *optional*): The temperature parameter for the width dimension in sine-based embeddings.

    Returns:
        `nn.Module`: A positional encoding module.
    """
    N_steps = config.d_model // 2
    position_embeddings = DinoDetrPositionEmbeddingSineHW(
        N_steps,
        temperatureH=config.pe_temperature_H,
        temperatureW=config.pe_temperature_W,
        normalize=True,
    )

    return position_embeddings


def gen_sine_position_embeddings(reference_points: torch.FloatTensor, d_model: int):
    """
    Generates sine-based positional embeddings for the given reference points.

    This function computes sine and cosine positional embeddings for 2D or 4D reference points, which can be used
    in transformer models to encode spatial information. The embeddings are computed separately for each dimension
    and concatenated.

    Args:
        reference_points (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, 2)` for 2D reference points or `(batch_size, num_queries, 4)`
            for 4D reference points. The last dimension corresponds to (x, y) or (x, y, width, height).
        d_model (`int`):
            The dimensionality of the model. Must be an even number, as the sine and cosine embeddings are split
            equally across dimensions.

    Returns:
        `torch.FloatTensor`: A tensor of shape `(batch_size, num_queries, d_model)` containing the sine-based
        positional embeddings.

    Raises:
        `ValueError`: If the last dimension of `reference_points` is not 2 or 4.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(d_model / 2, dtype=torch.float32, device=reference_points.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model / 2))
    x_embed = reference_points[:, :, 0] * scale
    y_embed = reference_points[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    if reference_points.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif reference_points.size(-1) == 4:
        w_embed = reference_points[:, :, 2] * scale
        h_embed = reference_points[:, :, 3] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown reference_points shape(-1):{reference_points.size(-1)}")
    return pos


def generate_initial_queries(
    memory: torch.FloatTensor,
    memory_padding_mask: torch.LongTensor,
    spatial_shapes: torch.FloatTensor,
    learned_wh=torch.FloatTensor,
):
    """
    Generates output proposals and memory for the encoder in the Dino DETR model.

    This function computes proposals for bounding boxes based on the encoder's output memory and spatial shapes. It
    also applies padding masks and scales the proposals using learned width and height parameters.

    Args:
        memory (`torch.FloatTensor`):
            A tensor of shape `(batch_size, total_spatial_elements, d_model)` representing the encoder's output memory.
        memory_padding_mask (`torch.LongTensor`):
            A tensor of shape `(batch_size, total_spatial_elements)` indicating which spatial elements are padded.
        spatial_shapes (`torch.FloatTensor`):
            A tensor of shape `(num_levels, 2)` where each row contains the height and width of a feature map level.
        learned_wh (`torch.FloatTensor`):
            A tensor of shape `(2,)` representing the learned width and height parameters for scaling the proposals.

    Returns:
        `Tuple[torch.FloatTensor, torch.FloatTensor]`:
            - `queries_init` (`torch.FloatTensor`): A tensor of shape `(batch_size, total_spatial_elements, d_model)`
              containing the processed memory with padding and invalid proposals masked.
            - `query_reference_points_init` (`torch.FloatTensor`): A tensor of shape `(batch_size, total_spatial_elements, 4)`
              containing the bounding box proposals in the format `(center_x, center_y, width, height)`.

    Raises:
        `ValueError`: If the input tensors have incompatible shapes.
    """
    batch_size, _, _ = memory.shape
    proposals = []
    current_height_width_prod = 0
    for level, (height, width) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[
            :, current_height_width_prod : (current_height_width_prod + height * width)
        ].view(batch_size, height, width, 1)
        valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, dtype=torch.float32, device=memory.device),
            torch.linspace(0, width - 1, width, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # height, width, 2

        scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale

        if learned_wh is not None:
            wh = torch.ones_like(grid) * learned_wh.sigmoid() * (2.0**level)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**level)

        proposal = torch.cat((grid, wh), -1).view(batch_size, -1, 4)
        proposals.append(proposal)
        current_height_width_prod += height * width

    query_reference_points_init = torch.cat(proposals, 1)
    query_reference_points_init_valid = (
        (query_reference_points_init > 0.01) & (query_reference_points_init < 0.99)
    ).all(-1, keepdim=True)
    query_reference_points_init = torch.log(
        query_reference_points_init / (1 - query_reference_points_init)
    )  # unsigmoid
    query_reference_points_init = query_reference_points_init.masked_fill(
        memory_padding_mask.unsqueeze(-1), float("inf")
    )
    query_reference_points_init = query_reference_points_init.masked_fill(
        ~query_reference_points_init_valid, float("inf")
    )

    queries_init = memory
    queries_init = queries_init.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    queries_init = queries_init.masked_fill(~query_reference_points_init_valid, float(0))

    return queries_init, query_reference_points_init


class DinoDetrPreTrainedModel(PreTrainedModel):
    config_class = DinoDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DinoDetrConvEncoder", r"DinoDetrEncoderLayer", r"DinoDetrDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, DinoDetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, DinoDetrMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)


class DinoDetrDecoderLayer(nn.Module):
    """
    A single layer of the Dino DETR decoder.

    This layer consists of self-attention, cross-attention, and feed-forward submodules, with optional configurations
    for key-aware projections and different types of self-attention mechanisms.

    Args:
        config (`DinoDetrConfig`):
            The configuration object containing model parameters. Must include the following attributes:
            - `num_heads` (int): The number of attention heads.
            - `decoder_n_points` (int): The number of points for deformable attention.
            - `dropout` (float): The dropout probability.
            - `d_model` (int): The hidden size of the model.
            - `d_ffn` (int): The hidden size of the feed-forward network.
            - `activation` (str): The activation function to use in the feed-forward network.

    Inputs:
        queries (`torch.FloatTensor`):
            A tensor of shape `(num_queries, batch_size, d_model)` representing the input queries.
        query_position_embeddings (`torch.FloatTensor`):
            A tensor of shape `(num_queries, batch_size, d_model)` representing the positional embeddings for the queries.
        query_reference_points (`torch.FloatTensor`):
            A tensor of shape `(num_queries, batch_size, 4)` representing the reference points for the queries.
        memory (`torch.FloatTensor`):
            A tensor of shape `(memory_size, batch_size, d_model)` representing the encoder's output memory.
        memory_key_padding_mask (`torch.LongTensor`):
            A tensor of shape `(batch_size, memory_size)` indicating which memory elements are padded.
        memory_level_start_index (`torch.FloatTensor`):
            A tensor indicating the start index of each level in the memory.
        memory_spatial_shapes (`torch.FloatTensor`):
            A tensor of shape `(num_levels, 2)` where each row contains the height and width of a feature map level.
        memory_spatial_shapes_list (`List[torch.FloatTensor]`):
            A list of tensors representing the spatial shapes of each memory level.
        self_attn_mask (`Optional[torch.LongTensor]`, *optional*):
            A tensor of shape `(num_queries, num_queries)` representing the self-attention mask.

    Returns:
        `Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor]]]`:
            - `queries` (`torch.FloatTensor`): The updated queries after passing through the decoder layer.
    """

    def __init__(self, config: DinoDetrConfig):
        super().__init__()
        # Cross attention
        self.cross_attn = DinoDetrMultiscaleDeformableAttention(config, config.num_heads, config.decoder_n_points)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)

        # Self attention
        self.self_attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Fully Connected Layer
        self.linear1 = nn.Linear(config.d_model, config.d_ffn)
        self.activation = ACT2FN[config.activation]
        self.dropout3 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_ffn, config.d_model)
        self.dropout4 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.d_model)

        self.key_aware_proj = None

    @staticmethod
    def with_pos_embed(tensor: torch.FloatTensor, pos: torch.FloatTensor):
        return tensor if pos is None else tensor + pos

    # Fully connected
    def forward_ffn(self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]):
        transformed_values = self.linear2(self.dropout3(self.activation(self.linear1(pixel_values))))
        output = pixel_values + self.dropout4(transformed_values)
        output = self.norm3(output)
        return output

    # Self attention
    def forward_sa(
        self,
        queries: torch.FloatTensor,
        query_position_embeddings: torch.FloatTensor,
        self_attn_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        attn_weights = None
        q = k = self.with_pos_embed(queries, query_position_embeddings)
        transformed_queries, attn_weights = self.self_attn(q, k, queries, attn_mask=self_attn_mask)
        queries = queries + self.dropout2(transformed_queries)
        queries = self.norm2(queries)

        return queries, attn_weights

    # Cross Attention
    def forward_ca(
        self,
        queries: torch.FloatTensor,
        query_position_embeddings: torch.FloatTensor,
        query_reference_points: torch.FloatTensor,
        memory: torch.FloatTensor,
        memory_key_padding_mask: torch.LongTensor,
        memory_level_start_index: torch.FloatTensor,
        memory_spatial_shapes: torch.FloatTensor,
        memory_spatial_shapes_list: list[torch.FloatTensor],
        **kwargs: Unpack[TransformersKwargs],
    ):
        transformed_queries, attn_weights = self.cross_attn(
            hidden_states=self.with_pos_embed(queries, query_position_embeddings).transpose(0, 1),
            reference_points=query_reference_points.transpose(0, 1).contiguous(),
            encoder_hidden_states=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            spatial_shapes_list=memory_spatial_shapes_list,
            level_start_index=memory_level_start_index,
            encoder_attention_mask=memory_key_padding_mask,
        )
        transformed_queries = transformed_queries.transpose(0, 1)
        queries = queries + self.dropout1(transformed_queries)
        queries = self.norm1(queries)

        return queries, attn_weights

    def forward(
        self,
        queries: torch.FloatTensor,
        query_position_embeddings: torch.FloatTensor,
        query_reference_points: torch.FloatTensor,
        memory: torch.FloatTensor,
        memory_key_padding_mask: torch.LongTensor,
        memory_level_start_index: torch.FloatTensor,
        memory_spatial_shapes: torch.FloatTensor,
        memory_spatial_shapes_list: list[torch.FloatTensor],
        self_attn_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Self Attention
        queries, attn_weights = self.forward_sa(
            queries=queries,
            query_position_embeddings=query_position_embeddings,
            self_attn_mask=self_attn_mask,
            **kwargs,
        )

        # Deformable Cross Attention
        queries, attn_weights = self.forward_ca(
            queries=queries,
            query_position_embeddings=query_position_embeddings,
            query_reference_points=query_reference_points,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_level_start_index=memory_level_start_index,
            memory_spatial_shapes=memory_spatial_shapes,
            memory_spatial_shapes_list=memory_spatial_shapes_list,
            **kwargs,
        )

        # Fully Connected Layer
        queries = self.forward_ffn(queries, **kwargs)

        outputs = (queries,)

        return outputs


class DinoDetrEncoder(DinoDetrPreTrainedModel):
    """
    The encoder module for the Dino DETR model.

    This module processes input embeddings and positional embeddings through multiple encoder layers, applying
    self-attention and feed-forward networks. It supports optional dropout and two-stage decoding configurations.

    Args:
        encoder_layer (`DinoDetrEncoderLayer`):
            A single encoder layer to be cloned and stacked.
        norm (`torch.nn.Module`):
            A normalization layer applied to the encoder's output.
        config (`DinoDetrConfig`):
            The configuration object containing model parameters. Must include the following attributes:
            - `num_encoder_layers` (int): The number of encoder layers.
            - `num_queries` (int): The number of queries.
            - `d_model` (int): The hidden size of the model.

    Inputs:
        input_embeddings (`torch.FloatTensor`):
            A tensor of shape `(batch_size, total_spatial_elements, d_model)` representing the input embeddings.
        position_embeddings (`torch.FloatTensor`):
            A tensor of shape `(batch_size, total_spatial_elements, d_model)` representing the positional embeddings.
        spatial_shapes (`torch.FloatTensor`):
            A tensor of shape `(num_levels, 2)` where each row contains the height and width of a feature map level.
        spatial_shapes_list (`List[Tuple[int, int]]`):
            A list of tuples representing the spatial shapes of each memory level.
        level_start_index (`torch.FloatTensor`):
            A tensor indicating the start index of each level in the memory.
        valid_ratios (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_levels, 2)` representing the valid ratios for each level.
        key_padding_mask (`torch.LongTensor`):
            A tensor of shape `(batch_size, total_spatial_elements)` indicating which spatial elements are padded.

    Returns:
        `DinoDetrEncoderOutput`:
            - `output` (`torch.FloatTensor`): The final output of the encoder.
            - `intermediate_output` (`Optional[torch.FloatTensor]`): Stacked intermediate hidden states.
            - `intermediate_ref` (`Optional[torch.FloatTensor]`): Stacked intermediate reference points.

    """

    def __init__(
        self,
        encoder_layer: DinoDetrEncoderLayer,
        norm: torch.nn.Module,
        config: DinoDetrConfig,
    ):
        super().__init__(config)
        if config.num_encoder_layers > 0:
            self.layers = _get_clones(
                encoder_layer,
                config.num_encoder_layers,
                layer_share=False,
            )
        else:
            self.layers = []
            del encoder_layer
        self.num_queries = config.num_queries
        self.num_encoder_layers = config.num_encoder_layers
        self.norm = norm
        self.d_model = config.d_model
        self.post_init()

    @staticmethod
    def get_reference_points(
        spatial_shapes: torch.FloatTensor,
        valid_ratios: torch.FloatTensor,
        device: torch.device,
    ):
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @can_return_tuple
    def forward(
        self,
        input_embeddings: torch.FloatTensor,
        position_embeddings: torch.FloatTensor,
        spatial_shapes: torch.FloatTensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: torch.FloatTensor,
        valid_ratios: torch.FloatTensor,
        key_padding_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Forward pass for the encoder module.

        Args:
            input_embeds (`torch.FloatTensor`):
                A tensor of shape `(batch_size, sum(height_i * width_i), hidden_size)` representing the input embeddings.
            position_embeddings (`torch.FloatTensor`):
                A tensor of shape `(batch_size, sum(height_i * width_i), hidden_size)` representing the positional embeddings.
            spatial_shapes (`torch.FloatTensor`):
                A tensor of shape `(num_levels, 2)` where each row contains the height and width of a feature map level.
            level_start_index (`torch.FloatTensor`):
                A tensor of shape `(num_levels,)` indicating the start index of each level in the flattened spatial dimensions.
            valid_ratios (`torch.FloatTensor`):
                A tensor of shape `(batch_size, num_levels, 2)` representing the valid ratios for each level.
            key_padding_mask (`torch.FloatTensor`):
                A tensor of shape `(batch_size, sum(height_i * width_i))` indicating which spatial elements are padded.

        Returns:
            `torch.FloatTensor`:
                A tensor of shape `(batch_size, sum(height_i * width_i), hidden_size)` representing the output embeddings.
        """

        output = input_embeddings

        if self.num_encoder_layers > 0:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=input_embeddings.device)

        intermediate_output = []
        intermediate_ref = []
        encoder_states = ()

        for layer_id, layer in enumerate(self.layers):
            encoder_states = encoder_states + (output,)

            output_layer = layer(
                hidden_states=output,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                attention_mask=key_padding_mask,
            )

            output = output_layer[0]

        encoder_states = encoder_states + (output,)
        if self.norm is not None:
            output = self.norm(output)

        intermediate_output = intermediate_ref = None

        return DinoDetrEncoderOutput(
            output=output,
            intermediate_output=intermediate_output,
            intermediate_ref=intermediate_ref,
            encoder_states=encoder_states,
        )


class DinoDetrDecoder(DinoDetrPreTrainedModel):
    """
    The decoder module for the Dino DETR model.

    This module processes queries through multiple decoder layers, applying self-attention, cross-attention, and feed-forward networks.
    It supports optional dropout, query perturbation, and iterative refinement of reference points.

    Args:
        decoder_layer (`DinoDetrDecoderLayer`):
            A single decoder layer to be cloned and stacked.
        norm (`torch.nn.Module`):
            A normalization layer applied to the decoder's output.
            A function to apply perturbations to the decoder queries during training.
        config (`DinoDetrConfig`):
            The configuration object containing model parameters. Must include the following attributes:
            - `num_decoder_layers` (int): The number of decoder layers.
            - `num_feature_levels` (int): The number of feature levels.
            - `query_dim` (int): The dimensionality of the query embeddings.
            - `d_model` (int): The hidden size of the model.

    Inputs:
        queries (`torch.FloatTensor`):
            A tensor of shape `(num_queries, batch_size, d_model)` representing the input queries.
        memory (`torch.FloatTensor`):
            A tensor of shape `(memory_size, batch_size, d_model)` representing the encoder's output memory.
        refpoints_unsigmoid (`torch.FloatTensor`):
            A tensor of shape `(num_queries, batch_size, 2 or 4)` representing the initial reference points.
        spatial_shapes_list (`List[Tuple[int, int]]`):
            A list of tuples representing the spatial shapes of each memory level.
        self_attn_mask (`Optional[torch.LongTensor]`, *optional*):
            A tensor of shape `(num_queries, num_queries)` representing the self-attention mask.
        memory_key_padding_mask (`Optional[torch.LongTensor]`, *optional*):
            A tensor of shape `(batch_size, memory_size)` indicating which memory elements are padded.
        level_start_index (`Optional[torch.FloatTensor]`, *optional*):
            A tensor indicating the start index of each level in the memory.
        spatial_shapes (`Optional[torch.FloatTensor]`, *optional*):
            A tensor of shape `(num_levels, 2)` where each row contains the height and width of a feature map level.
        valid_ratios (`Optional[torch.FloatTensor]`, *optional*):
            A tensor of shape `(batch_size, num_levels, 2)` representing the valid ratios for each level.

    Returns:
        `DinoDetrDecoderOutput` object containing:
            - `intermediate` (`List[torch.FloatTensor]`): Stacked intermediate hidden states.
            - `ref_points` (`List[torch.FloatTensor]`): Stacked intermediate reference points.
    """

    def __init__(
        self,
        decoder_layer: DinoDetrDecoderLayer,
        norm: torch.nn.Module,
        config: DinoDetrConfig,
    ):
        super().__init__(config)
        if config.num_decoder_layers > 0:
            self.layers = _get_clones(
                decoder_layer,
                config.num_decoder_layers,
                layer_share=False,
            )
        else:
            self.layers = []
        self.num_decoder_layers = config.num_decoder_layers
        self.norm = norm
        self.num_feature_levels = config.num_feature_levels
        self.ref_point_head = DinoDetrMLPPredictionHead(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        self.query_pos_sine_scale = None
        self.bbox_embed = None
        self.class_embed = None
        self.d_model = config.d_model
        self.ref_anchor_head = None

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        queries: torch.FloatTensor,
        memory: torch.FloatTensor,
        refpoints_unsigmoid: torch.FloatTensor,
        spatial_shapes_list: list[tuple[int, int]],
        self_attn_mask: Optional[torch.LongTensor] = None,
        memory_key_padding_mask: Optional[torch.LongTensor] = None,
        level_start_index: Optional[torch.FloatTensor] = None,
        spatial_shapes: Optional[torch.FloatTensor] = None,
        valid_ratios: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Forward pass for the decoder.

        Args:
            tgt (`torch.FloatTensor`):
                A tensor of shape `(num_queries, batch_size, d_model)` representing the target (decoder input) embeddings.
            memory (`torch.FloatTensor`):
                A tensor of shape `(memory_size, batch_size, d_model)` representing the encoder's output memory.
            pos (`torch.FloatTensor`):
                A tensor of shape `(memory_size, batch_size, d_model)` representing the positional embeddings for the memory.
            refpoints_unsigmoid (`torch.FloatTensor`):
                A tensor of shape `(num_queries, batch_size, 2 or 4)` representing the unsigmoided reference points.
            valid_ratios (`torch.FloatTensor`):
                A tensor of shape `(batch_size, num_levels, 2)` representing the valid ratios for each feature map level.
            spatial_shapes (`torch.FloatTensor`):
                A tensor of shape `(num_levels, 2)` where each row contains the height and width of a feature map level.

        Returns:
            `torch.FloatTensor`:
                A tensor of shape `(num_queries, batch_size, d_model)` representing the updated target embeddings.
        """

        output = queries

        intermediate = [output]
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # Format reference points
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            elif reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]

            # Create query positional embeddings
            query_sine_embed = gen_sine_position_embeddings(
                reference_points_input[:, :, 0, :], d_model=self.d_model
            )  # nq, bs, 256*2
            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256

            output_layer = layer(
                output,
                query_position_embeddings=query_pos,
                query_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_spatial_shapes_list=spatial_shapes_list,
                self_attn_mask=self_attn_mask,
                **kwargs,
            )
            output = output_layer[0]

            # Compute new reference points
            if self.bbox_embed is not None:
                new_reference_points_unsigmoid = self.bbox_embed[layer_id](output) + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points_unsigmoid.sigmoid()
                reference_points = new_reference_points
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return DinoDetrDecoderOutput(
            intermediate=[itm_out.transpose(0, 1) for itm_out in intermediate],
            ref_points=[itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        )


class DinoDetrEncoderDecoder(DinoDetrPreTrainedModel):
    """
    The encoder-decoder module for the Dino DETR model.

    This module combines an encoder and a decoder to process input features and produce predictions for object detection tasks.
    It supports multi-scale features, two-stage decoding, and query perturbation.

    Args:
        config (`DinoDetrConfig`):
            The configuration object containing model parameters. Must include the following attributes:
            - `num_feature_levels` (int): The number of feature levels.
            - `num_queries` (int): The number of queries.
            - `d_model` (int): The hidden size of the model.
            - `normalize_before` (bool): Whether to apply normalization before the encoder layers.
            - `num_encoder_layers` (int): The number of encoder layers.
            - `num_decoder_layers` (int): The number of decoder layers.

    Inputs:
        pixel_values (`List[torch.FloatTensor]`):
            A list of tensors of shape `(batch_size, channels, height, width)` representing the input features.
        pixel_masks (`List[torch.LongTensor]`):
            A list of tensors of shape `(batch_size, height, width)` representing the input masks.
        pixel_position_embeddings (`List[torch.FloatTensor]`):
            A list of tensors of shape `(batch_size, channels, height, width)` representing the positional embeddings.
        query_reference_points (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, 4)` representing the reference points for the queries.
        queries (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, d_model)` representing the input queries.
        attn_mask (`Optional[torch.FloatTensor]`, *optional*):
            A tensor representing the attention mask for the decoder.

    Returns:
        `DinoDetrEncoderDecoderOutput` object containing:
            - `hidden_states` (`torch.FloatTensor`): The final hidden states of the decoder.
            - `reference_points` (`torch.FloatTensor`): The final reference points of the decoder.
            - `hidden_states_encoder` (`Optional[torch.FloatTensor]`): The final hidden states of the encoder.
            - `reference_points_encoder` (`Optional[torch.FloatTensor]`): The final reference points of the encoder.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_feature_levels = config.num_feature_levels
        self.num_queries = config.num_queries
        self.d_model = config.d_model
        # Define encoder
        encoder_layer = DinoDetrEncoderLayer(config)
        encoder_norm = None
        self.encoder = DinoDetrEncoder(encoder_layer, encoder_norm, config)
        # Define decoder
        decoder_layer = DinoDetrDecoderLayer(config)
        decoder_norm = nn.LayerNorm(config.d_model)
        self.decoder = DinoDetrDecoder(
            decoder_layer=decoder_layer,
            norm=decoder_norm,
            config=config,
        )
        # Define embedding layers
        if config.num_feature_levels > 1:
            if config.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))
            else:
                self.level_embed = None
        self.content_query_embeddings = nn.Embedding(self.num_queries, config.d_model)
        nn.init.normal_(self.content_query_embeddings.weight.data)
        # Define layers for the two stage Dino method
        self.enc_output = nn.Linear(config.d_model, config.d_model)
        self.enc_output_norm = nn.LayerNorm(config.d_model)
        self.two_stage_wh_embedding = None
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self.post_init()

    def get_valid_ratio(self, mask: torch.FloatTensor):
        """
        Compute the valid ratio of the height and width for a given mask.

        Args:
            mask (torch.FloatTensor): A 3D tensor of shape `(batch_size, height, width)`
                where `1` indicates valid positions and `0` indicates padded positions.

        Returns:
            torch.FloatTensor: A tensor of shape `(batch_size, 2)` containing the valid
            height and width ratios for each item in the batch. The first column corresponds
            to the width ratio, and the second column corresponds to the height ratio.

        Example:
            >>> mask = torch.tensor([[[1, 1, 0], [1, 0, 0]]], dtype=torch.float32)
            >>> valid_ratio = self.get_valid_ratio(mask)
            >>> print(valid_ratio)
            tensor([[0.6667, 0.5000]])
        """
        _, height, width = mask.shape
        valid_H = torch.sum(mask[:, :, 0], 1)
        valid_W = torch.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / height
        valid_ratio_w = valid_W.float() / width
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries: int):
        self.content_query_reference_points = nn.Embedding(use_num_queries, 4)

    @can_return_tuple
    def forward(
        self,
        pixel_values: list[torch.FloatTensor],
        pixel_masks: list[torch.LongTensor],
        pixel_position_embeddings: list[torch.FloatTensor],
        contrastive_query_reference_points: torch.FloatTensor,
        contrastive_queries: torch.FloatTensor,
        attn_mask: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Forward pass for the encoder-decoder module.

        Args:
            pixel_values (`List[torch.FloatTensor]`):
                A list of tensors of shape `(batch_size, channels, height, width)` representing the input features.
            pixel_masks (`List[torch.LongTensor]`):
                A list of tensors of shape `(batch_size, height, width)` representing the input masks.
            pixel_position_embeddings (`List[torch.FloatTensor]`):
                A list of tensors of shape `(batch_size, channels, height, width)` representing the positional embeddings.
            contrastive_query_reference_points (`torch.FloatTensor`):
                A tensor of shape `(batch_size, num_queries, 4)` representing the reference points for the queries.
            contrastive_queries (`torch.FloatTensor`):
                A tensor of shape `(batch_size, num_queries, d_model)` representing the input queries.
            attn_mask (`Optional[torch.FloatTensor]`, *optional*):
                A tensor representing the attention mask for the decoder.

        Returns:
            `DinoDetrEncoderDecoderOutput` object containing:
                - `hidden_states` (`torch.FloatTensor`): The final hidden states of the decoder.
                - `reference_points` (`torch.FloatTensor`): The final reference points of the decoder.
                - `hidden_states_encoder` (`Optional[torch.FloatTensor]`): The final hidden states of the encoder.
                - `reference_points_encoder` (`Optional[torch.FloatTensor]`): The final reference points of the encoder.
        """

        # Format input for encoder, mainly flatten
        src_flatten = []
        mask_flatten = []
        level_pos_embed_flatten = []
        spatial_shapes_list = []
        for level, (src, mask, pos_embed) in enumerate(zip(pixel_values, pixel_masks, pixel_position_embeddings)):
            batch_size, c, height, width = src.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                level_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            else:
                level_pos_embed = pos_embed
            level_pos_embed_flatten.append(level_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        level_pos_embed_flatten = torch.cat(level_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in pixel_masks], 1)

        # Apply Encoder
        outputs_encoder_part = self.encoder(
            input_embeddings=src_flatten,
            position_embeddings=level_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            **kwargs,
        )
        (memory,) = (outputs_encoder_part["output"],)

        # Create and select topk queries. generate_initial_queries initializes bounding boxes on a 2d grid,
        # memory is simply masked
        mask_flatten = ~mask_flatten
        input_hw = None
        queries_init, query_reference_points_init = generate_initial_queries(
            memory, mask_flatten, spatial_shapes, input_hw
        )
        queries_init = self.enc_output_norm(self.enc_output(queries_init))
        query_reference_points_init = self.enc_out_bbox_embed(queries_init) + query_reference_points_init
        queries_class_init = self.enc_out_class_embed(queries_init)
        topk_indices = torch.topk(queries_class_init.max(-1)[0], self.num_queries, dim=1)[1]

        # Create topk reference_points
        query_reference_points = torch.gather(
            query_reference_points_init,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4),
        )

        # Create topk queries
        queries = torch.gather(
            queries_init,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, self.d_model),
        )

        # Create encoder decoder output
        hidden_states_encoder = queries.unsqueeze(0)
        reference_points_encoder = query_reference_points.sigmoid().unsqueeze(0)

        # Either use learnable queries (shared across all train/test images) or use per sample queries
        query_reference_points = query_reference_points.detach()
        queries = self.content_query_embeddings.weight[:, None, :].repeat(1, batch_size, 1).transpose(0, 1)

        # Combine queries and reference points with their contrastive denoising versions
        if contrastive_query_reference_points is not None and contrastive_queries is not None:
            query_reference_points = torch.cat([contrastive_query_reference_points, query_reference_points], dim=1)
            queries = torch.cat([contrastive_queries, queries], dim=1)

        # Apply decoder on the encoder output and combined queries
        outputs_decoder_part = self.decoder(
            queries=queries.transpose(0, 1),
            memory=memory.transpose(0, 1),
            self_attn_mask=attn_mask,
            memory_key_padding_mask=mask_flatten,
            refpoints_unsigmoid=query_reference_points.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        hidden_states, reference_points = (
            outputs_decoder_part["intermediate"],
            outputs_decoder_part["ref_points"],
        )

        return DinoDetrEncoderDecoderOutput(
            hidden_states=hidden_states,
            reference_points=reference_points,
            hidden_states_encoder=hidden_states_encoder,
            reference_points_encoder=reference_points_encoder,
        )


DINO_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DinoDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DINO_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DinoDetrImageProcessor.__call__`]
            for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
"""


@add_start_docstrings(
    """
    The bare Dino DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    DINO_DETR_START_DOCSTRING,
)
class DinoDetrModel(DinoDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [
        "bbox_embed",
        "class_embed",
        r"bbox_embed\.[1-9]\d*",
        r"class_embed\.[1-9]\d*",
        r"transformer\.decoder\.bbox_embed\.[1-9]\d*",
        r"transformer\.decoder\.class_embed\.[1-9]\d*",
    ]
    _can_record_outputs = {
        "encoder_self_attentions": OutputRecorder(
            DinoDetrMultiscaleDeformableAttention, layer_name="self_attn", index=1
        ),
        "decoder_self_attentions": OutputRecorder(MultiheadAttention, layer_name="self_attn", index=1),
        "decoder_cross_attentions": OutputRecorder(
            DinoDetrMultiscaleDeformableAttention, layer_name="cross_attn", index=1
        ),
        # "encoder_hidden_states": OutputRecorder(DinoDetrEncoder, layer_name="encoder", index=3),
        # "decoder_hidden_states": OutputRecorder(DinoDetrDecoder, layer_name="decoder", index=0),
        # "hidden_states": OutputRecorder(DinoDetrDecoderLayer, index=0),
        # "hidden_states": OutputRecorder(DinoDetrDecoder, layer_name="decoder", index=0),
        "encoder_hidden_states": DinoDetrEncoderLayer,
        "decoder_hidden_states": DinoDetrDecoderLayer,
    }
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: DinoDetrConfig):
        super().__init__(config)
        # Create deformable transformer
        self.transformer = DinoDetrEncoderDecoder(config=config)
        self.label_enc = nn.Embedding(config.denoising_num_classes + 1, config.d_model)

        # Create backbone + positional encoding
        backbone = DinoDetrConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = DinoDetrConvModel(backbone, position_embeddings)
        d_model = config.d_model

        # Prepare input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.conv_encoder.intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.conv_encoder.intermediate_channel_sizes[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, d_model, kernel_size=1),
                        nn.GroupNorm(32, d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            d_model,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(32, d_model),
                    )
                )
                in_channels = d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.backbone.conv_encoder.intermediate_channel_sizes[-1], d_model, kernel_size=1),
                        nn.GroupNorm(32, d_model),
                    )
                ]
            )

        # Prepare class & box embed
        self.class_embed = nn.Linear(config.d_model, config.num_classes)
        self.bbox_embed = DinoDetrMLPPredictionHead(d_model, d_model, 4, 3)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data.fill_(bias_value)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.bbox_embed = _get_clones(self.bbox_embed, config.num_decoder_layers, layer_share=True)
        self.class_embed = _get_clones(self.class_embed, config.num_decoder_layers, layer_share=True)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # Adjust embeddings based on two stage approach
        self.transformer.enc_out_bbox_embed = copy.deepcopy(self.bbox_embed[0])
        self.transformer.enc_out_class_embed = copy.deepcopy(self.class_embed[0])

        for layer in self.transformer.decoder.layers:
            layer.label_embedding = None
        self.label_embedding = None

        self.post_init()

    @can_return_tuple
    @add_start_docstrings_to_model_forward(DINO_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DinoDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    @check_model_inputs()
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[list[dict]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DinoDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("dino_detr")
        >>> model = DinoDetrModel.from_pretrained("dino_detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        """

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Apply backbone
        features, position_embeddings = self.backbone(pixel_values, pixel_mask)
        srcs = []
        masks = []
        for level, (src, mask) in enumerate(features):
            srcs.append(self.input_proj[level](src))
            masks.append(mask)

        # Deal with the case of more feature levels that backbone outputs
        if self.config.num_feature_levels > len(srcs):
            srcs_length = len(srcs)
            for additional_level in range(srcs_length, self.config.num_feature_levels):
                if additional_level == srcs_length:
                    src = self.input_proj[additional_level](features[-1][0])
                else:
                    src = self.input_proj[additional_level](srcs[-1])
                mask = F.interpolate(pixel_mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                additional_position_embedding = self.backbone.position_embedding(src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                position_embeddings.append(additional_position_embedding)

        # Create contrastive denoising queries
        if self.training and self.config.denoising_query_number > 0 and labels is not None:
            contrastive_queries, contrastive_query_reference_points, attn_mask, denoising_meta = (
                get_contrastive_denoising_training_group(
                    targets=labels,
                    num_classes=self.config.num_classes,
                    num_queries=self.config.num_queries,
                    class_embed=self.label_enc,
                    num_denoising_queries=self.config.denoising_query_number,
                    label_noise_ratio=self.config.denoising_label_noise_ratio,
                    box_noise_scale=self.config.denoising_box_noise_scale,
                )
            )
        else:
            contrastive_queries = contrastive_query_reference_points = attn_mask = denoising_meta = None

        # Apply transformer encoder decoder
        outputs_transformer_part = self.transformer(
            pixel_values=srcs,
            pixel_masks=masks,
            pixel_position_embeddings=position_embeddings,
            contrastive_query_reference_points=contrastive_query_reference_points,
            contrastive_queries=contrastive_queries,
            attn_mask=attn_mask,
            **kwargs,
        )
        (
            hidden_states,
            reference_points,
            hidden_states_encoder,
            reference_points_encoder,
        ) = (
            outputs_transformer_part["hidden_states"],
            outputs_transformer_part["reference_points"],
            outputs_transformer_part["hidden_states_encoder"],
            outputs_transformer_part["reference_points_encoder"],
        )

        return DinoDetrModelOutput(
            last_hidden_state=hidden_states[-1],
            hidden_states_model=hidden_states,
            references=reference_points,
            encoder_last_hidden_state=hidden_states_encoder,
            encoder_reference=reference_points_encoder,
            denoising_meta=denoising_meta,
        )


@add_start_docstrings(
    """
    Dino DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    DINO_DETR_START_DOCSTRING,
)
class DinoDetrForObjectDetection(DinoDetrPreTrainedModel):
    _tied_weights_keys = [
        "bbox_embed",
        "class_embed",
        r"bbox_embed\.[1-9]\d*",
        r"class_embed\.[1-9]\d*",
        r"transformer\.decoder\.bbox_embed\.[1-9]\d*",
        r"transformer\.decoder\.class_embed\.[1-9]\d*",
    ]
    _can_record_outputs = {
        "encoder_self_attentions": OutputRecorder(
            DinoDetrMultiscaleDeformableAttention, layer_name="self_attn", index=1
        ),
        "decoder_self_attentions": OutputRecorder(MultiheadAttention, layer_name="self_attn", index=1),
        "decoder_cross_attentions": OutputRecorder(
            DinoDetrMultiscaleDeformableAttention, layer_name="cross_attn", index=1
        ),
        # "hidden_states": DinoDetrDecoderLayer,
        "encoder_hidden_states": DinoDetrEncoderLayer,
        "decoder_hidden_states": DinoDetrDecoderLayer,
    }

    def __init__(self, config: DinoDetrConfig):
        super().__init__(config)

        # Dino DETR encoder-decoder model
        self.model = DinoDetrModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @add_start_docstrings_to_model_forward(DINO_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DinoDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    @check_model_inputs()
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        labels: Optional[list[dict]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DinoDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("dino_detr")
        >>> model = DinoDetrForObjectDetection.from_pretrained("dino_detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)[
        ...     0
        ... ]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected <object> with confidence 0.7475 at location <location>
        Detected <object> with confidence 0.7341 at location <location>
        Detected <object> with confidence 0.7229 at location <location>
        ```"""

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        denoising_meta = None

        # Apply backbone, encoder-decoder and contrastive group to inputs
        outputs_model_part = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
            **kwargs,
        )

        last_hidden_state = outputs_model_part.last_hidden_state
        hidden_states = outputs_model_part.hidden_states_model[1:]
        reference_points = outputs_model_part.references
        hidden_states_encoder = outputs_model_part.encoder_last_hidden_state
        reference_points_encoder = outputs_model_part.encoder_reference
        if self.training:
            denoising_meta = outputs_model_part.denoising_meta

        # Convert hidden states to bounding boxes
        hidden_states[0] += self.model.label_enc.weight[0, 0] * 0.0
        outputs_coord_list = []
        outputs_class_list = []
        for (
            layer_reference_points_sigmoid,
            layer_bbox_embed,
            layer_cls_embed,
            layer_hidden_states,
        ) in zip(
            reference_points[:-1],
            self.model.transformer.decoder.bbox_embed,
            self.model.transformer.decoder.class_embed,
            hidden_states,
        ):
            layer_outputs_unsigmoid = layer_bbox_embed(layer_hidden_states) + inverse_sigmoid(
                layer_reference_points_sigmoid
            )
            outputs_coord_list.append(layer_outputs_unsigmoid.sigmoid())
            outputs_class_list.append(layer_cls_embed(layer_hidden_states))

        outputs_coord = torch.stack(outputs_coord_list)
        outputs_class = torch.stack(outputs_class_list)

        # Apply denoising post processing
        if self.config.denoising_query_number > 0 and denoising_meta is not None:
            outputs_class, outputs_coord = denoising_post_process(
                outputs_class,
                outputs_coord,
                denoising_meta,
                self.config.auxiliary_loss,
                self._set_aux_loss,
            )

        # Compute auxiliary loss
        if self.config.auxiliary_loss:
            out_aux_loss = self._set_aux_loss(outputs_class, outputs_coord)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits=outputs_class[-1],
                labels=labels,
                device=self.device,
                pred_boxes=outputs_coord[-1],
                denoising_meta=denoising_meta,
                outputs_class=outputs_class,
                outputs_coord=outputs_coord,
                class_cost=self.config.class_cost,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost,
                num_labels=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                auxiliary_loss=self.config.auxiliary_loss,
                cls_loss_coefficient=self.config.cls_loss_coefficient,
                bbox_loss_coefficient=self.config.bbox_loss_coefficient,
                giou_loss_coefficient=self.config.giou_loss_coefficient,
                mask_loss_coefficient=self.config.mask_loss_coefficient,
                use_denoising=self.config.use_denoising,
                use_masks=self.config.use_masks,
                dice_loss_coefficient=self.config.dice_loss_coefficient,
                num_decoder_layers=self.config.num_decoder_layers,
            )

        dict_outputs = DinoDetrObjectDetectionOutput(
            last_hidden_state=last_hidden_state,
            reference=reference_points[-1],
            encoder_last_hidden_state=hidden_states_encoder,
            encoder_reference=reference_points_encoder,
            loss=loss,
            loss_dict=loss_dict,
            logits=outputs_class[-1],
            pred_boxes=outputs_coord_list[-1],
            auxiliary_outputs=out_aux_loss,
            denoising_meta=denoising_meta,
        )
        return dict_outputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DinoDetrImageProcessor(DetrImageProcessor):
    def post_process_object_detection(
        self, outputs, threshold: float = 0.3, target_sizes: Union[TensorType, list[tuple]] = None, top_k: int = 300
    ):
        """
        Converts the raw output of [`DinoDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DinoDetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            top_k (`int`, *optional*, defaults to 100):
                Keep only top k bounding boxes before filtering by thresholding.

        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = out_logits.sigmoid()
        prob = prob.view(out_logits.shape[0], -1)
        k_value = min(top_k, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results


__all__ = [
    "DinoDetrImageProcessor",
    "DinoDetrForObjectDetection",
    "DinoDetrModel",
    "DinoDetrPreTrainedModel",
]
