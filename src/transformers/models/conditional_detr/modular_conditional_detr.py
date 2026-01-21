import math
from collections.abc import Callable

import torch
from torch import nn

from ...image_transforms import (
    center_to_corners_format,
)
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    is_timm_available,
    logging,
)
from ...utils.generic import OutputRecorder, can_return_tuple, check_model_inputs
from ..deformable_detr.modeling_deformable_detr import inverse_sigmoid
from ..detr.image_processing_detr_fast import DetrImageProcessorFast
from ..detr.modeling_detr import (
    DetrConvEncoder,
    DetrDecoderLayer,
    DetrDecoderOutput,
    DetrEncoder,
    DetrEncoderLayer,
    DetrForObjectDetection,
    DetrForSegmentation,
    DetrLearnedPositionEmbedding,
    DetrMLP,
    DetrMLPPredictionHead,
    DetrModel,
    DetrModelOutput,
    DetrObjectDetectionOutput,
    DetrPreTrainedModel,
    DetrSegmentationOutput,
    DetrSelfAttention,
    DetrSinePositionEmbedding,
    eager_attention_forward,
)
from .configuration_conditional_detr import ConditionalDetrConfig


if is_timm_available():
    pass


logger = logging.get_logger(__name__)


class ConditionalDetrImageProcessorFast(DetrImageProcessorFast):
    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: TensorType | list[tuple] = None, top_k: int = 100
    ):
        """
        Converts the raw output of [`ConditionalDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`ConditionalDetrObjectDetectionOutput`]):
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

    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple[int, int]] | None = None):
        """
        Converts the output of [`ConditionalDetrForSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`ConditionalDetrForSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                A list of tuples (`tuple[int, int]`) containing the target size (height, width) of each image in the
                batch. If unset, predictions will not be resized.
        Returns:
            `list[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        # Conditional DETR does not have a null class, so we use all classes
        masks_classes = class_queries_logits.softmax(dim=-1)
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation


class ConditionalDetrDecoderOutput(DetrDecoderOutput):
    r"""
    cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
        used to compute the weighted average in the cross-attention heads.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
        Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
        layernorm.
    reference_points (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, 2 (anchor points))`):
        Reference points (reference points of each layer of the decoder).
    """

    reference_points: tuple[torch.FloatTensor] | None = None


class ConditionalDetrModelOutput(DetrModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
        Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
        layernorm.
    reference_points (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, 2 (anchor points))`):
        Reference points (reference points of each layer of the decoder).
    """

    reference_points: tuple[torch.FloatTensor] | None = None


# function to generate sine positional embedding for 2d coordinates
def gen_sine_position_embeddings(pos_tensor, d_model):
    scale = 2 * math.pi
    dim = d_model // 2
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos.to(pos_tensor.dtype)


class ConditionalDetrObjectDetectionOutput(DetrObjectDetectionOutput):
    pass


class ConditionalDetrSegmentationOutput(DetrSegmentationOutput):
    pass


class ConditionalDetrConvEncoder(DetrConvEncoder):
    pass


class ConditionalDetrSinePositionEmbedding(DetrSinePositionEmbedding):
    pass


class ConditionalDetrLearnedPositionEmbedding(DetrLearnedPositionEmbedding):
    pass


class ConditionalDetrSelfAttention(DetrSelfAttention):
    pass


class ConditionalDetrDecoderSelfAttention(nn.Module):
    """
    Multi-headed self-attention for Conditional DETR decoder layers.

    This attention module handles separate content and position projections, which are then combined
    before applying standard self-attention. Position embeddings are added to both queries and keys.
    """

    def __init__(
        self,
        config: ConditionalDetrConfig,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout
        self.is_causal = False

        # Content and position projections
        self.q_content_proj = nn.Linear(hidden_size, hidden_size)
        self.q_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.k_content_proj = nn.Linear(hidden_size, hidden_size)
        self.k_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
                Input hidden states from the decoder layer.
            query_position_embeddings (`torch.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
                Position embeddings for queries and keys. Required (unlike standard attention). Processed through
                separate position projections (`q_pos_proj`, `k_pos_proj`) and added to content projections.
            attention_mask (`torch.Tensor` of shape `(batch_size, 1, num_queries, num_queries)`, *optional*):
                Attention mask to avoid attending to padding tokens.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = (
            (self.q_content_proj(hidden_states) + self.q_pos_proj(query_position_embeddings))
            .view(hidden_shape)
            .transpose(1, 2)
        )
        key_states = (
            (self.k_content_proj(hidden_states) + self.k_pos_proj(query_position_embeddings))
            .view(hidden_shape)
            .transpose(1, 2)
        )
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ConditionalDetrDecoderCrossAttention(nn.Module):
    """
    Multi-headed cross-attention for Conditional DETR decoder layers.

    This attention module handles the special cross-attention logic in Conditional DETR:
    - Separate content and position projections for queries and keys
    - Concatenation of query sine embeddings with queries (doubling query dimension)
    - Concatenation of key position embeddings with keys (doubling key dimension)
    - Output dimension remains hidden_size despite doubled input dimensions
    """

    def __init__(
        self,
        config: ConditionalDetrConfig,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = dropout
        self.is_causal = False

        # Content and position projections
        self.q_content_proj = nn.Linear(hidden_size, hidden_size)
        self.q_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.k_content_proj = nn.Linear(hidden_size, hidden_size)
        self.k_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_pos_sine_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection: input is hidden_size * 2 (from concatenated q/k), output is hidden_size
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Compute scaling for expanded head_dim (q and k have doubled dimensions after concatenation)
        # This matches the original Conditional DETR implementation where embed_dim * 2 is used
        expanded_head_dim = (hidden_size * 2) // num_attention_heads
        self.scaling = expanded_head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        query_sine_embed: torch.Tensor,
        encoder_position_embeddings: torch.Tensor,
        query_position_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
                Decoder hidden states (queries).
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, encoder_seq_len, hidden_size)`):
                Encoder output hidden states (keys and values).
            query_sine_embed (`torch.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
                Sine position embeddings for queries. **Concatenated** (not added) with query content,
                doubling the query dimension.
            encoder_position_embeddings (`torch.Tensor` of shape `(batch_size, encoder_seq_len, hidden_size)`):
                Position embeddings for keys. **Concatenated** (not added) with key content, doubling the key dimension.
            query_position_embeddings (`torch.Tensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Additional position embeddings. When provided (first layer only), **added** to query content
                before concatenation with `query_sine_embed`. Also causes `encoder_position_embeddings` to be
                added to key content before concatenation.
            attention_mask (`torch.Tensor` of shape `(batch_size, 1, num_queries, encoder_seq_len)`, *optional*):
                Attention mask to avoid attending to padding tokens.
        """
        query_input_shape = hidden_states.shape[:-1]
        kv_input_shape = encoder_hidden_states.shape[:-1]
        query_hidden_shape = (*query_input_shape, self.num_attention_heads, self.head_dim)
        kv_hidden_shape = (*kv_input_shape, self.num_attention_heads, self.head_dim)

        # Apply content and position projections
        query_input = self.q_content_proj(hidden_states)
        key_input = self.k_content_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)
        key_pos = self.k_pos_proj(encoder_position_embeddings)

        # Combine content and position embeddings
        if query_position_embeddings is not None:
            query_input = query_input + self.q_pos_proj(query_position_embeddings)
            key_input = key_input + key_pos

        # Reshape and concatenate position embeddings (doubling head_dim)
        query_input = query_input.view(query_hidden_shape)
        key_input = key_input.view(kv_hidden_shape)
        query_sine_embed = self.q_pos_sine_proj(query_sine_embed).view(query_hidden_shape)
        key_pos = key_pos.view(kv_hidden_shape)

        query_states = torch.cat([query_input, query_sine_embed], dim=-1).view(*query_input_shape, -1)
        key_states = torch.cat([key_input, key_pos], dim=-1).view(*kv_input_shape, -1)

        # Reshape for attention computation
        expanded_head_dim = query_states.shape[-1] // self.num_attention_heads
        query_states = query_states.view(*query_input_shape, self.num_attention_heads, expanded_head_dim).transpose(
            1, 2
        )
        key_states = key_states.view(*kv_input_shape, self.num_attention_heads, expanded_head_dim).transpose(1, 2)
        value_states = value_states.view(kv_hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*query_input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ConditionalDetrMLP(DetrMLP):
    pass


class ConditionalDetrEncoderLayer(DetrEncoderLayer):
    pass


class ConditionalDetrDecoderLayer(DetrDecoderLayer):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__()
        self.self_attn = ConditionalDetrDecoderSelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn = ConditionalDetrDecoderCrossAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        spatial_position_embeddings: torch.Tensor | None = None,
        query_position_embeddings: torch.Tensor | None = None,
        query_sine_embed: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        is_first: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            spatial_position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Spatial position embeddings (2D positional encodings) that are added to the queries and keys in each self-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                object_queries that are added to the queries and keys
                in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            query_position_embeddings=query_position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, _ = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                query_sine_embed=query_sine_embed,
                encoder_position_embeddings=spatial_position_embeddings,
                # Only pass query_position_embeddings for the first layer
                query_position_embeddings=query_position_embeddings if is_first else None,
                **kwargs,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class ConditionalDetrMLPPredictionHead(DetrMLPPredictionHead):
    pass


class ConditionalDetrPreTrainedModel(DetrPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"detr\.model\.backbone\.model\.layer\d+\.0\.downsample\.1\.num_batches_tracked"
    ]


class ConditionalDetrEncoder(DetrEncoder):
    pass


class ConditionalDetrDecoder(ConditionalDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`ConditionalDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for Conditional DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: ConditionalDetrConfig
    """

    _can_record_outputs = {
        "hidden_states": ConditionalDetrDecoderLayer,
        "attentions": OutputRecorder(ConditionalDetrDecoderSelfAttention, layer_name="self_attn", index=1),
        "cross_attentions": OutputRecorder(ConditionalDetrDecoderCrossAttention, layer_name="encoder_attn", index=1),
    }

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.hidden_size = config.d_model

        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.ModuleList([ConditionalDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in Conditional DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(config.d_model)

        # query_scale is the FFN applied on f to generate transformation T
        self.query_scale = ConditionalDetrMLPPredictionHead(self.hidden_size, self.hidden_size, self.hidden_size, 2)
        self.ref_point_head = ConditionalDetrMLPPredictionHead(self.hidden_size, self.hidden_size, 2, 2)
        for layer_id in range(config.decoder_layers - 1):
            # Set q_pos_proj to None for layers after the first (only first layer uses query position embeddings)
            self.layers[layer_id + 1].encoder_attn.q_pos_proj = None

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        spatial_position_embeddings=None,
        object_queries_position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ConditionalDetrDecoderOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            spatial_position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Spatial position embeddings that are added to the queries and keys in each cross-attention layer.
            object_queries_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            encoder_attention_mask = create_bidirectional_mask(
                self.config,
                inputs_embeds,
                encoder_attention_mask,
            )

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        reference_points_before_sigmoid = self.ref_point_head(
            object_queries_position_embeddings
        )  # [num_queries, batch_size, 2]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        obj_center = reference_points[..., :2].transpose(0, 1)
        # get sine embedding for the query vector
        query_sine_embed_before_transformation = gen_sine_position_embeddings(obj_center, self.config.d_model)

        for idx, decoder_layer in enumerate(self.layers):
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            if idx == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(hidden_states)
            # apply transformation
            query_sine_embed = query_sine_embed_before_transformation * pos_transformation

            hidden_states = decoder_layer(
                hidden_states,
                None,
                spatial_position_embeddings,
                object_queries_position_embeddings,
                query_sine_embed,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                is_first=(idx == 0),
                **kwargs,
            )

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)

        # stack intermediate decoder activations
        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)

        return ConditionalDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            reference_points=reference_points,
        )


class ConditionalDetrModel(DetrModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ConditionalDetrModelOutput:
        r"""
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
        >>> model = AutoModel.from_pretrained("microsoft/conditional-detr-resnet-50")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features = self.backbone(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        projected_feature_map = self.input_projection(feature_map)

        # Generate position embeddings
        spatial_position_embeddings = self.position_embedding(
            shape=feature_map.shape, device=device, dtype=pixel_values.dtype, mask=mask
        )

        # Third, flatten the feature map of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + spatial_position_embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, height*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, height*width)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                spatial_position_embeddings=spatial_position_embeddings,
                **kwargs,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings through the decoder (which is conditioned on the encoder output)
        object_queries_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        queries = torch.zeros_like(object_queries_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            spatial_position_embeddings=spatial_position_embeddings,
            object_queries_position_embeddings=object_queries_position_embeddings,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=flattened_mask,
            **kwargs,
        )

        return ConditionalDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            reference_points=decoder_outputs.reference_points,
        )


class ConditionalDetrForObjectDetection(DetrForObjectDetection):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.class_labels_classifier = nn.Linear(config.d_model, config.num_labels)

    # taken from https://github.com/Atten4Vis/conditionalDETR/blob/master/models/conditional_detr.py
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ConditionalDetrObjectDetectionOutput:
        r"""
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
        >>> model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
        ...     0
        ... ]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.833 at location [38.31, 72.1, 177.63, 118.45]
        Detected cat with confidence 0.831 at location [9.2, 51.38, 321.13, 469.0]
        Detected cat with confidence 0.804 at location [340.3, 16.85, 642.93, 370.95]
        Detected remote with confidence 0.683 at location [334.48, 73.49, 366.37, 190.01]
        Detected couch with confidence 0.535 at location [0.52, 1.19, 640.35, 475.1]
        ```"""
        # First, sent images through CONDITIONAL_DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs,
        )

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)

        reference = outputs.reference_points
        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)

        hs = sequence_output
        tmp = self.bbox_predictor(hs)
        tmp[..., :2] += reference_before_sigmoid
        pred_boxes = tmp.sigmoid()
        # pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                outputs_coords = []
                intermediate = outputs.intermediate_hidden_states
                outputs_class = self.class_labels_classifier(intermediate)
                for lvl in range(intermediate.shape[0]):
                    tmp = self.bbox_predictor(intermediate[lvl])
                    tmp[..., :2] += reference_before_sigmoid
                    outputs_coord = tmp.sigmoid()
                    outputs_coords.append(outputs_coord)
                outputs_coord = torch.stack(outputs_coords)
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            )

        return ConditionalDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class ConditionalDetrForSegmentation(DetrForSegmentation):
    pass


__all__ = [
    "ConditionalDetrImageProcessorFast",
    "ConditionalDetrForObjectDetection",
    "ConditionalDetrForSegmentation",
    "ConditionalDetrModel",
    "ConditionalDetrPreTrainedModel",
]
