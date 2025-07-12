# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from typing import Optional, Union

import torch
from torch import nn

from transformers.modeling_outputs import ModelOutput
from transformers.utils.generic import TransformersKwargs, check_model_inputs

from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..sam.configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
from ..sam.modeling_sam import (
    SamFeedForward,
    SamImageSegmentationOutput,
    SamLayerNorm,
    SamModel,
    SamPreTrainedModel,
    SamTwoWayTransformer,
    SamVisionAttention,
    SamVisionEncoder,
    SamVisionEncoderOutput,
    SamVisionLayer,
    SamVisionModel,
)


logger = logging.get_logger(__name__)


class SamHQPromptEncoderConfig(SamPromptEncoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamHQPromptEncoderModel`].The [`SamHQPromptEncoderModel`]
    module is used to encode the input 2D points and bounding boxes. Instantiating a configuration defaults will yield a
    similar configuration to that of the SAM_HQ model. The configuration is used to store the configuration of the model.
    [Uminosachi/sam-hq](https://huggingface.co/Uminosachi/sam-hq) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model's output.Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1024):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
    """

    pass


class SamHQVisionConfig(SamVisionConfig):
    pass


class SamHQMaskDecoderConfig(SamMaskDecoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamHQMaskDecoder`]. It is used to instantiate a SAM_HQ
    mask decoder to the specified arguments, defining the model architecture. Instantiating a configuration defaults
    will yield a similar configuration to that of the SAM_HQ-vit-h
    [facebook/sam_hq-vit-huge](https://huggingface.co/facebook/sam_hq-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function used inside the `SamHQMaskDecoder` module.
        mlp_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_downsample_rate (`int`, *optional*, defaults to 2):
            The downsampling rate of the attention layer.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            The number of outputs from the `SamHQMaskDecoder` module. In the Segment Anything paper, this is set to 3.
        iou_head_depth (`int`, *optional*, defaults to 3):
            The number of layers in the IoU head module.
        iou_head_hidden_dim (`int`, *optional*, defaults to 256):
            The dimensionality of the hidden states in the IoU head module.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        vit_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the Vision Transformer (ViT) used in the `SamHQMaskDecoder` module.
    """

    def __init__(
        self,
        vit_dim=768,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.vit_dim = vit_dim


class SamHQConfig(SamConfig):
    r"""
    [`SamHQConfig`] is the configuration class to store the configuration of a [`SamHQModel`]. It is used to instantiate a
    SAM-HQ model according to the specified arguments, defining the vision model, prompt-encoder model and mask decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    SAM-HQ-ViT-H [sushmanth/sam_hq_vit_h](https://huggingface.co/sushmanth/sam_hq_vit_h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `SamHQVisionConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamHQVisionConfig`].
        prompt_encoder_config (Union[`dict`, `SamHQPromptEncoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamHQPromptEncoderConfig`].
        mask_decoder_config (Union[`dict`, `SamHQMaskDecoderConfig`], *optional*):
            Dictionary of configuration options used to initialize [`SamHQMaskDecoderConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    pass


class SamHQVisionEncoderOutput(SamVisionEncoderOutput):
    r"""
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The image embeddings obtained by applying the projection layer to the pooler_output.
    intermediate_embeddings (`list(torch.FloatTensor)`, *optional*):
        A list of intermediate embeddings collected from certain blocks within the model, typically those without
        windowed attention. Each element in the list is of shape `(batch_size, sequence_length, hidden_size)`.
        This is specific to SAM-HQ and not present in base SAM.
    """

    intermediate_embeddings: Optional[list[torch.FloatTensor]] = None


@dataclass
class SamHQMMaskDecoderOutputs(ModelOutput):
    r"""
    masks (`torch.FloatTensor` of shape `(batch_size, num_prompts, num_masks, height, width)`):
        The predicted masks for the input image. The masks are of shape `(batch_size, num_prompts, num_masks, height, width)`.
    iou_scores (`torch.FloatTensor` of shape `(batch_size, num_prompts, num_masks)`):
        The predicted IoU scores for each mask. The scores are of shape `(batch_size, num_prompts, num_masks)`.
    mask_decoder_attentions (`torch.FloatTensor`, *optional*):
        The attention weights from the mask decoder, if `output_attentions=True` was passed during the forward pass.
        This is specific to SAM-HQ and not present in base SAM.
    """

    masks: torch.FloatTensor
    iou_scores: Optional[torch.FloatTensor] = None
    mask_decoder_attentions: Optional[torch.FloatTensor] = None


class SamHQImageSegmentationOutput(SamImageSegmentationOutput):
    pass


class SamHQVisionAttention(SamVisionAttention):
    pass


class SamHQVisionLayer(SamVisionLayer):
    pass


class SamHQPreTrainedModel(SamPreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, SamHQVisionEncoder):
            if module.pos_embed is not None:
                module.pos_embed.data.zero_()


class SamHQVisionEncoder(SamVisionEncoder, SamHQPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": SamHQVisionLayer,
        "attentions": SamHQVisionAttention,
    }

    @check_model_inputs
    def forward(
        self, pixel_values: Optional[torch.FloatTensor] = None, **kwargs: Unpack[TransformersKwargs]
    ) -> Union[tuple, SamHQVisionEncoderOutput]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        intermediate_embeddings = []

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)

            # Collect embeddings from non-windowed blocks
            if hasattr(layer_module, "window_size") and layer_module.window_size == 0:
                intermediate_embeddings.append(hidden_states)

        hidden_states = self.neck(hidden_states)

        return SamHQVisionEncoderOutput(
            last_hidden_state=hidden_states,
            intermediate_embeddings=intermediate_embeddings,
        )


class SamHQLayerNorm(SamLayerNorm):
    pass


class SamHQTwoWayTransformer(SamTwoWayTransformer):
    pass


class SamHQFeedForward(SamFeedForward):
    pass


class SamHQMaskDecoder(nn.Module):
    def __init__(self, config: SamHQMaskDecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamHQTwoWayTransformer(config)

        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamHQLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamHQFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = SamHQFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

        self.hq_token = nn.Embedding(1, self.hidden_size)
        self.hq_mask_mlp = SamHQFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        # Compress ViT features
        self.compress_vit_conv1 = nn.ConvTranspose2d(config.vit_dim, self.hidden_size, kernel_size=2, stride=2)
        self.compress_vit_norm = SamHQLayerNorm(self.hidden_size, data_format="channels_first")
        self.compress_vit_conv2 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 8, kernel_size=2, stride=2)

        # Embedding encoder
        self.encoder_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.encoder_norm = SamHQLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.encoder_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)

        # Embedding mask feature
        self.mask_conv1 = nn.Conv2d(self.hidden_size // 8, self.hidden_size // 4, kernel_size=3, stride=1, padding=1)
        self.mask_norm = SamHQLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.mask_conv2 = nn.Conv2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        intermediate_embeddings: Optional[list[torch.Tensor]] = None,
        attention_similarity: Optional[torch.Tensor] = None,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> SamHQMMaskDecoderOutputs:
        """
        Predict high-quality masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                The embeddings from the image encoder.
            image_positional_embedding (`torch.Tensor`):
                Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes.
            dense_prompt_embeddings (`torch.Tensor`):
                The embeddings of the mask inputs.
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            hq_token_only (bool):
                Whether to use only the high-quality token output or combine with SAM output.
            intermediate_embeddings (`torch.Tensor`):
                Intermediate embeddings from the vision encoder for feature fusion.
            attention_similarity (`torch.Tensor`, *optional*):
                Optional tensor for attention similarity computation.
            target_embedding (`torch.Tensor`, *optional*):
                Optional target embedding for transformer processing.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple of tensors containing:
                - A tensor of shape `(batch_size, num_prompts, num_masks, height, width)` containing the output masks.
                - A tensor of shape `(batch_size, num_prompts, num_masks)` containing the iou predictions for each mask.
                - (Optional) A tuple containing attention tensors if output_attentions is True.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]

        has_intermediate = intermediate_embeddings is not None and len(intermediate_embeddings) > 0

        if has_intermediate:
            vit_features = intermediate_embeddings[0].permute(0, 3, 1, 2).contiguous()

        embed_encode = self.encoder_conv1(image_embeddings)
        embed_encode = self.activation(self.encoder_norm(embed_encode))
        embed_encode = self.encoder_conv2(embed_encode)

        if has_intermediate:
            compressed_vit_features = self.compress_vit_conv1(vit_features)
            compressed_vit_features = self.activation(self.compress_vit_norm(compressed_vit_features))
            compressed_vit_features = self.compress_vit_conv2(compressed_vit_features)

            hq_features = embed_encode + compressed_vit_features
        else:
            hq_features = embed_encode

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hq_token.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if torch.any(sparse_prompt_embeddings != 0):
            tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        point_embedding, iou_token_out = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        upscaled_embedding_hq = self.mask_conv1(upscaled_embedding)
        upscaled_embedding_hq = self.activation(self.mask_norm(upscaled_embedding_hq))
        upscaled_embedding_hq = self.mask_conv2(upscaled_embedding_hq)

        if hq_features.shape[0] == 1:
            hq_features = hq_features.repeat(batch_size * point_batch_size, 1, 1, 1)
        elif hq_features.shape[0] == batch_size and batch_size * point_batch_size != batch_size:
            hq_features = hq_features.repeat_interleave(point_batch_size, 0)
        upscaled_embedding_hq = upscaled_embedding_hq + hq_features

        hyper_in_list = []
        for mask_token_index in range(self.num_mask_tokens):
            if mask_token_index < self.num_mask_tokens - 1:
                current_mlp = self.output_hypernetworks_mlps[mask_token_index]
            else:
                current_mlp = self.hq_mask_mlp
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, mask_token_index, :])]

        hyper_in = torch.stack(hyper_in_list, dim=2)
        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        upscaled_embedding_hq = upscaled_embedding_hq.reshape(
            batch_size, point_batch_size, num_channels, height * width
        )

        masks_sam = (hyper_in[:, :, : self.num_mask_tokens - 1] @ upscaled_embedding).reshape(
            batch_size, point_batch_size, -1, height, width
        )
        masks_hq = (hyper_in[:, :, self.num_mask_tokens - 1 :] @ upscaled_embedding_hq).reshape(
            batch_size, point_batch_size, -1, height, width
        )
        masks = torch.cat([masks_sam, masks_hq], dim=2)

        iou_pred = self.iou_prediction_head(iou_token_out)

        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_pred = iou_pred[:, :, mask_slice]
            # Sort the IoU scores in descending order and get indices
            iou_pred_sorted, sort_indices = torch.sort(iou_pred, dim=2, descending=True)
            # Reorder the masks according to sorted scores
            masks_sam = masks[:, :, mask_slice, :, :]
            masks_sam = torch.gather(
                masks_sam,
                2,
                sort_indices[..., None, None].expand(-1, -1, -1, masks_sam.shape[3], masks_sam.shape[4]),
            )
            # Update iou_pred with sorted scores
            iou_pred = iou_pred_sorted
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, :, mask_slice]
            masks_sam = masks[:, :, mask_slice, :, :]

        masks_hq = masks[:, :, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]
        if hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq

        return masks, iou_pred


class SamHQVisionModel(SamVisionModel):
    pass


@auto_docstring(
    custom_intro="""
    Segment Anything Model HQ (SAM-HQ) for generating masks, given an input image and optional 2D location and bounding boxes.
    """
)
class SamHQModel(SamModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]
    _keys_to_ignore_on_load_missing = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = SamHQVisionEncoder(config.vision_config)

        self.mask_decoder = SamHQMaskDecoder(config.mask_decoder_config)

        self.post_init()

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
        """
        vision_output = self.vision_encoder(pixel_values=pixel_values)
        image_embeddings = vision_output[0]
        intermediate_embeddings = vision_output[1]
        return image_embeddings, intermediate_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        hq_token_only: bool = False,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        intermediate_embeddings: Optional[list[torch.FloatTensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[dict[str, torch.Tensor]]:
        r"""
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box
        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM_HQ model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        hq_token_only (`bool`, *optional*, defaults to `False`):
            Whether to use only the HQ token path for mask generation. When False, combines both standard and HQ paths.
            This is specific to SAM-HQ's architecture.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
        intermediate_embeddings (`List[torch.FloatTensor]`, *optional*):
            Intermediate embeddings from vision encoder's non-windowed blocks, used by SAM-HQ for enhanced mask quality.
            Required when providing pre-computed image_embeddings instead of pixel_values.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("sushmanth/sam_hq_vit_b")
        >>> processor = AutoProcessor.from_pretrained("sushmanth/sam_hq_vit_b")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get high-quality segmentation mask
        >>> outputs = model(**inputs)

        >>> # For high-quality mask only
        >>> outputs = model(**inputs, hq_token_only=True)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`."
                f" got {input_points.shape}."
            )

        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_boxes must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`."
                f" got {input_boxes.shape}."
            )

        # Add validation for point and box batch sizes
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    f"You should provide as many bounding boxes as input points per box. Got {point_batch_size} and {box_batch_size}."
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(pixel_values, **kwargs)
            image_embeddings = vision_outputs.last_hidden_state
            intermediate_embeddings = vision_outputs.intermediate_embeddings
        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        # Predict masks
        mask_decoder_output = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
            intermediate_embeddings=intermediate_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
        )
        return SamHQImageSegmentationOutput(
            iou_scores=mask_decoder_output[1],
            pred_masks=mask_decoder_output[0],
            vision_hidden_states=vision_outputs.hidden_states,
            vision_attentions=vision_outputs.attentions,
        )


__all__ = [
    "SamHQVisionConfig",
    "SamHQMaskDecoderConfig",
    "SamHQPromptEncoderConfig",
    "SamHQConfig",
    "SamHQModel",
    "SamHQPreTrainedModel",
    "SamHQVisionModel",
]
