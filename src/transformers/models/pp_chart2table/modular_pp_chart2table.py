from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as tvF

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images
from ...modeling_rope_utils import RopeParameters
from ..got_ocr2.modeling_got_ocr2 import (
    GotOcr2ModelOutputWithPast,
    GotOcr2Model,
    GotOcr2PreTrainedModel,
    GotOcr2ForConditionalGeneration,
    GotOcr2VisionEncoder,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2PreTrainedModel,
)
from ...utils import (
    auto_docstring,
    logging,
    TransformersKwargs,
)
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import ProcessorMixin, TensorType, Unpack

from ...image_utils import SizeDict

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-Chart2Table_safetensors",)
class PPChart2TableVisionConfig(PreTrainedConfig):
    """
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of transformer encoder layers in the vision backbone.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the patch embedding vectors.
    num_channels (`int`, *optional*, defaults to 3):
        Number of input channels (3 for RGB images, 1 for grayscale).
    image_size (`int`, *optional*, defaults to 1024):
        Size (height/width) of the input images (assumed to be square).
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each transformer encoder layer.
    patch_size (`int`, *optional*, defaults to 16):
        Size (height/width) of the image patches extracted from the input image.
    qkv_bias (`bool`, *optional*, defaults to `True`):
        Whether to include bias terms in the query, key, value projection layers of self-attention.
    use_rel_pos (`bool`, *optional*, defaults to `True`):
        Whether to use relative positional embeddings in the self-attention mechanism.
    global_attn_indexes (`Optional[list[int]]`, *optional*, defaults to [2, 5, 8, 11]):
        List of layer indexes where global attention (instead of window attention) is applied.
        If `None`, defaults to [2, 5, 8, 11].
    window_size (`int`, *optional*, defaults to 14):
        Size of the attention window for window-based self-attention (only effective when use_rel_pos=True).
    output_channels (`int`, *optional*, defaults to 256):
        Dimensionality of the final visual feature output channels.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability applied to the attention weights.
    """

    model_type = "pp_chart2table_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        num_hidden_layers=12,
        hidden_size=768,
        output_channels=256,
        mlp_dim=3072,
        num_channels=3,
        image_size=1024,
        num_attention_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        use_abs_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.mlp_dim=mlp_dim
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.use_abs_pos = use_abs_pos
        self.global_attn_indexes = global_attn_indexes
        self.window_size = window_size
        self.output_channels = output_channels
        self.attention_dropout = attention_dropout
        super().__init__(**kwargs)


@auto_docstring(
    custom_intro="""
    
    """,

)
class PPChart2TableTextConfig(PreTrainedConfig):
    r"""
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities in self-attention layers.
    bos_token_id (`int`, *optional*, defaults to 151643):
        The token ID representing the beginning of a sequence (BOS) for text generation.
    eos_token_id (`int`, *optional*, defaults to 151643):
        The token ID representing the end of a sequence (EOS) for text generation.
    pad_token_id (Optional[int], optional, *optional*, defaults to -1):
        The index of the padding token. Defaults to -1.
    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the feed-forward and attention layers of the decoder.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the hidden representations in the Transformer decoder layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    intermediate_size (`int`, *optional*, defaults to 2816):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer decoder blocks.
    max_position_embeddings (`int`, *optional*, defaults to 32768):
        The maximum sequence length that this model might ever be used with for text input/output.
    num_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each self-attention layer in the Transformer decoder.
    num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer decoder.
    num_key_value_heads (`int`, *optional*, defaults to 16):
        Number of key/value heads for implementing Grouped Query Attention (GQA). If equal to `num_attention_heads`,
        Multi Head Attention (MHA) is used; if 1, Multi Query Attention (MQA) is used. For more details, see
        [this paper](https://huggingface.co/papers/2305.13245).
    rms_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon value used by the RMS normalization layers to avoid division by zero.
    rope_theta (`float`, *optional*, defaults to 1000000.0):
        The base period of the RoPE (Rotary Position Embedding) embeddings, controlling the frequency of positional encoding.
    rope_parameters (`RopeParameters` or `dict`, *optional*):
        Configuration parameters for RoPE embeddings, including scaling parameters for longer sequence lengths beyond
        `max_position_embeddings`.
    sliding_window (`int`, *optional*, defaults to 32768):
        Window size for Sliding Window Attention (SWA) in the decoder layers (only active if `use_sliding_window=True`).
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether the model's input and output word embeddings should be tied (shared weights).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether to return the last key/value attention states to speed up sequential decoding (only relevant for autoregressive
        generation).
    vocab_size (`int`, *optional*, defaults to 151860):
        Vocabulary size of the PPChart2TableText model. Defines the number of distinct tokens that can be represented
        by `input_ids`.
    layer_types (`list[str]`, *optional*):
        Attention pattern for each decoder layer (e.g., `"full_attention"` or `"sliding_attention"`). If not specified,
        automatically determined by `sliding_window`.
    """

    model_type = "pp_chart2table_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `PPChart2TableText`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_config_key = "text_config"

    def __init__(
        self,
        attention_dropout: float = 0.0,
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        pad_token_id: int = -1,
        hidden_act: str = "silu",
        hidden_size: int = 1024,
        initializer_range: float = 0.02,
        intermediate_size: int = 2816,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        num_key_value_heads: int = 16,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        sliding_window: int = 32768,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        vocab_size: int = 151860,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.attention_dropout = attention_dropout

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.sliding_window is not None else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters

        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    
    """
)
class PPChart2TableConfig(PreTrainedConfig):
    r"""
    vision_config (Optional[Dict], optional, *optional*)::
        The [PPChart2TableVisionConfig] for the vision sub-model. Defaults to None.
    text_config (Optional[Dict], optional, *optional*)::
        The [PPChart2TableTextConfig] for the text sub-model. Defaults to None.
    image_token_index (Optional[int], optional, *optional*, defaults to 151859)::
        The index of the image token. Defaults to 151859.
    image_seq_length (Optional[int], optional, *optional*, defaults to 576)::
        The sequence length for the image. Defaults to 576.
    pad_token_id (Optional[int], optional, *optional*, defaults to -1):
        The index of the padding token. Defaults to -1.
    net_channels (`int`, *optional*, defaults to 512):
        Dimensionality of intermediate network channels in the vision backbone.
    output_channels (`int`, *optional*, defaults to 1024):
        Dimensionality of intermediate network channels in the vision backbone.
    """

    model_type = "pp_chart2table"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"vision_config": PPChart2TableVisionConfig, "text_config": PPChart2TableTextConfig}

    def __init__(
        self,
        vision_config: dict | None = None,
        text_config: dict | None = None,
        image_token_index: Optional[int] = 151859,
        image_seq_length: Optional[int] = 576,
        pad_token_id: Optional[int] = -1,
        net_channels: Optional[int] = 512,
        output_channels: Optional[int] = 1024,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.image_seq_length = image_seq_length
        self.pad_token_id = pad_token_id
        self.net_channels = net_channels
        self.output_channels = output_channels

        if vision_config is None:
            vision_config = {}
        self.vision_config = PPChart2TableVisionConfig(**vision_config)

        if text_config is None:
            text_config = {}
        self.text_config = PPChart2TableTextConfig(**text_config)

        text_config_keys = [
            "attention_dropout",
            "bos_token_id",
            "eos_token_id",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "sliding_window",
            "tie_word_embeddings",
            "dtype",
            "use_cache",
            "vocab_size",
        ]
        for key in text_config_keys:
            if hasattr(self.text_config, key):
                setattr(self, key, getattr(self.text_config, key))

        super().__init__(**kwargs)


@auto_docstring
class PPChart2TableImageProcessorFast(BaseImageProcessorFast):

    resample = 3
    image_mean = [0.40821073, 0.4578275, 0.48145466]
    image_std = [0.27577711, 0.26130258, 0.26862954]
    size = {"height": 1024, "width": 1024}
    patch_size = 16
    merge_size = 4
    do_resize = True
    do_rescale = True
    do_normalize = True

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            # BGR to RGB conversion
            stacked_images = stacked_images[:, [2, 1, 0], :, :]
            processed_images_grouped[shape] = stacked_images

        pixel_values = reorder_images(processed_images_grouped, grouped_images_index)

        return BatchFeature(
            data={"pixel_values": pixel_values},
            tensor_type=return_tensors,
        )


@auto_docstring
class PPChart2TableProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images,
        text=None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
        else:
            image_inputs = {}
        img_cnt = len(image_inputs)
        _, _, height, _ = image_inputs["pixel_values"].shape
        num_patches = height // self.image_processor.patch_size // self.image_processor.merge_size
        prompt = (
            "<|im_start|>system\n"
            "You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n"
            "<img>" + "<imgpad>" * (num_patches * num_patches) + "</img>\n"
            "Chart to table<|im_end|><|im_start|>assistant\n"
        )
        input_ids = torch.tensor(self.tokenizer([prompt]).input_ids)
        input_ids = input_ids.repeat(img_cnt, 1)
        input_ids = {"input_ids": input_ids}
        return BatchFeature(data={**input_ids, **image_inputs})

    def postprocess(self, model_pred, **kwargs):
        return self.tokenizer.batch_decode(
            model_pred[0],
            skip_special_tokens=kwargs.get("skip_special_tokens", True),
            clean_up_tokenization_spaces=False,
        )


class PPChart2TableVisionPreTrainedModel(GotOcr2PreTrainedModel):
    input_modalities = ("image", "text")


class PPChart2TableVisionEncoder(GotOcr2VisionEncoder, PPChart2TableVisionPreTrainedModel):
    pass



@auto_docstring
class PPChart2TableTextPreTrainedModel(Qwen2PreTrainedModel):
    pass


class PPChart2TableTextModel(Qwen2Model):
    pass


@dataclass
class PPChart2TableModelOutputWithPast(GotOcr2ModelOutputWithPast):
    pass


@auto_docstring
class PPChart2TableModel(GotOcr2Model):

    def __init__(self, config: PPChart2TableConfig):
        super().__init__(config)
        self.vision_downsample1 = nn.Conv2d(config.vision_config.output_channels, config.net_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.vision_downsample2 = nn.Conv2d(config.net_channels, config.output_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.language_model = PPChart2TableTextModel._from_config(config.text_config)
        self.multi_modal_projector = nn.Linear(config.output_channels, config.text_config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Get input embeddings from the text decoder (for weight tying/loading)."""
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings for the text decoder (for weight tying/loading)."""
        self.language_model.embed_tokens = value

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        
        image_output = self.vision_tower(pixel_values)
        last_hidden_state = image_output.last_hidden_state
        last_hidden_state = self.vision_downsample1(last_hidden_state)
        last_hidden_state = self.vision_downsample2(last_hidden_state)
        image_output.pooler_output = self.multi_modal_projector(last_hidden_state.flatten(2).transpose(2, 1))

        return image_output


@auto_docstring(
    custom_intro="""
    PP-Chart2Table model for conditional generation (table text generation from chart images),
    extending the core model with a language modeling (LM) head and generation utilities.
    """
)
class PPChart2TableForConditionalGeneration(GotOcr2ForConditionalGeneration):
    pass


__all__ = [
    "PPChart2TableForConditionalGeneration",
    "PPChart2TableModel",
    "PPChart2TablePreTrainedModel",
    "PPChart2TableConfig",
    "PPChart2TableTextPreTrainedModel",
    "PPChart2TableTextModel",
    "PPChart2TableVisionPreTrainedModel",
    "PPChart2TableVisionModel",
    "PPChart2TableVisionConfig",
    "PPChart2TableTextConfig",
    "PPChart2TableImageProcessorFast",
    "PPChart2TableProcessor",
]
