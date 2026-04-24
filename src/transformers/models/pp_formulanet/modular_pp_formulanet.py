# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import DynamicCache, EncoderDecoderCache
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    ImageInput,
)
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import (
    ProcessingKwargs,
    Unpack,
)
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import requires
from ..mbart.modeling_mbart import MBartForCausalLM
from ..nougat.image_processing_nougat import NougatImageProcessor
from ..nougat.processing_nougat import NougatProcessor
from ..slanext.configuration_slanext import SLANeXtConfig
from ..slanext.modeling_slanext import (
    SLANeXtBackbone,
    SLANeXtPreTrainedModel,
    SLANeXtVisionAttention,
    SLANeXtVisionEncoder,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetConfig(SLANeXtConfig):
    r"""
    vision_config (`dict` or [`PPFormulaNetVisionConfig`], *optional*):
        Configuration for the vision encoder. If `None`, a default [`PPFormulaNetVisionConfig`] is used.
    post_conv_in_channels (`int`, *optional*, defaults to 256):
        Number of input channels for the post-encoder convolution layer.
    post_conv_mid_channels (`int`, *optional*, defaults to 512):
       Number of intermediate channels for the post-encoder convolution layer.
    post_conv_out_channels (`int`, *optional*, defaults to 1024):
        Number of output channels for the post-encoder convolution layer.
    max_length (`int`, *optional*, defaults to 1537):
        Controls the maximum length to use by one of the truncation/padding parameters.
    """

    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    out_channels = AttributeError()
    hidden_size = AttributeError()
    max_text_length = AttributeError()

    post_conv_in_channels: int = 256
    post_conv_mid_channels: int = 512
    post_conv_out_channels: int = 1024
    vocab_size: int = 50000
    max_position_embeddings: int = 2560
    encoder_layers: int = 12
    encoder_attention_heads: int = 16
    decoder_layers: int = 8
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 16
    decoder_layerdrop: float | int = 0.0
    activation_function: str = "gelu"
    d_model: int = 512
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    scale_embedding: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    decoder_start_token_id: int | None = 2
    forced_eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    max_length: int = 1537


@auto_docstring
@requires(backends=("torch",))
class PPFormulaNetImageProcessor(NougatImageProcessor):
    image_mean = [0.7931, 0.7931, 0.7931]
    image_std = [0.1738, 0.1738, 0.1738]
    size = {"height": 768, "width": 768}


class PPFormulaNetProcessor(NougatProcessor):
    r"""
    [`PPFormulaNetProcessor`] offers all the functionalities of [`PPFormulaNetImageProcessor`] and [`NougatTokenizer`]. See the
    [`~PPFormulaNetProcessor.__call__`] and [`~PPFormulaNetProcessor.decode`] for more information.
    Args:
        image_processor ([`PPFormulaNetImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`NougatTokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        images: ImageInput,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        """
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
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
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            ProcessingKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        return image_inputs

    def normalize(self, s: str) -> str:
        """Normalizes a string by removing unnecessary spaces.

        Args:
            s (str): String to normalize.

        Returns:
            str: Normalized string.
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = r"[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = []
        for x in re.findall(text_reg, s):
            pattern = r"\\[a-zA-Z]+"
            pattern = r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})"
            matches = re.findall(pattern, x[0])
            for m in matches:
                if (
                    m
                    not in [
                        "\\operatorname",
                        "\\mathrm",
                        "\\text",
                        "\\mathbf",
                    ]
                    and m.strip() != ""
                ):
                    s = s.replace(m, m + "XXXXXXX")
                    s = s.replace(" ", "")
                    names.append(s)
        if len(names) > 0:
            s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s.replace("XXXXXXX", " ")

    def remove_chinese_text_wrapping(self, formula):
        pattern = re.compile(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}")

        def replacer(match):
            return match.group(1)

        replaced_formula = pattern.sub(replacer, formula)
        return replaced_formula.replace('"', "")

    def post_process_generation(self, text: str) -> str:
        """Post-processes a string by fixing text and normalizing it.

        Args:
            text (str): String to post-process.

        Returns:
            str: Post-processed string.
        """
        text = self.remove_chinese_text_wrapping(text)
        try:
            from ftfy import fix_text

            text = fix_text(text)
        except ImportError:
            logger.warning(
                "ftfy is not installed, skipping fix_text. "
                "Output may contain unnormalized unicode, extra spaces, or escaped artifacts"
            )
        text = self.normalize(text)
        return text

    def post_process(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        generated_texts = self.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)
        return [self.post_process_generation(text) for text in generated_texts]


class PPFormulaNetPreTrainedModel(SLANeXtPreTrainedModel):
    _keep_in_fp32_modules_strict = []

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(module)

        # Initialize positional embeddings to zero (PPFormulaNetVisionEncoder holds pos_embed)
        if isinstance(module, PPFormulaNetVisionEncoder):
            if module.pos_embed is not None:
                init.constant_(module.pos_embed, 0.0)

        # Initialize relative positional embeddings to zero (PPFormulaNetVisionAttention holds rel_pos_h/w)
        if isinstance(module, PPFormulaNetVisionAttention):
            if module.use_rel_pos:
                init.constant_(module.rel_pos_h, 0.0)
                init.constant_(module.rel_pos_w, 0.0)


class PPFormulaNetBackbone(SLANeXtBackbone):
    def __init__(
        self,
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config)
        del self.post_conv
        self.post_conv1 = nn.Conv2d(
            config.post_conv_in_channels, config.post_conv_mid_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.post_conv2 = nn.Conv2d(
            config.post_conv_mid_channels,
            config.post_conv_out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.mm_projector_vary = nn.Linear(config.post_conv_out_channels, config.post_conv_out_channels)
        self.enc_to_dec_proj = nn.Linear(config.post_conv_out_channels, config.hidden_size)

        self.post_init()

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]):
        vision_output = self.vision_tower(hidden_states, **kwargs)
        hidden_states = self.post_conv1(vision_output.last_hidden_state)
        hidden_states = self.post_conv2(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.mm_projector_vary(hidden_states)
        hidden_states = self.enc_to_dec_proj(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=vision_output.hidden_states,
            attentions=vision_output.attentions,
        )


class PPFormulaNetVisionAttention(SLANeXtVisionAttention):
    pass


class PPFormulaNetVisionEncoder(SLANeXtVisionEncoder):
    pass


class PPFormulaNetHead(MBartForCausalLM):
    pass


@dataclass
@auto_docstring
class PPFormulaNetForTableRecognitionOutput(BaseModelOutput):
    r"""
    head_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Hidden-states of the PPFormulaNetSLAHead at each prediction step, varies up to max `self.config.max_text_length` states (depending on early exits).
    head_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Attentions of the PPFormulaNetSLAHead at each prediction step, varies up to max `self.config.max_text_length` attentions (depending on early exits).
    """

    head_hidden_states: torch.FloatTensor | None = None
    head_attentions: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    PPFormulaNet Table Recognition model for table recognition tasks. Wraps the core PPFormulaNetPreTrainedModel
    and returns outputs compatible with the Transformers table recognition API.
    """
)
class PPFormulaNetForTextRecognition(PPFormulaNetPreTrainedModel):
    def __init__(self, config: PPFormulaNetConfig):
        super().__init__(config)
        self.backbone = PPFormulaNetBackbone(config=config)
        self.head = PPFormulaNetHead(config=config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | PPFormulaNetForTableRecognitionOutput:
        backbone_outputs = self.backbone(pixel_values, **kwargs)
        encoder_hidden_states = backbone_outputs.last_hidden_state

        # Start generation from decoder BOS with shape [batch_size, 1].
        batch_size = encoder_hidden_states.shape[0]
        input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        # In this decoder-only `generate` path we still use cross-attention via `encoder_hidden_states`, but
        # `GenerationMixin` auto-creates a plain `DynamicCache` by default. Explicitly passing
        # `EncoderDecoderCache(self_cache, cross_cache)` keeps self-attn and cross-attn cache lengths separated,
        # avoiding decoder position-length contamination/overflow. This is a local, minimal fix and does not change
        # MBart architecture or rewrite decoder forward logic.
        past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
        head_outputs = self.head.generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            max_length=self.config.max_length,
            return_dict_in_generate=True,
            **kwargs,
        )
        return PPFormulaNetForTableRecognitionOutput(
            last_hidden_state=head_outputs.sequences,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
            head_hidden_states=head_outputs.hidden_states,
            head_attentions=head_outputs.attentions,
        )


__all__ = [
    "PPFormulaNetProcessor",
    "PPFormulaNetImageProcessor",
    "PPFormulaNetConfig",
    "PPFormulaNetBackbone",
    "PPFormulaNetForTextRecognition",
    "PPFormulaNetPreTrainedModel",
    "PPFormulaNetHead",
]
