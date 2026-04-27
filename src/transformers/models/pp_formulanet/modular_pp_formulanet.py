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
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    ImageInput,
)
from ...modeling_outputs import (
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import (
    ProcessingKwargs,
    Unpack,
)
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.import_utils import requires
from ..florence2.modeling_florence2 import Florence2ForConditionalGeneration, Florence2Model
from ..mbart.modeling_mbart import MBartDecoderWrapper
from ..nougat.image_processing_nougat import NougatImageProcessor
from ..nougat.processing_nougat import NougatProcessor
from ..slanext.configuration_slanext import SLANeXtVisionConfig
from ..slanext.modeling_slanext import (
    SLANeXtPreTrainedModel,
    SLANeXtVisionAttention,
    SLANeXtVisionEncoder,
)


logger = logging.get_logger(__name__)


@auto_docstring(
    checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors"
)  # or "PaddlePaddle/PP-FormulaNet-L_safetensors"
@strict
class PPFormulaNetVisionConfig(SLANeXtVisionConfig):
    pass


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetTextConfig(PreTrainedConfig):
    r"""
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

    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 1024
    post_conv_mid_channels: int = 512
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
    is_encoder_decoder: bool = True


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetConfig(PreTrainedConfig):
    r"""
    vision_config (`dict` or [`PPFormulaNetVisionConfig`], *optional*):
        Configuration for the vision encoder. If `None`, a default [`PPFormulaNetVisionConfig`] is used.
    """

    model_type = "pp_formulanet"
    sub_configs = {"text_config": PPFormulaNetTextConfig, "vision_config": PPFormulaNetVisionConfig}

    text_config: dict | PPFormulaNetTextConfig | None = None
    vision_config: dict | PPFormulaNetVisionConfig | None = None
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = PPFormulaNetTextConfig(**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None. Initializing the PPFormulaNetTextConfig with default values.")
            self.text_config = PPFormulaNetTextConfig()

        if isinstance(self.vision_config, dict):
            self.vision_config = PPFormulaNetVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            logger.info("vision_config is None. Initializing the PPFormulaNetVisionConfig with default values.")
            self.vision_config = PPFormulaNetVisionConfig()

        super().__post_init__(**kwargs)


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
    """

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
        return BatchFeature({"input_ids": None, **image_inputs})

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
            logger.warning_once(
                "ftfy is not installed, skipping fix_text. "
                "Output may contain unnormalized unicode, extra spaces, or escaped artifacts"
            )
        text = self.normalize(text)
        return text

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
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


@dataclass
class PPFormulaNetSeq2SeqModelOutput(Seq2SeqModelOutput):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None


class PPFormulaNetVisionAttention(SLANeXtVisionAttention):
    pass


class PPFormulaNetVisionEncoder(SLANeXtVisionEncoder):
    pass


class PPFormulaNetMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(
            config.post_conv_in_channels, config.post_conv_mid_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            config.post_conv_mid_channels,
            config.post_conv_out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.linear_1 = nn.Linear(config.post_conv_out_channels, config.post_conv_out_channels)
        self.linear_2 = nn.Linear(config.post_conv_out_channels, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PPFormulaNetVisionModel(SLANeXtVisionEncoder):
    pass


class PPFormulaNetTextModel(MBartDecoderWrapper):
    pass


class PPFormulaNetModel(Florence2Model):
    def __init__(self, config):
        super().__init__(config)

        self.language_model = PPFormulaNetTextModel(config.text_config)
        self.vision_tower = PPFormulaNetVisionModel(config=config.vision_config)
        self.multi_modal_projector = PPFormulaNetMultiModalProjector(config.text_config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        encoder_outputs: list[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | PPFormulaNetSeq2SeqModelOutput:
        if encoder_outputs is None:
            if pixel_values is not None:
                encoder_outputs = self.get_image_features(pixel_values)
            image_features = encoder_outputs.pooler_output.to(self.language_model.device, self.language_model.dtype)
        else:
            image_features = self.multi_modal_projector(encoder_outputs.last_hidden_state)

        if decoder_input_ids is None:
            decoder_start_token_id = self.config.text_config.decoder_start_token_id
            decoder_input_ids = torch.ones(
                (image_features.size()[0], 1), dtype=torch.long, device=self.language_model.device
            )
            decoder_input_ids *= decoder_start_token_id

        decoder_outputs = self.language_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=image_features,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return PPFormulaNetSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def get_encoder(self):
        return self.vision_tower


@auto_docstring(
    custom_intro="""
    PPFormulaNet Table Recognition model for table recognition tasks. Wraps the core PPFormulaNetPreTrainedModel
    and returns outputs compatible with the Transformers table recognition API.
    """
)
class PPFormulaNetForConditionalGeneration(Florence2ForConditionalGeneration):
    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        return GenerationMixin._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

    def get_encoder(self):
        return self.model.vision_tower


__all__ = [
    "PPFormulaNetProcessor",
    "PPFormulaNetImageProcessor",
    "PPFormulaNetConfig",
    "PPFormulaNetTextConfig",
    "PPFormulaNetModel",
    "PPFormulaNetVisionConfig",
    "PPFormulaNetForConditionalGeneration",
    "PPFormulaNetPreTrainedModel",
]
