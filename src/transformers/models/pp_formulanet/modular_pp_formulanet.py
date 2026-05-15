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
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    ImageInput,
)
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    Seq2SeqLMOutput,
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
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..florence2.modeling_florence2 import Florence2ForConditionalGeneration
from ..mbart.configuration_mbart import MBartConfig
from ..mbart.modeling_mbart import MBartDecoder, shift_tokens_right
from ..nougat.image_processing_nougat import NougatImageProcessor
from ..nougat.processing_nougat import NougatProcessor
from ..slanext.configuration_slanext import SLANeXtVisionConfig
from ..slanext.modeling_slanext import (
    SLANeXtPreTrainedModel,
    SLANeXtVisionAttention,
    SLANeXtVisionEncoder,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetVisionConfig(SLANeXtVisionConfig):
    r"""
    output_channels (`int`, *optional*, defaults to 256):
        Dimensionality of the output channels in the Patch Encoder.
    window_size (`int`, *optional*, defaults to 14):
        Window size for relative position.
    global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
        The indexes of the global attention layers.
    mlp_dim (`int`, *optional*, defaults to 3072):
        The dimensionality of the MLP layer in the Transformer encoder.
    post_conv_in_channels (`int`, *optional*, defaults to 256):
        Number of input channels for the post-encoder convolution layer.
    post_conv_out_channels (`int`, *optional*, defaults to 1024):
        Number of output channels for the post-encoder convolution layer.
    post_conv_mid_channels (`int`, *optional*, defaults to 512):
        Number of intermediate channels for the post-encoder convolution layer.
    decoder_hidden_size (`int`, *optional*, defaults to 512):
        The hidden size of the decoder that the encoder features are projected to.
    """

    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 1024
    post_conv_mid_channels: int = 512
    decoder_hidden_size: int = 512


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetTextConfig(MBartConfig):
    base_config_key = "text_config"
    vocab_size: int = 50000
    max_position_embeddings: int = 2560
    decoder_layers: int = 8
    decoder_ffn_dim: int = 2048
    d_model: int = 512
    scale_embedding: bool = True
    decoder_start_token_id: int | None = 2
    tie_word_embeddings: bool = False
    is_encoder_decoder: bool = True
    classifier_dropout = AttributeError()
    encoder_ffn_dim = AttributeError()
    encoder_layerdrop = AttributeError()
    is_decoder = AttributeError()


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetConfig(PreTrainedConfig):
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


@auto_docstring
class PPFormulaNetProcessor(NougatProcessor):
    r"""
    [`PPFormulaNetProcessor`] offers all the functionalities of [`PPFormulaNetImageProcessor`] and [`NougatTokenizer`]. See the
    [`~PPFormulaNetProcessor.__call__`] and [`~PPFormulaNetProcessor.decode`] for more information.
    """

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

        # normalize() regex
        self._text_reg = re.compile(r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})")
        self._macro_pattern = re.compile(r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})")
        self._protected_macros = {"\\operatorname", "\\mathrm", "\\text", "\\mathbf"}

        letter = r"[a-zA-Z]"
        noletter = r"[\W_^\d]"
        self._rule_noletter_noletter = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter))
        self._rule_noletter_letter = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter))
        self._rule_letter_noletter = re.compile(r"(%s)\s+?(%s)" % (letter, noletter))

        # remove_chinese_text_wrapping() regex
        self._chinese_text_wrapping_pattern = re.compile(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}")

    def __call__(
        self,
        images: ImageInput,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        r"""
        images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
            The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
            tensor. Both channels-first and channels-last formats are supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            ProcessingKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        return BatchFeature({**image_inputs})

    def normalize(self, text: str) -> str:
        """Normalizes a string by removing unnecessary spaces."""
        names = []
        for x in self._text_reg.findall(text):
            matches = self._macro_pattern.findall(x[0])
            for m in matches:
                if m not in self._protected_macros and m.strip() != "":
                    text = text.replace(m, m + "XXXXXXX")
                    text = text.replace(" ", "")
                    names.append(text)

        if names:
            text = self._text_reg.sub(lambda match: str(names.pop(0)), text)

        new_text = text
        while True:
            text = new_text
            new_text = self._rule_noletter_noletter.sub(r"\1\2", text)
            new_text = self._rule_noletter_letter.sub(r"\1\2", new_text)
            new_text = self._rule_letter_noletter.sub(r"\1\2", new_text)
            if new_text == text:
                break

        return new_text.replace("XXXXXXX", " ")

    def remove_chinese_text_wrapping(self, formula: str) -> str:
        def replacer(match):
            return match.group(1)

        replaced_formula = self._chinese_text_wrapping_pattern.sub(replacer, formula)
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
    base_model_prefix = "model"
    # Note this goes for the decoder only, the encoder will inherently always use eager attention
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(module)

        # Initialize positional embeddings to zero (PPFormulaNetVisionModel holds pos_embed)
        if isinstance(module, PPFormulaNetVisionModel):
            if module.pos_embed is not None:
                init.constant_(module.pos_embed, 0.0)

        # Initialize relative positional embeddings to zero (PPFormulaNetVisionAttention holds rel_pos_h/w)
        if isinstance(module, PPFormulaNetVisionAttention):
            if module.use_rel_pos:
                init.constant_(module.rel_pos_h, 0.0)
                init.constant_(module.rel_pos_w, 0.0)


# overrider for PPFormulaNetModel's encoder output
@dataclass
class PPFormulaNetVisionEncoderOutput(BaseModelOutputWithPooling):
    pass


class PPFormulaNetVisionAttention(SLANeXtVisionAttention):
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
        self.linear_2 = nn.Linear(config.post_conv_out_channels, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PPFormulaNetVisionModel(SLANeXtVisionEncoder):
    def __init__(self, config: PPFormulaNetVisionConfig):
        super().__init__()
        self.multi_modal_projector = PPFormulaNetMultiModalProjector(config)

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self, pixel_values: torch.FloatTensor | None = None, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        hidden_states = self.neck(hidden_states)
        pooler_output = self.multi_modal_projector(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
        )


@auto_docstring
class PPFormulaNetTextModel(MBartDecoder):
    pass


class PPFormulaNetModel(PPFormulaNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.decoder = PPFormulaNetTextModel(config.text_config)
        self.encoder = PPFormulaNetVisionModel(config=config.vision_config)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor
        | None = None,  # Kept in the signature for compatibility to avoid duplicate-keyword errors.
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        encoder_outputs: list[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor
        | None = None,  # Kept in the signature for compatibility to avoid duplicate-keyword errors.
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | Seq2SeqModelOutput:
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.text_config.pad_token_id)

        if (encoder_outputs is None) ^ (pixel_values is not None):
            raise ValueError("You must specify exactly one of encoder_outputs or pixel_values")

        if encoder_outputs is None:
            encoder_outputs: BaseModelOutputWithPooling = self.encoder(
                pixel_values=pixel_values,
                **kwargs,
            )
        elif not isinstance(encoder_outputs, BaseModelOutputWithPooling):
            encoder_outputs = BaseModelOutputWithPooling(
                last_hidden_state=encoder_outputs[0],
                pooler_output=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                hidden_states=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.pooler_output,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@auto_docstring
class PPFormulaNetForConditionalGeneration(Florence2ForConditionalGeneration):
    _tied_weights_keys = {}

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: list[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Seq2SeqLMOutput:
        r"""
        Example:

        ```python
        >>> from io import BytesIO

        >>> import httpx
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, PPFormulaNetForConditionalGeneration

        >>> model_path = "PaddlePaddle/PP-FormulaNet_plus-L_safetensors" # or "PaddlePaddle/PP-FormulaNet-L_safetensors"
        >>> model = PPFormulaNetForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        >>> processor = AutoProcessor.from_pretrained(model_path)

        >>> image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png"
        >>> image = Image.open(BytesIO(httpx.get(image_url).content)).convert("RGB")
        >>> inputs = processor(images=image, return_tensors="pt").to(model.device)
        >>> outputs = model(**inputs)
        >>> result = processor.post_process(outputs)
        >>> print(result)
        ['\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega\\int_{C_{+}}d z\\frac{2z^{2}}{(z^{2}+\\omega^{2})^{\\nu+1}}\\breve{\\Psi}(\\omega;z)e^{i\\epsilon z}\\quad,']
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
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

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    # override this function to compatible with `_prepare_encoder_decoder_kwargs_for_generation`
    def get_encoder(self):
        return self.model.get_encoder()

    def get_input_embeddings(self):
        raise AttributeError("The PPFormulaNetModel does not have an `input_embedding` attribute.")

    def set_input_embeddings(self):
        raise AttributeError("The PPFormulaNetModel does not have an `input_embedding` attribute.")

    def get_placeholder_mask(self):
        raise AttributeError("The PPFormulaNet does not need placeholder mask.")

    def get_image_features(self):
        raise AttributeError("The PPFormulaNet does not need `get_image_features`.")

    def _prepare_encoder_decoder_kwargs_for_generation(self):
        raise AttributeError("The PPFormulaNet use default implementation.")


__all__ = [
    "PPFormulaNetProcessor",
    "PPFormulaNetImageProcessor",
    "PPFormulaNetConfig",
    "PPFormulaNetTextConfig",
    "PPFormulaNetModel",
    "PPFormulaNetTextModel",
    "PPFormulaNetVisionModel",
    "PPFormulaNetVisionConfig",
    "PPFormulaNetForConditionalGeneration",
    "PPFormulaNetPreTrainedModel",
]
