# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
from typing import ClassVar, List, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from ..paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ColPaliConfig"


class ColPaliConfig(PaliGemmaConfig):
    r"""
    This is the configuration class to store the configuration of a [`ColPaliModel`]. It is used to instantiate an
    ColPaliModel according to the specified arguments, defining the model architecture.

    The ColPali config is stricly equivalent to the PaliGemma config, but with a different model type.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "colpali"
        self.is_composition = False


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

    return device


@dataclass
class ColPaliOutput(ModelOutput):
    """
    Base class for ColPali embeddings output.

    Args:
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The embeddings of the model.
    """

    embeddings: torch.Tensor
    loss: Optional[torch.FloatTensor] = None


COLPALI_START_DOCSTRING = r"""
    ColPali is a PaliGemma variant to produce multi-vector representations from images.
    It was introduced in the paper [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449).

    ### Resources
    - A blog post detailing ColPali, a vision retrieval model, can be found [here](https://huggingface.co/blog/manu/colpali). 🌎
    - The training codebase for ColPali can be found [here](https://github.com/illuin-tech/colpali). 🌎
"""

COLPALI_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`PaliGemmaProcessor`] uses
            [`SiglipImageProcessor`] for processing images). If none, ColPali will only process text (query embeddings).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
"""


@add_start_docstrings(
    COLPALI_START_DOCSTRING,
    "Adapter from colpali-engine==0.3.0: https://github.com/illuin-tech/colpali.",
)
class ColPaliModel(PaliGemmaPreTrainedModel):
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config=config)

        model = PaliGemmaForConditionalGeneration(config=config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model

        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)

        self.post_init()

    @add_start_docstrings_to_model_forward(COLPALI_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ColPaliOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids,
            pixel_values,
            attention_mask,
            position_ids,
            past_key_values,
            token_type_ids,
            cache_position,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            num_logits_to_keep,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        proj = proj * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return proj

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings

        return model_embeds
