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
from typing import ClassVar, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn

from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from ..paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)


class ColPaliConfig(PaliGemmaConfig):
    r"""
    This is the configuration class to store the configuration of a [`ColPaliForRetrieval`]. It is used to instantiate an
    ColPaliForRetrieval according to the specified arguments, defining the model architecture.

    The ColPali config is stricly equivalent to the PaliGemma config, but with a different model type.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "colpali"
        self.is_composition = False
        self.embedding_dim = 128


class ColPaliProcessor(PaliGemmaProcessor):
    r"""
    Processor for ColPali.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_image = Image.new("RGB", (16, 16), color="black")

    @staticmethod
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

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali.
        """
        texts_doc = ["Describe the image."] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPali.
        """
        if suffix is None:
            suffix = "<pad>" * 10
        texts_query: List[str] = []

        for query in queries:
            query = f"Question: {query}"
            query += suffix  # add suffix (pad tokens)
            texts_query.append(query)

        batch_query = self(
            images=[self.mock_image] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.image_seq_length,
        )

        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][..., self.image_seq_length :]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., self.image_seq_length :]

        return batch_query

    def post_process_retrieval(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        device = device or self.get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores


@dataclass
class ColPaliModelOutput(PaliGemmaForConditionalGeneration):
    """
    Base class for ColPali embeddings output.

    Args:
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The embeddings of the model.
    """

    embeddings: torch.Tensor = None


@add_start_docstrings(
    """
    ColPali is a PaliGemma variant to produce multi-vector representations from images.
    It was introduced in the paper [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449).

    Resources:
    - A blog post detailing ColPali, a vision retrieval model, can be found [here](https://huggingface.co/blog/manu/colpali). 📝
    - The code for training ColPali and for the `colpali-engine` package can be found [here](https://github.com/illuin-tech/colpali). 🌎
    - Cookbooks to fine-tune ColPali (with optional quantization), generate similarity maps, ... can be found [here](https://github.com/tonywu71/colpali-cookbooks). 📚

    Adapted from colpali-engine==0.3.0: https://github.com/illuin-tech/colpali.
    """
)
class ColPaliForRetrieval(PaliGemmaForConditionalGeneration):
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: ColPaliConfig):
        super().__init__(config=config)

        self.embedding_dim = self.config.embedding_dim
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.embedding_dim)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.post_init()

    @add_start_docstrings_to_model_forward(
        """
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
    )
    @replace_return_docstrings(output_type=ColPaliModelOutput, config_class="ColPaliConfig")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
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
    ) -> Union[Tuple, ColPaliModelOutput]:
        r"""
        Returns:
        """
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=False,
            num_logits_to_keep=num_logits_to_keep,
        )
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        embeddings = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if not return_dict:
            return (embeddings,)

        return ColPaliModelOutput(embeddings=embeddings)

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
