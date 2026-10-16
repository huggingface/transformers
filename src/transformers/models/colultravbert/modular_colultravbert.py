# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import SubConfigSpec
from ...utils import auto_docstring
from ...utils.import_utils import requires
from ..auto import AutoConfig
from ..colmodernvbert.configuration_colmodernvbert import ColModernVBertConfig
from ..colmodernvbert.modeling_colmodernvbert import ColModernVBertForRetrieval, ColModernVBertPreTrainedModel
from ..colmodernvbert.processing_colmodernvbert import ColModernVBertProcessor


@auto_docstring(checkpoint="athrael-soju/ColUltraVBERT")
@strict
class ColUltraVBertConfig(ColModernVBertConfig):
    model_type = "colultravbert"
    sub_configs_defaults = {
        "vlm_config": SubConfigSpec(config_class=AutoConfig, model_type="ultravbert"),
    }


@requires(backends=("torch",))
@auto_docstring
class ColUltraVBertProcessor(ColModernVBertProcessor):
    def __init__(
        self,
        image_processor,
        tokenizer=None,
        chat_template=None,
        image_seq_len: int = 64,
        visual_prompt_prefix: str | None = None,
        query_prefix: str | None = None,
        **kwargs,
    ):
        super().__init__(
            image_processor, tokenizer, chat_template=chat_template, image_seq_len=image_seq_len, **kwargs
        )
        # The tokenizer prepends its own begin-of-text token.
        self.visual_prompt_prefix = visual_prompt_prefix or (
            f"User:{self.image_token}Describe the image.<end_of_utterance>\nAssistant:"
        )


class ColUltraVBertPreTrainedModel(ColModernVBertPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    ColUltraVBert embeds document pages and queries as L2-normalized multi-vectors for late-interaction retrieval,
    following the ColPali approach on top of [`UltraVBertModel`].
    """
)
class ColUltraVBertForRetrieval(ColModernVBertForRetrieval):
    pass


__all__ = [
    "ColUltraVBertConfig",
    "ColUltraVBertForRetrieval",
    "ColUltraVBertPreTrainedModel",
    "ColUltraVBertProcessor",
]
