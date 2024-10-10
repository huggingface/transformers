# coding=utf-8
# Copyright 2024 Mesh TensorFlow authors, HHEMv2 Authors and HuggingFace Inc. team.
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
"""PyTorch HHEMv2 model."""

from typing import Optional, Tuple, Union

import torch

from ...modeling_outputs import (
    TokenClassifierOutput,
)
from ...utils import (
    add_start_docstrings,
    logging,
    replace_return_docstrings,
)
from ..auto.tokenization_auto import AutoTokenizer
from ..t5 import T5ForTokenClassification, T5PreTrainedModel
from .configuration_hhemv2 import HHEMv2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HHEMv2Config"
_CHECKPOINT_FOR_DOC = "vectara/hallucination_evaluation_model"

HHEMV2_START_DOCSTRING = r"""

    HHEM-2.1-open is a major upgrade to [HHEM-1.0-Open](https://huggingface.co/vectara/hallucination_evaluation_model/tree/hhem-1.0-open)
    created by [Vectara](https://vectara.com) in November 2023. The HHEM model series are designed for detecting hallucinations in
    LLMs. They are particularly useful in the context of building retrieval-augmented-generation (RAG) applications where a set of
    facts is summarized by an LLM, and HHEM can be used to measure the extent to which this summary is factually consistent with
    the facts.

    Parameters:
        config ([`HHEMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    HHEMV2_START_DOCSTRING,
)
class HHEMv2Model(T5PreTrainedModel):
    config_class = HHEMv2Config
    _tied_weights_keys = ["transformer.shared.weight", "transformer.encoder.embed_tokens.weight"]
    _no_split_modules = None
    base_model_prefix = "t5"

    def __init__(self, config=HHEMv2Config()):
        super().__init__(config)
        self.t5 = T5ForTokenClassification(config)

        self.prompt = config.prompt
        self.tokenzier = AutoTokenizer.from_pretrained(config.foundation)

    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (labels is not None) and (labels.shape != input_ids.shape):
            assert labels.shape[0] == input_ids.shape[0]
            if (
                self.config.problem_type == "multi_label_classification"
            ):  # actually does not work for multi_label_classification because T5ForTokenClassification does not consider this problem type
                processed_labels = [label + [float(0)] * (input_ids.shape[1] - 2) for label in labels.tolist()]
            else:  # reshape the labels to make it work for T5ForTokenClassification
                processed_labels = [[label] + [0] * (input_ids.shape[1] - 1) for label in labels.tolist()]
            labels = torch.tensor(processed_labels, dtype=torch.long).to(input_ids.device)
            # print(labels.shape)
            # print(labels)

        self.t5.eval()
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            if labels is not None:
                loss, logits, output = outputs
                logits = logits[:, 0, :]
                outputs = (loss, logits, output)
            else:
                logits, output = outputs
                logits = logits[:, 0, :]
                outputs = (logits, output)
        else:
            logits = outputs.logits
            logits = logits[:, 0, :]
            outputs.logits = logits

        return outputs

    def predict(self, text_pairs):
        tokenizer = self.tokenzier
        pair_dict = [{"text1": pair[0], "text2": pair[1]} for pair in text_pairs]
        inputs = tokenizer([self.prompt.format(**pair) for pair in pair_dict], return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.t5(**inputs)
        logits = outputs.logits

        logits = logits[:, 0, :]

        transformed_probs = torch.softmax(logits, dim=-1)
        raw_scores = transformed_probs[:, 1]  # the probability of class 1
        return raw_scores
