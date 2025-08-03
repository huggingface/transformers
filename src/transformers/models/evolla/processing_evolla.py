# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""
Processor class for EVOLLA.
"""

import os
from typing import Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import (
    ProcessorMixin,
)
from ..auto import AutoTokenizer


PROTEIN_VALID_KEYS = ["aa_seq", "foldseek", "msa"]


class EvollaProcessor(ProcessorMixin):
    r"""
    Constructs a EVOLLA processor which wraps a LLama tokenizer and SaProt tokenizer (EsmTokenizer) into a single processor.

    [`EvollaProcessor`] offers all the functionalities of [`EsmTokenizer`] and [`LlamaTokenizerFast`]. See the
    docstring of [`~EvollaProcessor.__call__`] and [`~EvollaProcessor.decode`] for more information.

    Args:
        protein_tokenizer (`EsmTokenizer`):
            An instance of [`EsmTokenizer`]. The protein tokenizer is a required input.
        tokenizer (`LlamaTokenizerFast`, *optional*):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        protein_max_length (`int`, *optional*, defaults to 1024):
            The maximum length of the sequence to be generated.
        text_max_length (`int`, *optional*, defaults to 512):
            The maximum length of the text to be generated.
    """

    attributes = ["protein_tokenizer", "tokenizer"]
    valid_kwargs = ["sequence_max_length"]
    # protein_tokenizer_class = "EsmTokenizer"
    # tokenizer_class = "LlamaTokenizerFast"
    protein_tokenizer_class = "AutoTokenizer"
    tokenizer_class = "AutoTokenizer"
    protein_tokenizer_dir_name = "protein_tokenizer"
    # tokenizer_dir_name = "text_tokenizer"

    def __init__(self, protein_tokenizer, tokenizer=None, protein_max_length=1024, text_max_length=512, **kwargs):
        if protein_tokenizer is None:
            raise ValueError("You need to specify an `protein_tokenizer`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(protein_tokenizer, tokenizer)

        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.protein_max_length = protein_max_length
        self.text_max_length = text_max_length

    def process_proteins(self, proteins, protein_max_length=1024):
        sa_sequences = []
        for protein in proteins:
            aa_seq = protein.get("aa_seq")
            foldseek = protein.get("foldseek")
            sa_sequence = "".join([s.upper() + f.lower() for s, f in zip(aa_seq, foldseek)])
            sa_sequences.append(sa_sequence)

        sa_tokens = self.protein_tokenizer.batch_encode_plus(
            sa_sequences, return_tensors="pt", truncation=True, max_length=protein_max_length, padding=True
        )
        return sa_tokens

    def process_text(
        self,
        texts,
        text_max_length: int = 512,
    ):
        prompts = []
        for messages in texts:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=text_max_length,
        )
        return prompt_inputs

    def __call__(
        self,
        proteins: Optional[Union[list[dict], dict]] = None,
        messages_list: Optional[Union[list[list[dict]], list[dict]]] = None,
        protein_max_length: Optional[int] = None,
        text_max_length: Optional[int] = None,
        **kwargs,
    ):
        r"""This method takes batched or non-batched proteins and messages_list and converts them into format that can be used by
        the model.

        Args:
            proteins (`Union[List[dict], dict]`):
                A list of dictionaries or a single dictionary containing the following keys:
                    - `"aa_seq"` (`str`) -- The amino acid sequence of the protein.
                    - `"foldseek"` (`str`) -- The foldseek string of the protein.
            messages_list (`Union[List[List[dict]], List[dict]]`):
                A list of lists of dictionaries or a list of dictionaries containing the following keys:
                    - `"role"` (`str`) -- The role of the message.
                    - `"content"` (`str`) -- The content of the message.
            protein_max_length (`int`, *optional*, defaults to 1024):
                The maximum length of the sequence to be generated.
            text_max_length (`int`, *optional*, defaults to 512):
                The maximum length of the text.

        Return:
            a dict with following keys:
                - `protein_input_ids` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The input IDs for the protein sequence.
                - `protein_attention_mask` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The attention mask for the protein sequence.
                - `text_input_ids` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The input IDs for the text sequence.
                - `text_attention_mask` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The attention mask for the text sequence.
        """
        # proteins and messages_list should be provided
        if proteins is None or messages_list is None:
            raise ValueError("You need to specify `messages_list` and `proteins`.")

        protein_max_length = protein_max_length if protein_max_length is not None else self.protein_max_length
        text_max_length = text_max_length if text_max_length is not None else self.text_max_length

        # proteins should be List[dict]
        if isinstance(proteins, dict):
            proteins = [proteins]
        # messages_list should be List[List[dict]]
        if isinstance(messages_list, (list, tuple)) and not isinstance(messages_list[0], (list, tuple)):
            messages_list = [messages_list]
        # Check if batched proteins are in the correct format
        if isinstance(proteins, (list, tuple)) and not all(isinstance(p, dict) for p in proteins):
            raise ValueError("The proteins should be a list of dictionaries, but not all elements are dictionaries.")
        if isinstance(proteins, (list, tuple)) and not all(
            all(k in PROTEIN_VALID_KEYS for k in p.keys()) for p in proteins
        ):
            raise ValueError(
                "There should be a list of dictionaries with keys: "
                f"{', '.join(PROTEIN_VALID_KEYS)} for each protein."
                f"But got: {proteins}"
            )
        # Check if batched messages_list is in the correct format
        if isinstance(messages_list, (list, tuple)):
            for messages in messages_list:
                if not isinstance(messages, (list, tuple)):
                    raise ValueError(f"Each messages in messages_list should be a list instead of {type(messages)}.")
                if not all(isinstance(m, dict) for m in messages):
                    raise ValueError(
                        "Each message in messages_list should be a list of dictionaries, but not all elements are dictionaries."
                    )
                if any(len(m.keys()) != 2 for m in messages) or any(
                    set(m.keys()) != {"role", "content"} for m in messages
                ):
                    raise ValueError(
                        "Each message in messages_list should be a list of dictionaries with two keys: 'role' and 'content'."
                        f"But got: {messages}"
                    )
        else:
            raise ValueError(
                f"The messages_list should be a list of lists of dictionaries, but it's {type(messages_list)}."
            )
        sa_tokens = self.process_proteins(proteins, protein_max_length)

        text_tokens = self.process_text(messages_list, text_max_length)

        return BatchFeature(
            data={
                "protein_input_ids": sa_tokens["input_ids"],
                "protein_attention_mask": sa_tokens["attention_mask"],
                "input_ids": text_tokens["input_ids"],
                "attention_mask": text_tokens["attention_mask"],
            }
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def protein_batch_decode(self, *args, **kwargs):
        return self.protein_tokenizer.batch_decode(*args, **kwargs)

    def protein_decode(self, *args, **kwargs):
        return self.protein_tokenizer.decode(*args, **kwargs)

    # overwrite to save the protein tokenizer in a separate folder
    # Adapted from instructblip.processing_instructblip.py (https://github.com/huggingface/transformers/blob/9b479a245b793cac2a8b2e87c6d8e81bb24e20c4/src/transformers/models/instructblip/processing_instructblip.py#L191-L221)
    def save_pretrained(self, save_directory, **kwargs):
        # only save the protein tokenizer in sub_dir
        self.protein_tokenizer.save_pretrained(os.path.join(save_directory, self.protein_tokenizer_dir_name))

        # we modify the attributes so that only the text tokenizer are saved in the main folder
        protein_tokenizer_present = "protein_tokenizer" in self.attributes
        # find the correct position of it in the attributes list
        protein_tokenizer_index = self.attributes.index("protein_tokenizer") if protein_tokenizer_present else None
        if protein_tokenizer_present and protein_tokenizer_index is not None:
            self.attributes.remove("protein_tokenizer")

        outputs = super().save_pretrained(save_directory, **kwargs)

        if protein_tokenizer_present and protein_tokenizer_index is not None:
            self.attributes.insert(protein_tokenizer_index, "protein_tokenizer")

        return outputs

    # overwirte to load the protein tokenizer from a separate folder
    # Adapted from instructblip.processing_instructblip.py (https://github.com/huggingface/transformers/blob/9b479a245b793cac2a8b2e87c6d8e81bb24e20c4/src/transformers/models/instructblip/processing_instructblip.py#L191-L221)
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]
        protein_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder=cls.protein_tokenizer_dir_name
        )

        processor.protein_tokenizer = protein_tokenizer

        return processor


__all__ = ["EvollaProcessor"]
