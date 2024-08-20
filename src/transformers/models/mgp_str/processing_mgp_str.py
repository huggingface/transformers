# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for MGP-STR."""

import sys
import warnings
from typing import List, Optional, Union

from transformers import AutoTokenizer

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils.generic import ExplicitEnum
from ...utils.import_utils import is_torch_available


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


if is_torch_available():
    import torch


class DecodeType(ExplicitEnum):
    CHARACTER = "char"
    BPE = "bpe"
    WORDPIECE = "wp"


SUPPORTED_ANNOTATION_FORMATS = (DecodeType.CHARACTER, DecodeType.BPE, DecodeType.WORDPIECE)


class MgpstrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class MgpstrProcessor(ProcessorMixin):
    r"""
    Constructs a MGP-STR processor which wraps an image processor and MGP-STR tokenizers into a single

    [`MgpstrProcessor`] offers all the functionalities of `ViTImageProcessor`] and [`MgpstrTokenizer`]. See the
    [`~MgpstrProcessor.__call__`] and [`~MgpstrProcessor.batch_decode`] for more information.

    Args:
        image_processor (`ViTImageProcessor`, *optional*):
            An instance of `ViTImageProcessor`. The image processor is a required input.
        tokenizer ([`MgpstrTokenizer`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ViTImageProcessor"
    tokenizer_class = "MgpstrTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.tokenizer = tokenizer
        self.bpe_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.wp_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

        super().__init__(image_processor, tokenizer)

    @property
    def char_tokenizer(self):
        warnings.warn(
            "The `char_tokenizer` attribute is deprecated and will be removed in future versions, use `tokenizer` instead.",
            FutureWarning,
        )
        return self.tokenizer

    @char_tokenizer.setter
    def char_tokenizer(self, value):
        warnings.warn(
            "The `char_tokenizer` attribute is deprecated and will be removed in future versions, use `tokenizer` instead.",
            FutureWarning,
        )
        self.tokenizer = value

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        images: Optional[ImageInput] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[MgpstrProcessorKwargs],
    ) -> BatchFeature:
        """
        When used in normal mode, this method forwards all its arguments to ViTImageProcessor's
        [`~ViTImageProcessor.__call__`] and returns its output. This method also forwards the `text` and `kwargs`
        arguments to MgpstrTokenizer's [`~MgpstrTokenizer.__call__`] if `text` is not `None` to encode the text. Please
        refer to the doctsring of the above methods for more information.

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`ImageInput`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **labels** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            MgpstrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        data = {}
        if text is not None:
            text_features = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            data.update(text_features)
        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])
            data.update(image_features)
        return BatchFeature(data=data, tensor_type=output_kwargs["common_kwargs"].get("return_tensors"))

    def batch_decode(self, sequences):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.

        Returns:
            `Dict[str, any]`: Dictionary of all the outputs of the decoded results.
                generated_text (`List[str]`): The final results after fusion of char, bpe, and wp. scores
                (`List[float]`): The final scores after fusion of char, bpe, and wp. char_preds (`List[str]`): The list
                of character decoded sentences. bpe_preds (`List[str]`): The list of bpe decoded sentences. wp_preds
                (`List[str]`): The list of wp decoded sentences.

        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        char_preds, bpe_preds, wp_preds = sequences
        batch_size = char_preds.size(0)

        char_strs, char_scores = self._decode_helper(char_preds, "char")
        bpe_strs, bpe_scores = self._decode_helper(bpe_preds, "bpe")
        wp_strs, wp_scores = self._decode_helper(wp_preds, "wp")

        final_strs = []
        final_scores = []
        for i in range(batch_size):
            scores = [char_scores[i], bpe_scores[i], wp_scores[i]]
            strs = [char_strs[i], bpe_strs[i], wp_strs[i]]
            max_score_index = scores.index(max(scores))
            final_strs.append(strs[max_score_index])
            final_scores.append(scores[max_score_index])

        out = {}
        out["generated_text"] = final_strs
        out["scores"] = final_scores
        out["char_preds"] = char_strs
        out["bpe_preds"] = bpe_strs
        out["wp_preds"] = wp_strs
        return out

    def _decode_helper(self, pred_logits, format):
        """
        Convert a list of lists of bpe token ids into a list of strings by calling bpe tokenizer.

        Args:
            pred_logits (`torch.Tensor`):
                List of model prediction logits.
            format (`Union[DecoderType, str]`):
                Type of model prediction. Must be one of ['char', 'bpe', 'wp'].
        Returns:
            `tuple`:
                dec_strs(`str`): The decode strings of model prediction. conf_scores(`List[float]`): The confidence
                score of model prediction.
        """
        if format == DecodeType.CHARACTER:
            decoder = self.char_decode
            eos_token = 1
            eos_str = "[s]"
        elif format == DecodeType.BPE:
            decoder = self.bpe_decode
            eos_token = 2
            eos_str = "#"
        elif format == DecodeType.WORDPIECE:
            decoder = self.wp_decode
            eos_token = 102
            eos_str = "[SEP]"
        else:
            raise ValueError(f"Format {format} is not supported.")

        dec_strs, conf_scores = [], []
        batch_size = pred_logits.size(0)
        batch_max_length = pred_logits.size(1)
        _, preds_index = pred_logits.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, batch_max_length)[:, 1:]
        preds_str = decoder(preds_index)
        preds_max_prob, _ = torch.nn.functional.softmax(pred_logits, dim=2).max(dim=2)
        preds_max_prob = preds_max_prob[:, 1:]

        for index in range(batch_size):
            pred_eos = preds_str[index].find(eos_str)
            pred = preds_str[index][:pred_eos]
            pred_index = preds_index[index].cpu().tolist()
            pred_eos_index = pred_index.index(eos_token) if eos_token in pred_index else -1
            pred_max_prob = preds_max_prob[index][: pred_eos_index + 1]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1] if pred_max_prob.nelement() != 0 else 0.0
            dec_strs.append(pred)
            conf_scores.append(confidence_score)

        return dec_strs, conf_scores

    def char_decode(self, sequences):
        """
        Convert a list of lists of char token ids into a list of strings by calling char tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of char decoded sentences.
        """
        decode_strs = [seq.replace(" ", "") for seq in self.tokenizer.batch_decode(sequences)]
        return decode_strs

    def bpe_decode(self, sequences):
        """
        Convert a list of lists of bpe token ids into a list of strings by calling bpe tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of bpe decoded sentences.
        """
        return self.bpe_tokenizer.batch_decode(sequences)

    def wp_decode(self, sequences):
        """
        Convert a list of lists of word piece token ids into a list of strings by calling word piece tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of wp decoded sentences.
        """
        decode_strs = [seq.replace(" ", "") for seq in self.wp_tokenizer.batch_decode(sequences)]
        return decode_strs
