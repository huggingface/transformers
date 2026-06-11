# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for InstructBLIP. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""

from ...image_processing_utils import BatchFeature
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class InstructBlipVideoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
    }


@auto_docstring
class InstructBlipVideoProcessor(ProcessorMixin):
    valid_processor_kwargs = InstructBlipVideoProcessorKwargs

    def __init__(self, video_processor, tokenizer, qformer_tokenizer, num_query_tokens=None, **kwargs):
        r"""
        qformer_tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
        num_query_tokens (`int`, *optional*):
            Number of tokens used by the Qformer as queries, should be same as in model's config.
        """
        if not hasattr(tokenizer, "video_token"):
            video_token = AddedToken("<video>", normalized=False, special=True)
            tokenizer.add_tokens([video_token], special_tokens=True)
            self.video_token = video_token.content
        else:
            self.video_token = tokenizer.video_token
        self.image_token = self.video_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.num_query_tokens = num_query_tokens
        super().__init__(video_processor, tokenizer, qformer_tokenizer)

    @auto_docstring
    def __call__(
        self,
        videos: VideoInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[InstructBlipVideoProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            InstructBlipVideoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Tokenize original text with qformer BEFORE video token insertion; qformer needs BOS/EOS
        qformer_encoding = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            qformer_text_kwargs = {**output_kwargs["text_kwargs"], "add_special_tokens": True}
            qformer_text_encoding = self.qformer_tokenizer(text, **qformer_text_kwargs)
            qformer_encoding["qformer_input_ids"] = qformer_text_encoding.pop("input_ids")
            qformer_encoding["qformer_attention_mask"] = qformer_text_encoding.pop("attention_mask")

        model_inputs = super().__call__(videos=videos, text=text, **output_kwargs)
        model_inputs.update(qformer_encoding)
        return model_inputs

    def validate_inputs(
        self,
        videos: VideoInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(videos=videos, text=text, **kwargs)

        if videos is None and text is None:
            raise ValueError("You have to specify at least videos or text.")

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        if text is not None and videos is not None and self.num_query_tokens is not None:
            text = [self.video_token + sample for sample in text]
        return images, text, videos, audio

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        # InstructBLIP video uses 4 frames, each with num_query_tokens video tokens
        return self.video_token * self.num_query_tokens * 4

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        qformer_input_names = ["qformer_input_ids", "qformer_attention_mask"]
        return tokenizer_input_names + video_processor_input_names + qformer_input_names


__all__ = ["InstructBlipVideoProcessor"]
