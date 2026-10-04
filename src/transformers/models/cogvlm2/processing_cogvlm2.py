# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring
from ...video_utils import VideoInput


LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class CogVLM2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "videos_kwargs": {"return_metadata": False},
    }


@auto_docstring
class CogVLM2Processor(ProcessorMixin):
    valid_processor_kwargs = CogVLM2ProcessorKwargs

    def __init__(
        self,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        template_version="base",
        num_visual_tokens_per_frame=66,
    ):
        r"""
        template_version (`str`, *optional*, defaults to `"base"`):
            Default CogVLM2 prompt template. Supported values are `"base"`, `"chat"`, and `"vqa"`.
        num_visual_tokens_per_frame (`int`, *optional*, defaults to 66):
            Number of visual placeholder tokens per video frame. The released CogVLM2-Video
            checkpoint produces 64 spatial tokens plus BOI and EOI tokens.
        """
        self.template_version = template_version
        self.num_visual_tokens_per_frame = num_visual_tokens_per_frame
        self.template_version = template_version
        super().__init__(video_processor, tokenizer, chat_template=chat_template)

    @staticmethod
    def build_prompt(query, history=None, template_version="chat"):
        history = history or []
        if template_version == "base":
            return query
        if template_version == "vqa":
            answer_format = "Short answer:"
        elif template_version == "chat":
            answer_format = "Answer:"
        else:
            raise ValueError(f"Unknown template version: {template_version}")

        prompt = ""
        for old_query, response in history:
            prompt += f"Question: {old_query} {answer_format} {response}\n"
        return prompt + f"Question: {query} {answer_format}"

    def __call__(
        self,
        text=None,
        videos: VideoInput | None = None,
        *,
        history=None,
        template_version=None,
        add_time_indices=None,
        answer=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        **videos_kwargs,
    ):
        if text is None and videos is None:
            raise ValueError("You have to specify at least one of text or videos.")

        template_version = template_version or self.template_version

        if text is None:
            text = ""
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) or not all(isinstance(item, str) for item in text):
            raise ValueError("text must be a string or a list of strings.")

        template_version = template_version or self.template_version
        if add_time_indices is None:
            add_time_indices = template_version == "chat"

        if history is not None:
            if len(text) != 1:
                raise ValueError("history is only supported for a single text input.")
            text = [self.build_prompt(text[0], history=history, template_version=template_version)]
        elif template_version != "base":
            text = [self.build_prompt(item, template_version=template_version) for item in text]

        if answer is None:
            answers = [None] * len(text)
        elif isinstance(answer, str):
            if len(text) != 1:
                raise ValueError("A single answer string can only be used with one text sample.")
            answers = [answer]
        elif isinstance(answer, list) and len(answer) == len(text) and all(isinstance(item, str) for item in answer):
            answers = answer
        else:
            raise ValueError("answer must be a string or a list of strings matching the text batch size.")

        video_inputs = {}
        if videos is not None:
            video_inputs = self.video_processor(videos, return_tensors=return_tensors, **videos_kwargs)
            frame_counts = video_inputs.pop("video_frame_counts")
            frame_counts = frame_counts.tolist() if hasattr(frame_counts, "tolist") else list(frame_counts)
            if len(frame_counts) != len(text):
                raise ValueError(
                    "CogVLM2 expects one video per text sample: "
                    f"got {len(frame_counts)} videos and {len(text)} text samples."
                )
        else:
            frame_counts = [0] * len(text)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 128002
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("CogVLM2 requires the tokenizer to define bos_token_id.")

        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        visual_tokens_per_frame = self.num_visual_tokens_per_frame

        for sample_text, num_frames, sample_answer in zip(text, frame_counts, answers):
            input_ids = [bos_token_id]
            token_type_ids = [LANGUAGE_TOKEN_TYPE]

            for frame_idx in range(num_frames):
                input_ids.extend([pad_token_id] * visual_tokens_per_frame)
                token_type_ids.extend([VISION_TOKEN_TYPE] * visual_tokens_per_frame)
                if add_time_indices:
                    time_ids = self.tokenizer.encode(str(frame_idx), add_special_tokens=False)
                    input_ids.extend(time_ids)
                    token_type_ids.extend([LANGUAGE_TOKEN_TYPE] * len(time_ids))

            multimodal_prefix_length = len(input_ids)
            text_ids = self.tokenizer.encode(sample_text, add_special_tokens=False)
            input_ids.extend(text_ids)
            token_type_ids.extend([LANGUAGE_TOKEN_TYPE] * len(text_ids))

            if sample_answer is not None:
                answer_ids = self.tokenizer.encode(sample_answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                labels = [-100] * len(input_ids) + answer_ids
                input_ids.extend(answer_ids)
                token_type_ids.extend([LANGUAGE_TOKEN_TYPE] * len(answer_ids))
            else:
                labels = None

            if truncation:
                effective_max_length = max_length if max_length is not None else self.tokenizer.model_max_length
                if len(input_ids) > effective_max_length:
                    if effective_max_length < multimodal_prefix_length:
                        raise ValueError(
                            "Truncation would remove CogVLM2 visual placeholder tokens. "
                            "Increase max_length or reduce the number of video frames."
                        )
                    if self.tokenizer.truncation_side == "left" and num_frames:
                        raise ValueError(
                            "Left truncation is not supported for CogVLM2 video inputs because it would "
                            "remove visual placeholders from the beginning of the sequence."
                        )
                    if self.tokenizer.truncation_side == "left":
                        input_ids = input_ids[-effective_max_length:]
                        token_type_ids = token_type_ids[-effective_max_length:]
                        if labels is not None:
                            labels = labels[-effective_max_length:]
                    else:
                        input_ids = input_ids[:effective_max_length]
                        token_type_ids = token_type_ids[:effective_max_length]
                        if labels is not None:
                            labels = labels[:effective_max_length]

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(labels)
        encoded = self.tokenizer.pad(
            {"input_ids": batch_input_ids, "token_type_ids": batch_token_type_ids},
            padding=padding,
            max_length=max_length if padding == "max_length" else None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        data = dict(encoded)
        if any(labels is not None for labels in batch_labels):
            if not all(labels is not None for labels in batch_labels):
                raise ValueError("Either all samples in a batch must provide answer labels or none of them.")

            if hasattr(encoded["input_ids"], "shape"):
                padded_length = encoded["input_ids"].shape[-1]
            else:
                padded_length = max(len(ids) for ids in encoded["input_ids"])

            padded_labels = []
            for labels in batch_labels:
                difference = padded_length - len(labels)
                if self.tokenizer.padding_side == "left":
                    labels = [-100] * difference + labels
                else:
                    labels = labels + [-100] * difference
                padded_labels.append(labels)
            data["labels"] = padded_labels

        data.update(video_inputs)
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = getattr(self.tokenizer, "model_input_names", [])
        video_input_names = getattr(self.video_processor, "model_input_names", [])
        return list(dict.fromkeys(tokenizer_input_names + ["token_type_ids"] + video_input_names))


__all__ = ["CogVLM2Processor"]
