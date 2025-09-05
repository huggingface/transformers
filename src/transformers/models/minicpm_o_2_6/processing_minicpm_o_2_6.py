# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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
Processor class for MiniCPMO.
"""

import math
import re

from typing import Any, Dict, Optional, Union

import numpy as np
import json
from copy import deepcopy
from functools import lru_cache

from PIL import Image
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack, ImagesKwargs, AudioKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput

from ...feature_extraction_utils import BatchFeature
from ...utils import is_torch_device, is_torch_dtype, requires_backends, TensorType, logging
from ...utils.import_utils import is_jinja_available, is_torch_available

if is_torch_available():
    import torch

if is_jinja_available():
    from jinja2 import Environment

logger = logging.get_logger(__name__)


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


class MiniCPMOBatchFeature(BatchFeature):
    r"""
    Extend from BatchFeature for supporting various image size
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self

    def to(self, *args, **kwargs) -> "MiniCPMOBatchFeature":
        requires_backends(self, ["torch"])
        import torch

        def cast_tensor(v):
            # check if v is a floating point
            if torch.is_floating_point(v):
                # cast and send to device
                return v.to(*args, **kwargs)
            elif device is not None:
                return v.to(device=device)
            else:
                return v

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self


class MiniCPMOImageKwargs(ImagesKwargs, total=False):
    max_slice_nums: Optional[int]
    use_image_id: bool


class MiniCPMOAudioKwargs(AudioKwargs, total=False):
    audio_parts: Optional[list]
    chunk_input: bool


class MiniCPM_o_2_6ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MiniCPMOImageKwargs
    audio_kwargs: MiniCPMOAudioKwargs
    _defaults = {
        "images_kwargs": {
            "do_pad": True,
            "max_slice_nums": None,
            "use_image_id": True,
        },
        "audio_kwargs": {
            "audio_parts": None,
            "chunk_input": False,
            "sampling_rate": 16000,
        },
        "text_kwargs": {"add_special_tokens": True},
        "videos_kwargs": {},
        "common_kwargs": {"return_tensors": TensorType.PYTORCH},
    }


PARSE_TEMPLATE = "{%- for i, msg in enumerate(msgs) -%}{%- set role = msg[\"role\"] -%}{%- set content = msg[\"content\"] -%}{%- if role not in [\"system\", \"user\", \"assistant\"] -%}{{ raise_error(\"Invalid role: \" + role) }}{%- endif -%}{%- if i == 0 and role not in [\"user\", \"system\"] -%}{{ raise_error(\"The role of first msg should be user or system\") }}{%- endif -%}{%- if content is string -%}{%- set content = [content] -%}{%- endif -%}{%- set cur_msgs = [] -%}{%- for c in content -%}{%- if isinstance(c, Image.Image) -%}{{ collect_image(c, cur_msgs) }}{%- elif c.__class__.__name__ == \"ndarray\" -%}{{ collect_audio(c, i, cur_msgs) }}{%- elif c is string -%}{{ collect_text(c, cur_msgs) }}{%- endif -%}{%- endfor -%}{%- set _ = msg.update({\"content\": join_content(cur_msgs, omni_input)}) -%}{%- endfor -%}"


@lru_cache(maxsize=1)
def get_precompiled_template():
    env = Environment(
        cache_size=100,
        auto_reload=False,
        optimized=True
    )

    env.globals.update({
        'enumerate': enumerate,
        'isinstance': isinstance,
        'Image': Image,
        'np': np
    })

    return env.from_string(PARSE_TEMPLATE)


class MiniCPM_o_2_6Processor(ProcessorMixin):
    r"""
    Constructs a MiniCPMV processor which wraps a MiniCPMV image processor and a MiniCPMV tokenizer into a single processor.

    [`MiniCPMVProcessor`] offers all the functionalities of [`MiniCPMVImageProcessor`] and [`LlamaTokenizerWrapper`]. See the
    [`~MiniCPMVProcessor.__call__`] and [`~MiniCPMVProcessor.decode`] for more information.

    Args:
        image_processor ([`MiniCPMVImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["tokenizer", "image_processor", "feature_extractor"]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "MiniCPMVImageProcessorFast"
    feature_extractor_class = "MiniCPM_o_2_6FeatureExtractor"

    def __init__(self, tokenizer=None, image_processor=None, feature_extractor=None, chat_template=None, tts_chat_template=None):
        super().__init__(tokenizer, image_processor, feature_extractor, chat_template=chat_template)

        self.tts_chat_template = tts_chat_template

        self.image_tag = getattr(tokenizer, 'image_tag', "(<image>./</image>)")
        self.image_pattern = getattr(tokenizer, 'image_pattern', "\(<image>./</image>\)")
        self.audio_tag = getattr(tokenizer, 'audio_tag', "(<audio>./</audio>)")
        self.audio_pattern = getattr(tokenizer, 'audio_pattern', "\(<audio>./</audio>\)")
        self.split_pattern = f"({self.image_pattern}|{self.audio_pattern})"

        self.terminators = [tokenizer.eos_token, tokenizer.pad_token, tokenizer.tts_end]
        self.terminator_ids = [tokenizer.convert_tokens_to_ids(t) for t in self.terminators]
        

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]],
        images: ImageInput = None,
        audios: Union[np.ndarray, list[np.ndarray], list[list[np.ndarray]]] = None,
        **kwargs: Unpack[MiniCPM_o_2_6ProcessorKwargs],
    ) -> MiniCPMOBatchFeature:
        output_kwargs = self._merge_kwargs(MiniCPM_o_2_6ProcessorKwargs, self.tokenizer.init_kwargs, **kwargs)
        image_kwargs = output_kwargs["images_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]

        if images:
            image_inputs = self.image_processor(images, **image_kwargs)
        else:
            image_inputs = None

        if audios:
            audio_features, audio_feature_lens = self.feature_extractor(
                audios,
                audio_parts=audio_kwargs["audio_parts"],
                sampling_rate=audio_kwargs["sampling_rate"],
            )
            audio_phs = self.get_audios_placeholder(audios=audios, chunk_input=audio_kwargs["chunk_input"])
        else:
            audio_features, audio_feature_lens, audio_phs = [], [], []

        model_inputs = self._convert_omni_to_inputs(
            image_inputs,
            audio_phs,
            text,
            max_slice_nums=image_kwargs["max_slice_nums"],
            use_image_id=image_kwargs["use_image_id"],
            **text_kwargs,
        )

        model_inputs["audio_features"] = audio_features
        model_inputs["audio_feature_lens"] = audio_feature_lens

        return MiniCPMOBatchFeature(data={**model_inputs})

    def parse_msgs(self, msgs, omni_input=False, use_tts_template=False):
        images = []
        audios = []
        audio_parts = []

        for i, msg in enumerate(msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["system", "user", "assistant"]
            if i == 0:
                assert role in ["user", "system"], "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, np.ndarray):  # audio
                    audios.append(c)
                    audio_parts.append(i)
                    cur_msgs.append("(<audio>./</audio>)")
                    use_tts_template = True
                elif isinstance(c, str):
                    cur_msgs.append(c)
            if omni_input:
                msg["content"] = "".join(cur_msgs)
            else:
                msg["content"] = "\n".join(cur_msgs)

        return msgs, images, audios, audio_parts, use_tts_template

    def parse_msgs_jinja(self, msgs, omni_input=False, use_tts_template=False):
        results = {
            'images': [],
            'audios': [],
            'audio_parts': [],
            'use_tts_template': use_tts_template
        }

        # 获取预编译的模板
        template = get_precompiled_template()

        # 添加自定义函数
        def collect_image(img, msg_list):
            results['images'].append(img)
            msg_list.append("(<image>./</image>)")
            return ""

        def collect_audio(audio, i, msg_list):
            results['audios'].append(audio)
            results['audio_parts'].append(i)
            results['use_tts_template'] = True
            msg_list.append("(<audio>./</audio>)")
            return ""

        def collect_text(text, msg_list):
            msg_list.append(text)
            return ""

        def join_content(msg_list, omni_input):
            if omni_input:
                return "".join(msg_list)
            else:
                return "\n".join(msg_list)

        def raise_error(msg):
            raise AssertionError(msg)

        # 渲染模板
        template.render(
            msgs=msgs,
            omni_input=omni_input,
            collect_image=collect_image,
            collect_audio=collect_audio,
            collect_text=collect_text,
            join_content=join_content,
            raise_error=raise_error
        )

        return (
            msgs,
            results['images'],
            results['audios'],
            results['audio_parts'],
            results['use_tts_template']
        )

    def apply_chat_template(
        self,
        msgs,
        chunk_input=True,
        max_slice_nums=None,
        max_length=32768,
        omni_input=False,
        use_image_id=None,
        use_tts_template=False,
        **kwargs,
    ):
        """
        Unified chat function

        Args:
            msgs: the input chat msgs, support text: (string)  / image: (PIL.Image) / audio (numpy.ndarray)
            chunk_input: whether to split audio into 1s chunks
            max_length: the maximum length of input
            max_slice_nums: control the maximum number of image slices
            omni_input: determine whether it is omni mode
            use_image_id: for video understanding or omni understanding, use_image_id should be False
            use_tts_template: if the msgs contain audio, use_tts_template should be True
        """
        if isinstance(msgs[0], list):
            msgs_list = msgs
        else:
            msgs_list = [msgs]

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for msgs in msgs_list:
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            assert len(msgs) > 0, "msgs is empty"

            parsed_msgs, images, audios, audio_parts, use_tts_template = self.parse_msgs_jinja(deepcopy(msgs), omni_input, use_tts_template)
            prompts = self.tokenizer.apply_chat_template(
                parsed_msgs,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.tts_chat_template if use_tts_template else None,
            )

            prompts_lists.append(prompts)
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        inputs = self.__call__(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts=audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            chunk_input=chunk_input,
            return_tensors="pt",
            max_length=max_length,
        )
        return inputs

    def decode(self, result_ids, skeip_special_tokens: bool = False):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            start, end = 0, len(result)
            for i, tok in enumerate(result):
                if tok == self.tokenizer.bos_token_id:
                    start = i + 1
                else:
                    break
            for i in range(len(result) - 1, -1, -1):
                if result[i] in self.terminator_ids:
                    end = i
                else:
                    break
            result = result[start:end]
            result_text.append(self.tokenizer.decode(result, skip_special_tokens=skeip_special_tokens))
        return result_text

    def get_sys_prompt(self, ref_audio=None, mode="default", language="zh"):
        """
        Choose different system prompts according to different tasks
        Args:
            ref_audio: if ref_audio is not None, will use the voice cloning prompts, and the voice
                       generated by the model will refer to the timbre of ref audio
            mode:
                "default": default system prompt and not refer to any task
                "omni": input video and audio simultaneously
                "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
                "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
                "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
            language: prompts language, the model has the ability to automatically select the response language
                    based on the question language
        Returns:

        """
        if ref_audio is not None:
            assert isinstance(ref_audio, np.ndarray), "ref_audio error"
        if mode == "omni":
            if language == "zh":
                sys_prompt = "你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。"
                vc_prompt_prefix = sys_prompt + "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                sys_prompt = "You are a helpful assistant. You can accept video, audio and text input and output voice and text. "
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                logger.warning(
                    "Warning: ref_audio is None, speech generation will be performed based on the default voice."
                )
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                print("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio con't be None in voice_cloning mode.")

            return sys_msgs
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs

    def _convert(self, input_str, max_inp_length: Optional[int] = None, **kwargs):
        input_ids = self.tokenizer.encode(input_str, **kwargs)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        # image bound
        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_idx = torch.where(start_cond)[0]
        image_start_idx += 1
        image_end_idx = torch.where(end_cond)[0]

        valid_image_nums = max(len(image_start_idx), len(image_end_idx))

        image_bounds = torch.hstack(
            [
                image_start_idx[:valid_image_nums].unsqueeze(-1),
                image_end_idx[:valid_image_nums].unsqueeze(-1),
            ]
        )

        # audio bound
        audio_start_idx = torch.where(input_ids == self.tokenizer.audio_start_id)[0]
        audio_end_idx = torch.where(input_ids == self.tokenizer.audio_end_id)[0]
        assert len(audio_start_idx) == len(audio_end_idx)
        audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])

        spk_start_idx = torch.where(input_ids == self.tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids == self.tokenizer.spk_end_id)[0]
        assert len(spk_start_idx) == len(spk_end_idx)
        spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])

        return input_ids, image_bounds, audio_bounds, spk_bounds

    def _convert_omni_to_inputs(
        self,
        images,
        audio_phs,
        texts: Union[str, list[str]],
        truncation=None,
        max_length=None,
        max_slice_nums=None,
        use_image_id=None,
        return_tensors=None,
        **kwargs,
    ):
        if images is None and audio_phs is None:
            model_inputs = self.tokenizer(
                texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs
            )
            return MiniCPMOBatchFeature(data={**model_inputs})

        if isinstance(texts, str):
            texts = [texts]

        bs = len(texts)
        if images is not None:
            images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]
        else:
            images, image_sizes, tgt_sizes = [[]] * bs, [[]] * bs, [[]] * bs

        final_texts_list = []
        input_ids_list = []
        image_bounds_list = []
        audio_bounds_list = []
        spk_bounds_list = []

        for index, text in enumerate(texts):
            text_chunks = re.split(self.split_pattern, text)

            image_tags = re.findall(self.image_pattern, text)
            audio_tags = re.findall(self.audio_pattern, text)

            if image_tags:
                assert images is not None
                assert len(image_tags) == len(image_sizes[index])
            if audio_tags:
                assert audio_phs is not None
                assert len(audio_tags) == len(audio_phs[index])

            image_id = 0
            audio_id = 0
            for i, chunk in enumerate(text_chunks):
                if chunk == self.image_tag:
                    image_placeholder = self.get_slice_image_placeholder(
                        image_sizes[index][image_id], image_id, max_slice_nums, use_image_id
                    )
                    image_id += 1
                    text_chunks[i] = image_placeholder
                elif chunk == self.audio_tag:
                    audio_placeholder = audio_phs[index][audio_id]
                    audio_id += 1
                    text_chunks[i] = audio_placeholder

            final_text = "".join(text_chunks)
            input_ids, image_bounds, audio_bounds, spk_bounds = self._convert(final_text, max_length, **kwargs)

            final_texts_list.append(final_text)
            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
            audio_bounds_list.append(audio_bounds)
            spk_bounds_list.append(spk_bounds)

        model_inputs = self.tokenizer(
            final_texts_list,
            padding="longest",
            padding_side="left",
            return_tensors=return_tensors,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

        padded_input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        for i in range(bs):
            length = (attention_mask[i] == 0).sum().item()
            image_bounds_list[i] = image_bounds_list[i] + length
            audio_bounds_list[i] = audio_bounds_list[i] + length
            spk_bounds_list[i] = spk_bounds_list[i] + length
            attention_mask[i, :length] = False

        data = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes,
            "audio_bounds": audio_bounds_list,
            "spk_bounds": spk_bounds_list,
        }

        return data

    def get_slice_image_placeholder(self, image_size, image_idx=0, max_slice_nums=None, use_image_id=None):
        max_slice_nums = self.image_processor.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        grid = self.image_processor.get_sliced_grid(image_size=image_size, max_slice_nums=max_slice_nums)

        image_placeholder = (
            self.tokenizer.im_start
            + self.tokenizer.unk_token * self.image_processor.image_feature_size
            + self.tokenizer.im_end
        )
        use_image_id = self.image_processor.use_image_id if use_image_id is None else bool(use_image_id)
        if use_image_id:
            final_placeholder = (
                f"{self.tokenizer.im_id_start}{image_idx}{self.tokenizer.im_id_end}" + image_placeholder
            )
        else:
            final_placeholder = image_placeholder

        if self.image_processor.slice_mode:
            final_placeholder = final_placeholder + self.get_grid_placeholder(grid=grid)
        return final_placeholder

    def get_grid_placeholder(self, grid):
        if grid is None:
            return ""
        slice_image_placeholder = (
            self.tokenizer.slice_start
            + self.tokenizer.unk_token * self.image_processor.image_feature_size
            + self.tokenizer.slice_end
        )

        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines.append(slice_image_placeholder)
            slices.append("".join(lines))

        slice_placeholder = "\n".join(slices)
        return slice_placeholder

    def get_audios_placeholder(self, audios,
                               chunk_input: Optional[bool] = False,
                               chunk_length: Optional[int] = 1):
        audios_list = self.feature_extractor.format_audios(audios)
        audio_ph_list = []
        for audios in audios_list:
            if audios:
                audio_ph_list.append(
                    [self.get_single_audio_placeholder(len(a), chunk_input, chunk_length) for a in audios]
                )
            else:
                audio_ph_list.append([])
        return audio_ph_list

    def get_single_audio_placeholder(self, audio_lens, chunk_input, chunk_length):
        pool_step = 2
        feature_lens = math.ceil(audio_lens / self.feature_extractor.hop_length)

        feature_lens = (feature_lens - 1) // 2 + 1
        output_lens = (feature_lens - pool_step) // pool_step + 1

        if chunk_input:
            fbank_feat_in_chunk = int(chunk_length * 100)
            cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
            audio_embeds_in_chunk = (cnn_feat_in_chunk - pool_step) // pool_step + 1
            num_audio_chunks = (output_lens + audio_embeds_in_chunk - 1) // audio_embeds_in_chunk

            place_holders = ""
            total_unk_len = 0
            for _ in range(num_audio_chunks):
                unk_len = min(audio_embeds_in_chunk, output_lens - total_unk_len)
                place_holders += (
                    self.tokenizer.audio_start + self.tokenizer.unk_token * unk_len + self.tokenizer.audio_end
                )
                total_unk_len += unk_len
            audio_placeholder = place_holders
        else:
            audio_placeholder = (
                self.tokenizer.audio_start + self.tokenizer.unk_token * output_lens + self.tokenizer.audio_end
            )

        return audio_placeholder

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + feature_extractor_input_names))


__all__ = ["MiniCPM_o_2_6Processor"]
