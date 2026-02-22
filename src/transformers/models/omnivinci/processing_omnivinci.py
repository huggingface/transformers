# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import os.path as osp
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import PIL.Image
import torch
import whisper
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import load_image
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack

from .configuration_omnivinci import MEDIA_TOKENS, MM_BOS_EOS_TOKENS
from .media import Sound, Video, extract_media


def _collect_encoder_boundary_tokens(config) -> list[str]:
    token_keys = {"start_tokens", "end_tokens", "sep_tokens"}
    collected = []
    seen = set()

    def _maybe_add(token):
        if not isinstance(token, str) or token == "None" or token in seen:
            return
        seen.add(token)
        collected.append(token)

    def _visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in token_keys:
                    _maybe_add(value)
                _visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _visit(item)

    # Encoder implementations default `end_tokens` to "\n" when the config omits it.
    _maybe_add("\n")

    for attr in ("image_encoder", "video_encoder", "sound_encoder"):
        encoder_config = getattr(config, attr, None)
        if isinstance(encoder_config, str):
            try:
                encoder_config = json.loads(encoder_config)
            except Exception:
                continue
        _visit(encoder_config)

    return collected


def _expand2square(pil_img, background_color):
    """Expand a non-square PIL image with padding to make it square."""
    width, height = pil_img.size
    if pil_img.mode == "L":
        background_color = background_color[0]
    if width == height:
        return pil_img
    if width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = PIL.Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from candidate ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_s2_preprocess(image, s2_scales: list[int] | None = None, max_num=12, image_size=384):
    """Dynamically preprocess image using multi-scale S2 tiling."""
    if s2_scales is None:
        s2_scales = [384, 768, 1152]
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    min_num = (s2_scales[-1] // s2_scales[0]) ** 2

    processed_images = []

    for scale in s2_scales[:-1]:
        target_width = image_size * (scale // s2_scales[0])
        target_height = image_size * (scale // s2_scales[0])
        blocks = (scale // s2_scales[0]) ** 2
        resized_img = image.resize((target_width, target_height))
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    return processed_images, (target_aspect_ratio[1], target_aspect_ratio[0])


def _process_image(image_file, data_args, image_folder, enable_dynamic_s2=False):
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = PIL.Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = PIL.Image.open(image_file).convert("RGB")
    else:
        image = image_file
    image = image.convert("RGB")
    if hasattr(data_args.image_processor, "crop_size"):
        crop_size = data_args.image_processor.crop_size
    else:
        assert hasattr(data_args.image_processor, "size")
        crop_size = data_args.image_processor.size
    if "dynamic_s2" in data_args.image_aspect_ratio and enable_dynamic_s2:
        assert crop_size["height"] == crop_size["width"]
        images, block_size = _dynamic_s2_preprocess(
            image, s2_scales=data_args.s2_scales, max_num=data_args.max_tiles, image_size=crop_size["height"]
        )
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images), block_size

    if data_args.image_aspect_ratio == "resize":
        image = image.resize((crop_size["width"], crop_size["height"]))
    elif data_args.image_aspect_ratio == "pad":
        image = _expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image


def _process_images(images, image_processor, model_cfg):
    """Process a batch of images using the model image processor."""
    model_cfg.image_processor = image_processor
    new_images = [_process_image(image, model_cfg, None) for image in images]

    if not all(x.shape == new_images[0].shape for x in new_images):
        raise ValueError("The shape of images in new_images is different!")
    if len(new_images[0].shape) == 4:
        return torch.cat(new_images, dim=0)
    if len(new_images[0].shape) == 3:
        return torch.stack(new_images, dim=0)
    raise ValueError(f"new_images rank does not equal to 4, rank: {len(new_images[0].shape)}")


def _tokenize_conversation(
    messages: Sequence[dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    mm_use_bos_eos_tokens: bool = False,
    add_generation_prompt: bool = False,
    overrides: dict[str, str] | None = None,
    no_system_prompt: bool = False,
    return_ids_only: bool = True,
) -> torch.Tensor:
    for message in messages:
        message["value"] = message["value"].strip()

    conversation = []
    for m in messages:
        message = {}
        if m["from"] == "human":
            message["role"] = "user"
        elif m["from"] == "gpt":
            message["role"] = "assistant"
        elif m["from"] == "system":
            message["role"] = "system"
            if no_system_prompt:
                raise ValueError("message[role]=system is not allowed when no_system_prompt is set to True.")
        else:
            raise ValueError(f"Unexpected sender '{m['from']}' in conversation entry.")

        message["content"] = m["value"]
        if overrides is not None and m["from"] in overrides:
            message["content"] = overrides[m["from"]]
        conversation.append(message)

    if no_system_prompt:
        conversation = [{"role": "system", "content": ""}] + conversation

    text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )

    if mm_use_bos_eos_tokens:

        def add_mm_bos_eos_tokens(text: str) -> str:
            for k in ("image", "video", "sound"):
                _bos, _eos = MM_BOS_EOS_TOKENS[k]
                _media_token = MEDIA_TOKENS[k]
                if _media_token in text:
                    try:
                        text_parts = text.split(_media_token)
                        text_parts[0] = text_parts[0] + _bos
                        text_parts[-1] = _eos + text_parts[-1]
                        text = _media_token.join(text_parts)
                    except Exception:
                        print(f"mm_use_bos_eos_tokens error text: {text}")
            return text

        text = add_mm_bos_eos_tokens(text)

    tokenized = tokenizer(text, return_tensors="pt")
    if return_ids_only:
        return tokenized.input_ids[0]
    return tokenized


def _fetch_image_url_or_fpath(url_or_fpath: str) -> str:
    """Return a local file path for a URL or filesystem path."""
    if url_or_fpath.startswith(("http://", "https://")):
        import tempfile

        import requests

        # Download the image to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(url_or_fpath))

        response = requests.get(url_or_fpath, stream=True)
        response.raise_for_status()

        with open(temp_file, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        return temp_file

    fpath = url_or_fpath.replace("file://", "") if url_or_fpath.startswith("file://") else url_or_fpath
    if not osp.exists(fpath):
        raise ValueError(f"Unsupported image path: {url_or_fpath}")
    if not osp.isfile(fpath):
        raise ValueError(f"Path is not a file: {fpath}")
    return fpath


def _pad_fn(input_ids_list: list[torch.Tensor], padding_value=0, target_len=None, padding_side="left") -> torch.Tensor:
    if not input_ids_list:
        raise ValueError("input_ids_list must not be empty")

    sequences = [ids.squeeze(0) for ids in input_ids_list]

    if padding_side == "right":
        padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    elif padding_side == "left":
        reversed_sequences = [torch.flip(ids, dims=[0]) for ids in sequences]
        padded = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
        padded = torch.flip(padded, dims=[1])
    else:
        raise ValueError(f"Unsupported padding_side: {padding_side}")

    if target_len is not None:
        assert target_len >= padded.shape[1], "target_len must be greater than or equal to max_len"
        if target_len > padded.shape[1]:
            pad_width = target_len - padded.shape[1]
            pad_tensor = padded.new_full((padded.shape[0], pad_width), padding_value)
            if padding_side == "right":
                padded = torch.cat((padded, pad_tensor), dim=1)
            else:
                padded = torch.cat((pad_tensor, padded), dim=1)

    return padded


def _resolve_sound_feature_size(config) -> int:
    sound_tower_cfg = getattr(config, "sound_tower_cfg", None)
    if isinstance(sound_tower_cfg, dict):
        feature_size = sound_tower_cfg.get("num_mel_bins")
    else:
        feature_size = getattr(sound_tower_cfg, "num_mel_bins", None)
    if feature_size is None:
        feature_size = 128
    return int(feature_size)


def _resolve_target_audio_samples(sound: np.ndarray, audio_info, config) -> int:
    sampling_rate = config.audio_sampling_rate
    audio_n_samples = sound.shape[0]
    if isinstance(audio_info, dict) and audio_info.get("new_audio_n_samples") is not None:
        return int(audio_info["new_audio_n_samples"])

    target = int(np.ceil(audio_n_samples / (sampling_rate * 30)) * (sampling_rate * 30))
    if config.audio_chunk_length and not (
        isinstance(config.audio_chunk_length, str) and "max" in config.audio_chunk_length
    ):
        target = min(target, int(config.audio_chunk_length) * sampling_rate)
    return int(target)


def _extract_sound_features(sound_media: list, audio_infos: list | None, config) -> list:
    if audio_infos is None:
        audio_infos = []
    if audio_infos and len(audio_infos) != len(sound_media):
        raise ValueError("The number of audio info does not match the number of audio samples.")

    feature_size = _resolve_sound_feature_size(config)
    sampling_rate = config.audio_sampling_rate
    hop_length = config.audio_hop_length
    new_media = []

    for idx, sound in enumerate(sound_media):
        audio_info = audio_infos[idx] if idx < len(audio_infos) else None
        if isinstance(sound, dict) and "input_features" in sound:
            stft_features = sound
        else:
            if isinstance(sound, torch.Tensor):
                audio = sound.detach().cpu().float().numpy()
            else:
                audio = np.asarray(sound, dtype=np.float32)
            if audio.ndim != 1:
                audio = np.squeeze(audio)
            if audio.ndim != 1:
                raise ValueError(f"Expected mono waveform for sound input, got shape {audio.shape}.")

            cur_audio_n_samples = _resolve_target_audio_samples(audio, audio_info, config)
            cur_audio_duration = cur_audio_n_samples // sampling_rate
            whisper_feature_extractor = WhisperFeatureExtractor(
                feature_size=feature_size,
                chunk_length=cur_audio_duration,
                sampling_rate=sampling_rate,
                hop_length=hop_length,
            )
            audio = whisper.pad_or_trim(audio, length=cur_audio_n_samples)
            stft_features = whisper_feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt",
            )

            if isinstance(audio_info, dict):
                audio_info["new_audio_chunk_length"] = cur_audio_duration
                audio_info["new_audio_n_samples"] = cur_audio_n_samples
                audio_info["audio_end_sample_sec"] = audio_info["audio_start_sec"] + cur_audio_duration
                audio_info["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]

        if isinstance(audio_info, dict) and "new_audio_n_stft_frames" not in audio_info:
            audio_info["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]
        new_media.append(stft_features)

    return new_media


def _extract_value_from_conv(chat):
    value = []
    if isinstance(chat["content"], str):
        value.append(chat["content"])
        return value

    # otherwise, it's a list of content
    for content in chat["content"]:
        if content["type"] == "image":
            if "path" in content:
                # VILA style, can be either filepath or http url
                value.append(load_image(content["path"]))
            elif "image" in content:
                # Qwen style
                value.append(load_image(content["image"]))
            elif "image_pil" in content:
                # Qwen style
                assert isinstance(content["image_pil"], PIL.Image.Image), "Type of image_pil must be PIL.Image.Image"
                value.append(content["image_pil"])
            else:
                raise ValueError(f"Type = `image` , but no `path` or `image` in  {chat['content']}")
        elif content["type"] == "video":
            if "video" in content:
                # Qwen style
                value.append(Video(_fetch_image_url_or_fpath(content["video"])))
            else:
                raise ValueError(f"Type = `video` , but no `video` in {chat['content']}")
        elif content["type"] == "text":
            value.append(content["text"])
        elif content["type"] in ("audio", "sound"):
            key = "audio" if content["type"] == "audio" else "sound"
            value.append(Sound(_fetch_image_url_or_fpath(content[key])))
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")
    return value


class OmniVinciProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class OmniVinciProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = []

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, config=None, padding_side="left", **kwargs
    ):
        self.image_token = MEDIA_TOKENS["image"]
        self.video_token = MEDIA_TOKENS["video"]
        self.sound_token = MEDIA_TOKENS["sound"]
        self.config = config
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.tokenizer.padding_side = padding_side

        # Use <|endoftext|> token as padding token for Qwen models
        self.pad_token_id = self.tokenizer("<|endoftext|>").input_ids[0]
        self.eos_token_id = self.tokenizer.eos_token_id

        if self.config is not None:
            self.config.padding_side = self.padding_side
            self.config.pad_token_id = self.pad_token_id
            self.config.eos_token_id = self.eos_token_id
            if getattr(self.config, "bos_token_id", None) is None:
                self.config.bos_token_id = self.tokenizer.bos_token_id
            if getattr(self.config, "model_max_length", None) is None:
                self.config.model_max_length = getattr(self.tokenizer, "model_max_length", 2048)

            media_token_ids = {}
            for name, token in self.config.media_tokens.items():
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is None or token_id < 0:
                    tokenized = self.tokenizer(token, add_special_tokens=False).input_ids
                    if len(tokenized) != 1:
                        raise ValueError(f"Media token `{token}` must map to a single tokenizer id.")
                    token_id = tokenized[0]
                media_token_ids[name] = int(token_id)
            self.config.media_token_ids = media_token_ids

            self.config.encoder_text_token_ids = {
                token_text: [int(token_id) for token_id in self.tokenizer(token_text).input_ids]
                for token_text in _collect_encoder_boundary_tokens(self.config)
            }

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __repr__(self):
        return f"OmniVinciProcessor(image_processor=SigLip, tokenizer={self.tokenizer}, config={self.config})"

    def __call__(
        self,
        conversation=None,
        **kwargs: Unpack[OmniVinciProcessorKwargs],
    ) -> BatchFeature:
        """
        The `conv` will be look like
        [
            {
                'from': 'human',
                'value': [
                    <transformers_modules.NVILA-Lite-2B-hf-preview.media.Image object at 0x154e68e4c460>,
                    'What are the common elements in these pictures?'
                ]
            }
        ]
        and `conversation` will be a list of such `conv`s
        """
        if kwargs.get("text") is not None:
            conversation = kwargs.get("text")
        assert conversation is not None, "`conversation` or `text` is required"
        padding_side = kwargs.get("padding_side", self.padding_side)

        input_ids_list = []
        media = defaultdict(list)
        media_config = defaultdict(dict)
        for conv in conversation:
            feat = self.__single_call__(conv, **kwargs)
            input_ids_list.append(feat.input_ids)
            for name in feat.media:
                media[name] += feat.media[name]
            for name in feat.media_config:
                media_config[name].update(feat.media_config[name])

        # pad the input_ids to batchfy
        input_ids = _pad_fn(
            input_ids_list,
            padding_value=self.pad_token_id,
            padding_side=padding_side,
        )
        # Ignore the pad token in the attention mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask[input_ids == self.pad_token_id] = False
        bdata = BatchFeature(
            data={
                # "input_texts": input_texts,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "media": media,
                "media_config": media_config,
            }
        )
        return bdata

    def __single_call__(
        self,
        conversation,
        **kwargs: Unpack[OmniVinciProcessorKwargs],
    ) -> BatchFeature:
        conversation = copy.deepcopy(conversation)
        media = extract_media(conversation, self.config)
        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio == "dynamic_s2":
                    self.config.image_processor = self.image_processor
                    if isinstance(self.config.s2_scales, str):
                        self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                    images, block_sizes = _process_image(media["image"][0], self.config, None, enable_dynamic_s2=True)
                    images = images.half()
                    media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = _process_images(media["image"], self.image_processor, self.config).half()
                media[name] = list(images)
            elif name == "video":
                media[name] = [
                    _process_images(images, self.image_processor, self.config).half() for images in media[name]
                ]
            elif name == "sound":
                sounds = media["sound"]
                audio_infos = media.get("audio_info", [])
                media[name] = _extract_sound_features(list(sounds), audio_infos, self.config)
            elif name == "video_info":
                media[name] = [media["video_info"]]
            elif name == "audio_info":
                media[name] = [media["audio_info"]]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        inputs = _tokenize_conversation(
            conversation,
            self.tokenizer,
            mm_use_bos_eos_tokens=self.config.mm_use_bos_eos_tokens,
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
        )

        input_ids = inputs.unsqueeze(0)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "media": media,
                "media_config": media_config,
            }
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def convert_gpt_conv_to_vila_conv(self, conversation):
        vila_conv = []
        role_map = {"user": "human", "system": "system", "assistant": "gpt"}
        for chat in conversation:
            role = chat["role"]
            if role not in role_map:
                raise ValueError(f"Unsupported role: {role} in chat {chat}")
            vila_conv.append({"from": role_map[role], "value": _extract_value_from_conv(chat)})

        return vila_conv

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        return_dict=True,
        **kwargs,
    ):
        is_batched = (
            isinstance(conversation, (list, tuple))
            and len(conversation) > 0
            and isinstance(conversation[0], (list, tuple))
        )
        converted = (
            [self.convert_gpt_conv_to_vila_conv(conv) for conv in conversation]
            if is_batched
            else self.convert_gpt_conv_to_vila_conv(conversation)
        )

        if not tokenize:
            return converted

        batched_conversations = converted if is_batched else [converted]
        outputs = self(conversation=batched_conversations, add_generation_prompt=add_generation_prompt, **kwargs)
        if return_dict:
            return outputs
        return outputs["input_ids"]


__all__ = [
    "OmniVinciProcessor",
    "OmniVinciProcessorKwargs",
]
