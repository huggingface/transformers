# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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

import os
import re
from dataclasses import dataclass
from typing import Any, cast

import torch

from ... import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from ...utils import is_torchaudio_available, logging
from .configuration_moss_tts_delay import MossTTSDelayConfig
from .tts_robust_normalizer_single_script import normalize_tts_text


logger = logging.get_logger(__name__)

if is_torchaudio_available():
    import torchaudio


def _require_torchaudio():
    if not is_torchaudio_available():
        raise ImportError("MossTTSDelayProcessor requires torchaudio to load or resample waveform files.")


AUDIO_PLACEHOLDER = "<|audio|>"


@dataclass
class Message:
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class UserMessage(Message):
    text: str | None = None
    reference: list[str | torch.Tensor | None] | None = None
    instruction: str | None = None
    tokens: int | None = None
    quality: str | None = None
    sound_event: str | None = None
    ambient_sound: str | None = None
    language: str | None = None

    def __post_init__(self):
        template = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""

        audio_codes_list = []
        if self.reference is None:
            reference = "None"
        elif isinstance(self.reference, list):
            reference = []
            for speaker_idx, speaker_reference in enumerate(self.reference):
                if speaker_reference is not None:
                    reference.append(f"[S{speaker_idx + 1}]:\n{AUDIO_PLACEHOLDER}")
            reference = "\n".join(reference)
            audio_codes_list = [
                speaker_reference for speaker_reference in self.reference if speaker_reference is not None
            ]
        else:
            raise TypeError("`reference` should be exactly a list when it is not None.")

        content = (
            template.replace("{reference}", str(reference))
            .replace("{instruction}", str(self.instruction))
            .replace("{tokens}", str(self.tokens))
            .replace("{quality}", str(self.quality))
            .replace("{sound_event}", str(self.sound_event))
            .replace("{ambient_sound}", str(self.ambient_sound))
            .replace("{language}", str(self.language))
            .replace("{text}", str(self.text))
        )

        self._content = content
        self._audio_codes_list = audio_codes_list

    def to_dict(self):
        return {
            "role": "user",
            "content": self._content,
            "audio_codes_list": self._audio_codes_list,
        }


@dataclass
class AssistantMessage(Message):
    audio_codes_list: list[str | torch.Tensor]
    content: str = AUDIO_PLACEHOLDER

    def to_dict(self):
        return {
            "role": "assistant",
            "content": self.content,
            "audio_codes_list": self.audio_codes_list,
        }


USER_MESSAGE_FIELDS = (
    "text",
    "reference",
    "instruction",
    "tokens",
    "quality",
    "sound_event",
    "ambient_sound",
    "language",
)


class MossTTSDelayProcessor(ProcessorMixin):
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "AutoModel"

    tokenizer: PreTrainedTokenizerBase
    audio_tokenizer: Any

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        audio_tokenizer: Any = None,
        model_config: MossTTSDelayConfig | None = None,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, audio_tokenizer=audio_tokenizer, **kwargs)

        # Explicit assignments for type-checkers; ProcessorMixin sets these too.
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        if model_config is None:
            model_config = MossTTSDelayConfig()
        self.model_config = model_config

        self.imstart_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.newline_token_id = 198

        def _id_to_token(token_id: int) -> str:
            tok = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(tok, list):
                return tok[0] if len(tok) > 0 else ""
            return cast(str, tok)

        self.audio_user_slot_token = _id_to_token(self.model_config.audio_user_slot_token_id)
        self.audio_assistant_gen_slot_token = _id_to_token(self.model_config.audio_assistant_gen_slot_token_id)
        self.audio_assistant_delay_slot_token = _id_to_token(self.model_config.audio_assistant_delay_slot_token_id)
        self.audio_start_token = _id_to_token(self.model_config.audio_start_token_id)
        self.audio_end_token = _id_to_token(self.model_config.audio_end_token_id)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        kwargs.pop("_from_auto", None)

        audio_tokenizer_name_or_path = kwargs.pop("codec_path", None)
        if audio_tokenizer_name_or_path is None:
            processor_lookup_kwargs = dict(kwargs)
            try:
                processor_dict, _ = cls.get_processor_dict(
                    pretrained_model_name_or_path,
                    **processor_lookup_kwargs,
                )
                audio_tokenizer_name_or_path = processor_dict.get("audio_tokenizer_name_or_path")
                audio_tokenizer_dict = processor_dict.get("audio_tokenizer", {})
                if isinstance(audio_tokenizer_dict, dict):
                    audio_tokenizer_name_or_path = (
                        audio_tokenizer_dict.get("audio_tokenizer_name_or_path") or audio_tokenizer_name_or_path
                    )
            except Exception:
                audio_tokenizer_name_or_path = None
        if audio_tokenizer_name_or_path is None:
            audio_tokenizer_name_or_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

        model_config = cast(
            MossTTSDelayConfig,
            AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                trust_remote_code=trust_remote_code,
                **kwargs,
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        audio_tokenizer = AutoModel.from_pretrained(
            audio_tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return cls(
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            model_config=model_config,
            **kwargs,
        )

    def __call__(self, *args, **kwargs) -> BatchFeature:
        conversations = args[0] if len(args) > 0 else kwargs.pop("conversations")
        mode: str = kwargs.pop("mode", "generation")
        apply_chat_template: bool = kwargs.pop("apply_chat_template", True)
        n_vq: int | None = kwargs.pop("n_vq", None)

        # Common ProcessorMixin kwargs that we ignore because we always return torch tensors.
        kwargs.pop("return_tensors", None)
        kwargs.pop("padding", None)
        kwargs.pop("truncation", None)

        """
        mode only works when a Message is converted to a dict.
        """

        if mode not in {"generation", "continuation", "computing_loss"}:
            raise RuntimeError

        if isinstance(conversations, (Message, dict)):
            conversations = [conversations]

        truncation = False
        if mode == "continuation":
            truncation = True

        input_ids_list = []
        for conversation in conversations:
            if isinstance(conversation, (Message, dict)):
                conversation = [conversation]

            # Normalize early so downstream logic always deals with dict messages.
            conversation = [self._normalize_message(m) for m in conversation]

            if (mode == "generation") ^ (len(conversation) % 2 != 0):
                raise ValueError

            if (mode == "generation") ^ (conversation[-1]["role"] == "user"):
                raise ValueError

            unified_codes = []
            for message_idx, message in enumerate(conversation):
                if apply_chat_template:
                    add_generation_prompt = mode == "generation" and message_idx == len(conversation) - 1
                    try:
                        content = self.tokenizer.apply_chat_template(
                            [{"role": message["role"], "content": message["content"]}],
                            add_generation_prompt=add_generation_prompt,
                            tokenize=False,
                        )
                    except TypeError:
                        try:
                            content = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": message["role"],
                                        "content": message["content"],
                                    }
                                ],
                                add_generation_prompt=add_generation_prompt,
                            )
                        except Exception:
                            logger.warning("apply_chat_template failed; fallback to raw content.")
                            content = message["content"]
                else:
                    content = message["content"]

                if not isinstance(content, str):
                    content = str(content)

                # Batch-encode all path-based references in one call when possible.
                # This ensures we actually exercise audio_tokenizer.batch_encode for multi-reference prompts,
                # instead of repeatedly calling it with batch=1.
                raw_audio_items = message.get("audio_codes_list", [])

                audio_codes_list: list[torch.Tensor] = []
                if len(raw_audio_items) > 0:
                    encoded_items: list[torch.Tensor | None] = [None] * len(raw_audio_items)
                    paths: list[str] = []
                    path_positions: list[int] = []

                    for idx, item in enumerate(raw_audio_items):
                        if isinstance(item, torch.Tensor):
                            if n_vq is not None and item.shape[1] != n_vq:
                                raise RuntimeError(
                                    "audio_codes's n_vq is not equal to the parameter `n_vq`. Your can set the parameter `n_vq` as None if you have already tokenzied the wavs."
                                )
                            encoded_items[idx] = item
                            continue

                        if isinstance(item, (str, os.PathLike)):
                            paths.append(str(item))
                            path_positions.append(idx)
                            continue

                        raise TypeError("Each audio item must be a torch.Tensor of codes or a path-like string.")

                    if len(paths) > 0:
                        encoded_from_paths = self.encode_audios_from_path(paths, n_vq)
                        if len(encoded_from_paths) != len(paths):
                            raise RuntimeError("encode_audios_from_path returned an unexpected number of items.")
                        for pos, codes in zip(path_positions, encoded_from_paths):
                            encoded_items[pos] = codes

                    audio_codes_list = [cast(torch.Tensor, t) for t in encoded_items]
                unified_codes.append(self._get_unified_codes(message["role"], content, audio_codes_list, truncation))

            unified_codes = torch.cat(unified_codes)
            input_ids_list.append(unified_codes)

        return BatchFeature(data=self._pad(input_ids_list))

    @staticmethod
    def build_user_message(
        text: str | None = None,
        reference: list[str | torch.Tensor | None] | None = None,
        instruction: str | None = None,
        tokens: int | None = None,
        quality: str | None = None,
        sound_event: str | None = None,
        ambient_sound: str | None = None,
        language: str | None = None,
    ) -> dict:
        if reference is not None and not isinstance(reference, list):
            reference = [reference]
        text = normalize_tts_text(text)
        return UserMessage(
            text=text,
            reference=reference,
            instruction=instruction,
            tokens=tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
        ).to_dict()

    @staticmethod
    def build_assistant_message(
        audio_codes_list: list[str | torch.Tensor],
        content: str = AUDIO_PLACEHOLDER,
    ) -> dict:
        return AssistantMessage(
            audio_codes_list=audio_codes_list,
            content=content,
        ).to_dict()

    def _normalize_message(self, message: Message | dict) -> dict:
        if isinstance(message, Message):
            return message.to_dict()
        if not isinstance(message, dict):
            raise TypeError("Each message must be a Message or dict.")
        if "role" not in message:
            raise ValueError("Message dict must include a 'role' field.")
        if "content" in message and "audio_codes_list" in message:
            return message
        role = message["role"]
        if role == "user":
            kwargs = {key: message.get(key) for key in USER_MESSAGE_FIELDS}
            return self.build_user_message(**kwargs)
        if role == "assistant":
            return self.build_assistant_message(
                audio_codes_list=message.get("audio_codes_list", []),
                content=message.get("content", AUDIO_PLACEHOLDER),
            )
        raise ValueError(f"Unsupported role: {role}")

    def _pad(self, input_ids_list: list[torch.Tensor]):
        device = input_ids_list[0].device
        lengths = torch.tensor([w.shape[0] for w in input_ids_list], device=device)
        pad_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.model_config.audio_pad_code,
            padding_side="left",
        )
        other_channel_mask = (pad_input_ids.shape[1] - lengths).unsqueeze(1) > torch.arange(
            pad_input_ids.shape[1], device=device
        ).unsqueeze(0)
        pad_input_ids[..., 0][other_channel_mask] = self.model_config.pad_token_id
        attention_mask = torch.zeros(pad_input_ids.shape[0], pad_input_ids.shape[1], device=device)
        attention_mask[~other_channel_mask] = 1
        attention_mask = attention_mask.bool()
        return {
            "input_ids": pad_input_ids,  # [batch_size, seqlen, n_vq]
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _replace_audio_placeholders(
        content: str,
        lengths: list[int],
        n_vq: int,
        gen_slot_token: str,
        delay_slot_token: str,
        audio_start_token: str,
        audio_end_token: str,
    ) -> str:
        if n_vq < 1:
            raise ValueError(f"n_vq must be >= 1, got {n_vq}")

        num_placeholders = content.count(AUDIO_PLACEHOLDER)
        if num_placeholders != len(lengths):
            raise ValueError(
                f"Number of {AUDIO_PLACEHOLDER} ({num_placeholders}) does not match lengths ({len(lengths)})"
            )

        def build_audio_block(length: int) -> str:
            if length < 0:
                raise ValueError(f"length must be >= 0, got {length}")

            if length == 0:
                return f"{audio_start_token}{audio_end_token}"

            step_tokens = gen_slot_token * length + (delay_slot_token * (n_vq - 1))
            return f"{audio_start_token}{step_tokens}{audio_end_token}"

        lengths_iter = iter(lengths)

        def replacer(match: re.Match) -> str:
            length = next(lengths_iter)
            return build_audio_block(length)

        result = re.sub(re.escape(AUDIO_PLACEHOLDER), replacer, content)

        return result

    @staticmethod
    def _merge_consecutive_audio_placeholders(
        content: str,
        audio_codes_list: list[torch.Tensor],
    ) -> tuple[str, list[torch.Tensor]]:
        matches = list(re.finditer(re.escape(AUDIO_PLACEHOLDER), content))
        if len(matches) <= 1:
            return content, audio_codes_list

        if len(matches) != len(audio_codes_list):
            raise ValueError("Audio placeholders do not match the provided audio codes list.")

        new_audio_codes_list = []
        new_parts = []
        last_pos = 0
        i = 0
        while i < len(matches):
            j = i
            while j + 1 < len(matches) and content[matches[j].end() : matches[j + 1].start()].strip() == "":
                j += 1

            new_parts.append(content[last_pos : matches[i].start()])
            new_parts.append(AUDIO_PLACEHOLDER)
            last_pos = matches[j].end()

            if j == i:
                new_audio_codes_list.append(audio_codes_list[i])
            else:
                new_audio_codes_list.append(torch.cat(audio_codes_list[i : j + 1], dim=0))

            i = j + 1

        new_parts.append(content[last_pos:])
        return "".join(new_parts), new_audio_codes_list

    @staticmethod
    def apply_delay_pattern(codes: torch.Tensor, pad_code: int) -> torch.Tensor:
        delayed_tokens = torch.full(
            (codes.shape[0] + codes.shape[1] - 1, codes.shape[1]),
            pad_code,
            device=codes.device,
            dtype=codes.dtype,
        )
        for i in range(codes.shape[1]):
            delayed_tokens[i : i + codes.shape[0], i] = codes[:, i]
        return delayed_tokens

    @staticmethod
    def apply_de_delay_pattern(delay_codes: torch.Tensor) -> torch.Tensor:
        tokens = torch.full(
            (delay_codes.shape[0] - delay_codes.shape[1] + 1, delay_codes.shape[1]),
            0,
            device=delay_codes.device,
            dtype=delay_codes.dtype,
        )
        for i in range(delay_codes.shape[1]):
            tokens[:, i] = delay_codes[i : i + tokens.shape[0], i]
        return tokens

    def _get_unified_codes(
        self,
        role: str,
        content: str,
        audio_codes_list: list[torch.Tensor],
        truncation: bool,
    ) -> torch.Tensor:
        """
        此时的 content 已经是带上了对话格式
        """
        if role == "user":
            audio_gen_slot_token = audio_delay_slot_token = self.audio_user_slot_token
            truncation = False
        else:
            audio_gen_slot_token = self.audio_assistant_gen_slot_token
            audio_delay_slot_token = self.audio_assistant_delay_slot_token

        if len(audio_codes_list):
            n_vq = audio_codes_list[0].shape[1]
        else:
            n_vq = self.model_config.n_vq

        if len(audio_codes_list) > 1 and AUDIO_PLACEHOLDER in content:
            content, audio_codes_list = self._merge_consecutive_audio_placeholders(content, audio_codes_list)
        content = self._replace_audio_placeholders(
            content=content,
            lengths=[len(audio_codes) for audio_codes in audio_codes_list],
            n_vq=n_vq,
            gen_slot_token=audio_gen_slot_token,
            delay_slot_token=audio_delay_slot_token,
            audio_start_token=self.audio_start_token,
            audio_end_token=self.audio_end_token,
        )
        text_codes = torch.tensor(
            self.tokenizer.encode(content),
            device=audio_codes_list[0].device if audio_codes_list else None,
        )

        audio_start_indices = torch.where(text_codes == self.model_config.audio_start_token_id)[0]
        audio_end_indices = torch.where(text_codes == self.model_config.audio_end_token_id)[0]
        if len(audio_start_indices) != len(audio_codes_list) or len(audio_end_indices) != len(audio_codes_list):
            raise ValueError("Audio placeholders do not match the provided audio codes list.")

        delay_audio_codes_list = []
        if len(audio_codes_list) == 0:
            delay_audio_codes_list = torch.full(
                (len(text_codes), n_vq),
                self.model_config.audio_pad_code,
                device=text_codes.device,
                dtype=text_codes.dtype,
            )
        else:
            prefix_idx = 0
            for audio_start_idx_t, audio_end_idx_t, audio_codes in zip(
                audio_start_indices, audio_end_indices, audio_codes_list
            ):
                audio_start_idx = int(audio_start_idx_t.item())
                audio_end_idx = int(audio_end_idx_t.item())
                delay_audio_codes = self.apply_delay_pattern(audio_codes, self.model_config.audio_pad_code)
                pad_codes = torch.full(
                    (audio_start_idx - prefix_idx + 1, n_vq),
                    self.model_config.audio_pad_code,
                    device=audio_codes.device,
                    dtype=audio_codes.dtype,
                )
                delay_audio_codes_list.extend([pad_codes, delay_audio_codes])
                prefix_idx = audio_end_idx

            if truncation:
                delay_audio_codes_list[-1] = delay_audio_codes_list[-1][: -(n_vq - 1), :]
            else:
                last_audio_end_idx = int(audio_end_indices[-1].item())
                pad_codes = torch.full(
                    (len(text_codes) - last_audio_end_idx, n_vq),
                    self.model_config.audio_pad_code,
                    device=audio_codes_list[0].device,
                    dtype=audio_codes_list[0].dtype,
                )
                delay_audio_codes_list.append(pad_codes)

            delay_audio_codes_list = torch.cat(delay_audio_codes_list)

        if text_codes.shape[0] != delay_audio_codes_list.shape[0]:
            text_codes = text_codes[: delay_audio_codes_list.shape[0]]

        unified_codes = torch.cat([text_codes.unsqueeze(1), delay_audio_codes_list], dim=1)
        return unified_codes

    def _parse_text_codes(self, start_length, text_codes):
        text = cast(str, self.tokenizer.decode(text_codes))
        prefix = cast(str, self.tokenizer.decode(text_codes[:start_length]))
        text = text[len(prefix) :]

        AUDIO_PATTERN = re.compile(
            rf"(?:{self.audio_start_token})?"
            rf"(?:{self.audio_assistant_gen_slot_token})*"
            rf"(?:{self.audio_assistant_delay_slot_token})*"
            rf"{self.audio_end_token}"
        )

        def normalize_audio_segments(text: str) -> str:
            def repl(match: re.Match) -> str:
                seg = match.group(0)
                # Replace with <|audio|> if gen_slot is present in the segment;
                if self.audio_assistant_gen_slot_token in seg:
                    return AUDIO_PLACEHOLDER
                # Otherwise, remove it.
                return ""

            return AUDIO_PATTERN.sub(repl, text)

        return normalize_audio_segments(text)

    def _parse_audio_codes(self, start_length, audio_codes):
        # De-delay back to [T', n_vq]
        audio_codes = self.apply_de_delay_pattern(audio_codes)

        # Rows that are all pad are separators between real audio segments.
        is_pad = (audio_codes == self.model_config.audio_pad_code).all(dim=1)
        non_pad = ~is_pad
        if not non_pad.any():
            return []

        idx = torch.nonzero(non_pad).squeeze(1)
        breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
        if breaks.numel() == 0:
            segments_idx = [idx]
        else:
            segments_idx = torch.split(idx, breaks.tolist())

        audio_codes_list = [audio_codes[s] for s in segments_idx]

        # Batch-decode all audio segments together.
        decoded_audio_list = self.decode_audio_codes(audio_codes_list)

        # Keep codec causal context by decoding the whole first segment first,
        # then trim at waveform level according to start_length ratio.
        if start_length > 0 and len(audio_codes_list) > 0 and len(decoded_audio_list) > 0:
            first_codes_length = audio_codes_list[0].shape[0]
            if first_codes_length > 0:
                trim_ratio = max(0.0, min(float(start_length) / float(first_codes_length), 1.0))
                first_audio = decoded_audio_list[0]
                if trim_ratio >= 1.0:
                    decoded_audio_list = decoded_audio_list[1:]
                elif trim_ratio > 0.0:
                    trim_samples = int(first_audio.shape[-1] * trim_ratio)
                    decoded_audio_list[0] = first_audio[..., trim_samples:]

        return decoded_audio_list

    def decode(self, output: list[tuple[int, torch.Tensor]]):
        """
        1. 这里不管怎样，都需要一个完整的 assistant generation ids;
        2. 支持从任意位置进行截断；
        """

        genearted_messages = []
        for start_length, generation_ids in output:
            content = self._parse_text_codes(start_length, generation_ids[:, 0])
            audio_codes_list = self._parse_audio_codes(start_length, generation_ids[:, 1:])
            if content == "":
                message = None
            else:
                message = AssistantMessage(
                    content=content,
                    audio_codes_list=cast(list[str | torch.Tensor], audio_codes_list),
                )
            genearted_messages.append(message)
        return genearted_messages

    @staticmethod
    def loudness_normalize(
        wav: torch.Tensor,
        target_dbfs: float = -20,
        gain_range: tuple[float, float] = (-3.0, 3.0),
    ) -> torch.Tensor:
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = float(target_dbfs - current_dbfs)
        gain = max(gain_range[0], min(gain, gain_range[1]))
        factor = 10.0 ** (gain / 20.0)
        return wav * factor

    def _get_audio_tokenizer_device(self) -> torch.device:
        """Best-effort device inference for `self.audio_tokenizer`.

        Notes:
        - Old TAC wrapper exposed `.device`, but standard `torch.nn.Module` does not.
        - New MossAudioTokenizerModel is a `PreTrainedModel`; parameters define its device.
        """

        audio_tokenizer = getattr(self, "audio_tokenizer", None)
        if audio_tokenizer is None:
            logger.warning("audio_tokenizer is not set on processor. Using CPU as default.")
            return torch.device("cpu")

        device_attr = getattr(audio_tokenizer, "device", None)
        if isinstance(device_attr, torch.device):
            return device_attr

        try:
            return next(audio_tokenizer.parameters()).device
        except StopIteration:
            # No parameters (shouldn't happen for real models); default to CPU.
            logger.warning("No parameters found on audio_tokenizer. Using CPU as default.")
            return torch.device("cpu")

    def encode_audios_from_wav(
        self,
        wav_list: list[torch.Tensor],
        sampling_rate: int,
        n_vq: int | None = None,
    ):
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not set on processor.")
        audio_tokenizer = self.audio_tokenizer

        if isinstance(wav_list, torch.Tensor):
            wav_list = [wav_list]
        wav_list_ = []
        resample = False
        if sampling_rate != self.model_config.sampling_rate:
            resample = True
        device = self._get_audio_tokenizer_device()
        for wav in wav_list:
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            if resample:
                _require_torchaudio()
                wav = torchaudio.functional.resample(
                    waveform=wav,
                    orig_freq=sampling_rate,
                    new_freq=self.model_config.sampling_rate,
                )
            wav = wav.to(device)
            wav_list_.append(self.loudness_normalize(wav.squeeze(0)))

        # New MossAudioTokenizerModel API: prefer batch_encode(list[wav])
        if hasattr(audio_tokenizer, "batch_encode"):
            enc = audio_tokenizer.batch_encode(wav_list_, num_quantizers=n_vq)
            audio_codes = enc.audio_codes  # (NQ, B, T)
            audio_codes_lengths = enc.audio_codes_lengths  # (B,)
        else:
            # Fallback: use encode() with explicit padding.
            max_len = max(int(wav.shape[-1]) for wav in wav_list_)
            input_values = torch.zeros(len(wav_list_), 1, max_len, device=device, dtype=torch.float32)
            padding_mask = torch.zeros(len(wav_list_), max_len, device=device, dtype=torch.bool)
            for i, wav in enumerate(wav_list_):
                this_len = int(wav.shape[-1])
                input_values[i, 0, :this_len] = wav
                padding_mask[i, :this_len] = True
            enc = audio_tokenizer.encode(
                input_values,
                padding_mask=padding_mask,
                num_quantizers=n_vq,
                return_dict=True,
            )
            audio_codes = enc.audio_codes
            audio_codes_lengths = enc.audio_codes_lengths

        if audio_codes is None or audio_codes_lengths is None:
            raise RuntimeError("audio_tokenizer.encode() returned empty outputs (audio_codes/audio_codes_lengths).")

        # Keep processor's historical contract: list[Tensor] with shape (T, NQ)
        # and on CPU (so downstream text/audio packing remains device-agnostic).
        codes_list: list[torch.Tensor] = []
        for i in range(int(audio_codes.shape[1])):
            length_i = int(audio_codes_lengths[i].item())
            codes_i = audio_codes[:, i, :length_i].transpose(0, 1).contiguous().to(torch.long).cpu()
            codes_list.append(codes_i)
        return codes_list

    def encode_audios_from_path(self, wav_path_list: str | list[str], n_vq: int | None = None):
        if isinstance(wav_path_list, str):
            wav_path_list = [wav_path_list]

        if len(wav_path_list) == 0:
            raise ValueError("Empty wav_path_list")

        # Load + (if needed) resample each wav independently, so callers can
        # pass a heterogeneous batch of files while still benefiting from
        # audio_tokenizer.batch_encode.
        target_sr = int(self.model_config.sampling_rate)
        wav_list: list[torch.Tensor] = []
        for wav_path in wav_path_list:
            _require_torchaudio()
            wav, sr = torchaudio.load(wav_path)
            if int(sr) != target_sr:
                wav = torchaudio.functional.resample(
                    waveform=wav,
                    orig_freq=int(sr),
                    new_freq=target_sr,
                )
            wav_list.append(wav)

        return self.encode_audios_from_wav(wav_list, target_sr, n_vq)

    def decode_audio_codes(self, audio_tokens_list: torch.Tensor | list[torch.Tensor]):
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not set on processor.")
        audio_tokenizer = self.audio_tokenizer

        if isinstance(audio_tokens_list, torch.Tensor):
            audio_tokens_list = [audio_tokens_list]
        if len(audio_tokens_list) == 0:
            return []

        device = self._get_audio_tokenizer_device()

        # Processor uses (T, NQ); MossAudioTokenizer expects (NQ, T) (or (NQ, B, T)).
        codes_list = [
            codes.transpose(0, 1).contiguous().to(device=device, dtype=torch.long) for codes in audio_tokens_list
        ]

        # Fallback: pad to (NQ, B, T) + mask, then decode.
        nq = int(codes_list[0].shape[0])
        max_t = max(int(c.shape[1]) for c in codes_list)
        audio_codes = torch.zeros(nq, len(codes_list), max_t, device=device, dtype=torch.long)
        padding_mask = torch.zeros(len(codes_list), max_t, device=device, dtype=torch.bool)
        for i, c in enumerate(codes_list):
            t = int(c.shape[1])
            audio_codes[:, i, :t] = c
            padding_mask[i, :t] = True
        dec = audio_tokenizer.decode(audio_codes, padding_mask=padding_mask, return_dict=True, chunk_duration=8)
        audio = dec.audio
        audio_lengths = dec.audio_lengths

        if audio is None or audio_lengths is None:
            raise RuntimeError("audio_tokenizer.decode() returned empty outputs (audio/audio_lengths).")

        # Return historical contract: list of 1D waveforms (T,)
        wav_list: list[torch.Tensor] = []
        for i in range(int(audio.shape[0])):
            length_i = int(audio_lengths[i].item())
            wav = audio[i, 0, :length_i].contiguous().to(torch.float32).cpu()
            wav_list.append(wav)
        return wav_list


__all__ = ["AssistantMessage", "Message", "MossTTSDelayProcessor", "UserMessage"]
