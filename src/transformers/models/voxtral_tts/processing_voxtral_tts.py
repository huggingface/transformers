# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_mistral_common import MistralCommonBackend
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.hub import cached_file
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

VOICE_EMBEDDINGS_NAME = "voice_embeddings.safetensors"


@auto_docstring
@requires(backends=("mistral-common", "torch"))
class VoxtralTtsProcessor(ProcessorMixin):
    text_kwargs = {
        "padding": True,
        "add_special_tokens": False,
    }

    def __init__(self, tokenizer, voice_embeddings: dict[str, "torch.Tensor"] | None = None):
        r"""
        voice_embeddings (`dict[str, torch.Tensor]`, *optional*):
            Preset voices, mapping each voice name to its embeddings of shape `(num_audio_tokens, hidden_size)`.
            Loaded from (and saved to) `voice_embeddings.safetensors` next to the processor files.
        """
        if not isinstance(tokenizer, MistralCommonBackend):
            raise ValueError("`tokenizer` must be a `MistralCommonBackend` tokenizer.")
        super().__init__(tokenizer)
        self.voice_embeddings = voice_embeddings if voice_embeddings is not None else {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        hub_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "force_download", "proxies", "token", "revision", "local_files_only", "subfolder")
            if key in kwargs
        }
        voice_embeddings_file = cached_file(
            pretrained_model_name_or_path,
            VOICE_EMBEDDINGS_NAME,
            _raise_exceptions_for_missing_entries=False,
            **hub_kwargs,
        )
        if voice_embeddings_file is not None:
            from safetensors.torch import load_file

            processor.voice_embeddings = load_file(voice_embeddings_file)
        return processor

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        if self.voice_embeddings:
            from safetensors.torch import save_file

            os.makedirs(save_directory, exist_ok=True)
            voice_embeddings = {name: emb.contiguous() for name, emb in self.voice_embeddings.items()}
            save_file(voice_embeddings, os.path.join(save_directory, VOICE_EMBEDDINGS_NAME))
        return super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Stored in `voice_embeddings.safetensors`, not in the processor config
        output.pop("voice_embeddings", None)
        return output

    @property
    def available_voices(self) -> list[str]:
        return sorted(self.voice_embeddings)

    @property
    def sampling_rate(self) -> int:
        return self.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config.sampling_rate

    def _build_prompt(self, text: str, num_audio_tokens: int) -> list[int]:
        """`[BOS] [BEGIN_AUDIO] [AUDIO] * n [TEXT_TO_AUDIO] text [AUDIO_TO_TEXT] [BEGIN_AUDIO]`, as `mistral_common`."""
        instruct_tokenizer = self.tokenizer.tokenizer.instruct_tokenizer
        audio_encoder = instruct_tokenizer.audio_encoder
        special_ids = audio_encoder.special_ids
        return (
            instruct_tokenizer.start()
            + [special_ids.begin_audio]
            + [special_ids.audio] * num_audio_tokens
            + [audio_encoder.text_to_audio_token]
            + instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
            + [audio_encoder.audio_to_text_token, audio_encoder.begin_audio_token]
        )

    @auto_docstring
    def __call__(
        self,
        text: str | list[str],
        voice: str | list[str] | None = None,
        voice_embeddings: "torch.Tensor | list[torch.Tensor] | None" = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        r"""
        text (`str` or `list[str]`):
            The text(s) to synthesize.
        voice (`str` or `list[str]`, *optional*):
            Name(s) of the preset voice(s) to use, see `processor.available_voices`. A single voice is used for every
            text.
        voice_embeddings (`torch.Tensor` or `list[torch.Tensor]`, *optional*):
            Custom voice embeddings of shape `(num_audio_tokens, hidden_size)`, used instead of `voice`.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- Prompt token ids (left-padded).
            - **attention_mask** -- Attention mask of the prompts.
            - **voice_embeddings** -- Voice embeddings of all prompts, concatenated along the first dimension, which
              replace the `[AUDIO]` placeholder tokens.
        """
        texts = [text] if isinstance(text, str) else list(text)
        if (voice is None) == (voice_embeddings is None):
            raise ValueError("Pass exactly one of `voice` (a preset name) or `voice_embeddings`.")

        if voice is not None:
            voices = [voice] * len(texts) if isinstance(voice, str) else list(voice)
            unknown = [name for name in voices if name not in self.voice_embeddings]
            if unknown:
                raise ValueError(f"Unknown voice(s) {unknown}. Available voices: {self.available_voices}.")
            embeddings = [self.voice_embeddings[name] for name in voices]
        else:
            embeddings = (
                [voice_embeddings] * len(texts) if torch.is_tensor(voice_embeddings) else list(voice_embeddings)
            )
        if len(embeddings) != len(texts):
            raise ValueError(f"Got {len(embeddings)} voices for {len(texts)} texts.")

        output_kwargs = self._merge_kwargs(**kwargs)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None) or "pt"
        input_ids = [self._build_prompt(t, emb.shape[0]) for t, emb in zip(texts, embeddings)]
        encoding = self.tokenizer(input_ids, **output_kwargs["text_kwargs"], return_tensors=return_tensors)
        encoding["voice_embeddings"] = torch.cat(embeddings, dim=0)
        return BatchFeature(data=dict(encoding), tensor_type=return_tensors)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "voice_embeddings"]


__all__ = ["VoxtralTtsProcessor"]
