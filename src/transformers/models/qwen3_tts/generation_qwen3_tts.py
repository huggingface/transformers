# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Generation mixin for Qwen3-TTS."""

import torch

from ...generation import GenerationMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class Qwen3TTSGenerationMixin(GenerationMixin):
    """
    Generation mixin for Qwen3TTSForConditionalGeneration.

    Contains the TTS-specific generation logic including speaker prompt generation,
    in-context learning prompt generation, and the main generate method.
    """

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr, feature_extractor=None):
        if sr != 24000:
            raise ValueError(f"Only 24kHz audio is supported, got {sr}Hz")
        if feature_extractor is None:
            from .feature_extraction_qwen3_tts import Qwen3TTSFeatureExtractor

            feature_extractor = Qwen3TTSFeatureExtractor()
        features = feature_extractor(audio, return_tensors="pt", sampling_rate=sr)
        mels = features["input_features"].to(self.device).to(self.dtype)
        speaker_embedding = self.speaker_encoder(mels)[0]
        return speaker_embedding

    @torch.inference_mode()
    def generate_speaker_prompt(self, voice_clone_prompt: list[dict]):
        voice_clone_spk_embeds = []
        for index in range(len(voice_clone_prompt["ref_spk_embedding"])):
            ref_spk_embedding = (
                voice_clone_prompt["ref_spk_embedding"][index].to(self.talker.device).to(self.talker.dtype)
            )
            voice_clone_spk_embeds.append(ref_spk_embedding)
        return voice_clone_spk_embeds

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        # text embed (ref id + text id + eos) 1 T1 D
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        # codec embed (codec bos + codec) 1 T2 D
        codec_embed = []
        for i in range(self.talker.config.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(self.talker.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i : i + 1]))
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat(
            [
                self.talker.get_input_embeddings()(
                    torch.tensor(
                        [[self.config.talker_config.codec_bos_id]],
                        device=self.talker.device,
                        dtype=text_id.dtype,
                    )
                ),
                codec_embed,
            ],
            dim=1,
        )
        # compute lens
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id] * text_lens],
                    device=self.talker.device,
                    dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        else:
            if text_lens > codec_lens:
                return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
            else:
                text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
                return text_embed + codec_embed, tts_pad_embed

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[torch.Tensor] | None = None,
        instruct_ids: list[torch.Tensor] | None = None,
        ref_ids: list[torch.Tensor] | None = None,
        voice_clone_prompt: list[dict] | None = None,
        languages: list[str] | None = None,
        speakers: list[str] | None = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id if eos_token_id is not None else self.config.talker_config.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i != self.config.talker_config.codec_eos_token_id
            ],
            "output_hidden_states": getattr(kwargs, "output_hidden_states", True),
            "return_dict_in_generate": getattr(kwargs, "return_dict_in_generate", True),
        }

        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        # voice clone speaker prompt generate
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

        # instruct text prompt generate
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
                    )

        # tts text prompt generate
        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None] * len(input_ids)
        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:  # Instruct create speaker
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    else:
                        spk_id = self.config.talker_config.spk_id[speaker.lower()]
                        speaker_embed = self.talker.get_input_embeddings()(
                            torch.tensor(spk_id, device=self.talker.device, dtype=input_id.dtype)
                        )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None

            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in self.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self.config.talker_config.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker != ""
                and speaker is not None
                and self.config.talker_config.spk_is_dialect[speaker.lower()] is not False
            ):
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=self.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)  # 3 * [1 1 d]

            # codec: tag and speaker
            if language_id is None:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]
            else:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_think_id,
                        self.config.talker_config.codec_think_bos_id,
                        language_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]

            codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=self.talker.device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id, self.config.talker_config.codec_bos_id]],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat(
                    [codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1
                )

            # <|im_start|>assistant\n
            _talker_input_embed_role = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, :3]))

            # tts_pad * N + tts_bos
            _talker_input_embed = (
                torch.cat(
                    (tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed),
                    dim=1,
                )
                + codec_input_emebdding[:, :-1]
            )

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt["ref_code"] is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                # tts_text_first_token
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4]))
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:-5])),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[self.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[self.config.talker_config.codec_bos_id]],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (
                            self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 4:-5])),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        # for batch inference
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        # left padding for talker input embeds
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(sequences_reversed, batch_first=True, padding_value=0.0)
        talker_input_embeds = padded_reversed.flip(dims=[1])
        # generate mask
        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)
        # padding trailing text hiddens
        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(sequences_to_pad, batch_first=True, padding_value=0.0)
        arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
            len(trailing_text_original_lengths), -1
        )
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        # forward
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )

        talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
        talker_hidden_states = torch.cat([hid[0][-1][:, -1:] for hid in talker_result.hidden_states], dim=1)[:, :-1]

        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == self.config.talker_config.codec_eos_token_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])

        talker_codes_list = [talker_codes[i, :length] for i, length in enumerate(effective_lengths)]
        talker_hidden_states_list = [talker_hidden_states[i, :length, :] for i, length in enumerate(effective_lengths)]

        return talker_codes_list, talker_hidden_states_list
