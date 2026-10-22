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

import copy
from dataclasses import dataclass

import torch

from ...generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin
from ...utils import logging


logger = logging.get_logger(__name__)


@dataclass
class Qwen3TTSGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Output of [`Qwen3TTSForConditionalGeneration.generate`].

    Args:
        sequences (`list[torch.LongTensor]`):
            The generated audio codes, one `(codes_length, num_quantizers)` tensor per sample. Decode them to
            waveforms with [`~Qwen3TTSProcessor.decode`].
    """


class Qwen3TTSGenerationMixin(GenerationMixin):
    """
    Generation mixin for Qwen3TTSForConditionalGeneration.

    Contains the TTS-specific generation logic including speaker prompt generation,
    in-context learning prompt generation, and the main generate method.
    """

    def _prepare_generation_config(self, generation_config: GenerationConfig | None, **kwargs):
        """Resolve Qwen sampling defaults without overriding saved or caller-supplied settings."""
        subtalker_names = ("subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature")
        subtalker_kwargs = {name: kwargs.pop(name) for name in subtalker_names if name in kwargs}
        generation_config = copy.deepcopy(generation_config) if generation_config is not None else GenerationConfig()
        generation_config.update(**self.generation_config.to_dict(), defaults_only=True, allow_custom_entries=True)
        generation_config.update(
            defaults_only=True,
            max_new_tokens=4096 if generation_config.max_length is None and kwargs.get("max_length") is None else None,
            min_new_tokens=2 if not generation_config.min_length and kwargs.get("min_length") is None else None,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            eos_token_id=self.config.talker_config.codec_eos_token_id,
            suppress_tokens=[
                token_id
                for token_id in range(
                    self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size
                )
                if token_id != self.config.talker_config.codec_eos_token_id
            ],
            return_dict_in_generate=True,
        )
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        for name, default in zip(subtalker_names, (generation_config.do_sample, 50, 1.0, 0.9)):
            value = subtalker_kwargs.get(name, getattr(generation_config, name, None))
            value = default if value is None else value
            setattr(generation_config, name, value)
        if generation_config.use_cache is False:
            raise ValueError(
                "Qwen3-TTS generation requires `use_cache=True` to retain previous audio frame embeddings."
            )
        if generation_config.num_beams != 1 or generation_config.num_return_sequences != 1:
            raise ValueError("Qwen3-TTS supports only one generated sequence per input and `num_beams=1`.")
        return generation_config, model_kwargs

    def _sample(self, input_ids, logits_processor, stopping_criteria, generation_config, **model_kwargs):
        # Pipeline initialization also prepares generation configs; only sampling needs these step-level kwargs.
        for name in ("subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature"):
            model_kwargs[name] = getattr(generation_config, name)
        return super()._sample(
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            **model_kwargs,
        )

    @torch.no_grad()
    def _extract_speaker_embeddings(self, input_features, feature_attention_mask=None):
        """Encode each unpadded reference independently: pooling must never see padded mel frames."""
        if self.speaker_encoder is None:
            raise ValueError("`input_features` requires a checkpoint with a speaker encoder.")
        if input_features.ndim != 3 or input_features.shape[-1] != self.config.speaker_encoder_config.mel_dim:
            raise ValueError("`input_features` must have shape (batch_size, feature_length, mel_dim).")
        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                input_features.shape[:2], dtype=torch.bool, device=input_features.device
            )
        elif feature_attention_mask.shape != input_features.shape[:2]:
            raise ValueError("`feature_attention_mask` must match the first two dimensions of `input_features`.")
        if not feature_attention_mask.bool().any(dim=-1).all():
            raise ValueError("Each reference must contain at least one valid mel frame.")
        encoder_weight = next(self.speaker_encoder.parameters())
        input_features = input_features.to(device=encoder_weight.device, dtype=encoder_weight.dtype)
        feature_attention_mask = feature_attention_mask.to(device=encoder_weight.device, dtype=torch.bool)
        return torch.cat(
            [
                self.speaker_encoder(features[mask].unsqueeze(0))
                for features, mask in zip(input_features, feature_attention_mask)
            ],
            dim=0,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        past_hidden=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        codec_frames=None,
        **kwargs,
    ):
        if kwargs.get("use_cache", True) is False:
            raise ValueError(
                "Qwen3-TTS generation requires `use_cache=True` to retain previous audio frame embeddings."
            )
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if not is_first_iteration:
            primary_codes = input_ids[:, -1:]
            primary_embed = self.get_input_embeddings()(primary_codes)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, primary_embed), dim=1),
                max_new_tokens=self.config.talker_config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((primary_codes, predictor_result.sequences), dim=-1)
            if codec_frames is not None:
                codec_frames.append(codec_ids)
            codec_embeds = torch.cat(
                [primary_embed]
                + [
                    embedding(predictor_result.sequences[:, index : index + 1])
                    for index, embedding in enumerate(self.code_predictor.get_input_embeddings())
                ],
                dim=1,
            ).sum(1, keepdim=True)
            if generation_step < trailing_text_hidden.shape[1]:
                codec_embeds = codec_embeds + trailing_text_hidden[:, generation_step : generation_step + 1]
            else:
                codec_embeds = codec_embeds + tts_pad_embed
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = codec_embeds
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["past_hidden"] = outputs.past_hidden
        generation_step = model_kwargs.get("generation_step")
        model_kwargs["generation_step"] = 0 if generation_step is None else generation_step + 1
        return model_kwargs

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr, feature_extractor=None):
        if sr != 24000:
            raise ValueError(f"Only 24kHz audio is supported, got {sr}Hz")
        if feature_extractor is None:
            from .feature_extraction_qwen3_tts import Qwen3TTSFeatureExtractor

            feature_extractor = Qwen3TTSFeatureExtractor()
        features = feature_extractor(audio, return_tensors="pt", sampling_rate=sr)
        return self._extract_speaker_embeddings(features["input_features"])[0]

    @torch.inference_mode()
    def generate_speaker_prompt(self, voice_clone_prompt: list[dict]):
        voice_clone_spk_embeds = []
        for index in range(len(voice_clone_prompt["ref_spk_embedding"])):
            ref_spk_embedding = voice_clone_prompt["ref_spk_embedding"][index].to(self.device).to(self.dtype)
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
        text_embed = self.text_projection(self.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        # codec embed (codec bos + codec) 1 T2 D
        codec_embed = []
        for i in range(self.config.talker_config.num_code_groups):
            if i == 0:
                codec_embed.append(self.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(self.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i : i + 1]))
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat(
            [
                self.get_input_embeddings()(
                    torch.tensor(
                        [[self.config.talker_config.codec_bos_id]],
                        device=self.device,
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
            icl_input_embed = text_embed + self.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id] * text_lens],
                    device=self.device,
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
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        subtalker_dosample: bool | None = None,
        subtalker_top_k: int | None = None,
        subtalker_top_p: float | None = None,
        subtalker_temperature: float | None = None,
        eos_token_id: int | None = None,
        repetition_penalty: float | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ):
        """Generate codec frames from formatted text, optionally conditioned on unpadded reference mel features."""
        if kwargs.get("speaker_embeddings") is not None:
            raise ValueError("For generation, pass cached speaker embeddings through `voice_clone_prompt`.")
        kwargs.update(
            {
                name: value
                for name, value in {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "subtalker_dosample": subtalker_dosample,
                    "subtalker_top_k": subtalker_top_k,
                    "subtalker_top_p": subtalker_top_p,
                    "subtalker_temperature": subtalker_temperature,
                    "eos_token_id": eos_token_id,
                    "repetition_penalty": repetition_penalty,
                }.items()
                if value is not None
            }
        )
        if input_features is not None:
            if voice_clone_prompt is not None or any(speakers or []):
                raise ValueError("`input_features` cannot be combined with `voice_clone_prompt` or named `speakers`.")
            if input_features.shape[0] != len(input_ids):
                raise ValueError("`input_features` and `input_ids` must have the same batch size.")
            speaker_embeddings = self._extract_speaker_embeddings(input_features, feature_attention_mask)
            voice_clone_prompt = {
                "ref_spk_embedding": list(speaker_embeddings),
                "x_vector_only_mode": [True] * len(input_ids),
                "icl_mode": [False] * len(input_ids),
                "ref_code": None,
            }
        elif feature_attention_mask is not None:
            raise ValueError("`feature_attention_mask` requires `input_features`.")

        input_ids = [ids.to(self.device) for ids in input_ids]
        if instruct_ids is not None:
            instruct_ids = [ids.to(self.device) if ids is not None else None for ids in instruct_ids]

        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        # voice clone speaker prompt generate
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

        # instruct text prompt generate
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(self.text_projection(self.get_text_embeddings()(instruct_id)))

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
                        speaker_embed = self.get_input_embeddings()(
                            torch.tensor(spk_id, device=self.device, dtype=input_id.dtype)
                        )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            if language is None:
                raise ValueError("Each input needs a language; pass `languages=['Auto']` to detect it from the text.")

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

            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.text_projection(
                self.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=self.device,
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

            codec_input_emebdding_0 = self.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=self.device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = self.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id, self.config.talker_config.codec_bos_id]],
                    device=self.device,
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
            _talker_input_embed_role = self.text_projection(self.get_text_embeddings()(input_id[:, :3]))

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
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.device),
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
                        self.text_projection(self.get_text_embeddings()(input_id[:, 3:4]))
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
                                    self.text_projection(self.get_text_embeddings()(input_id[:, 3:-5])),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + self.get_input_embeddings()(
                                torch.tensor(
                                    [[self.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                    device=self.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + self.get_input_embeddings()(
                                torch.tensor(
                                    [[self.config.talker_config.codec_bos_id]],
                                    device=self.device,
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
                            self.text_projection(self.get_text_embeddings()(input_id[:, 4:-5])),
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

        codec_frames = []
        talker_result = super().generate(
            generation_config=generation_config,
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            codec_frames=codec_frames,
            **kwargs,
        )
        output_kwargs = (
            {name: value for name, value in talker_result.items() if name != "sequences"}
            if not isinstance(talker_result, torch.Tensor)
            else {}
        )

        if not codec_frames:
            return Qwen3TTSGenerateOutput(
                sequences=[
                    torch.empty((0, self.config.talker_config.num_code_groups), dtype=torch.long, device=self.device)
                    for _ in range(batch_size)
                ],
                **output_kwargs,
            )
        talker_codes = torch.stack(codec_frames, dim=1)

        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == self.config.talker_config.codec_eos_token_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])

        talker_codes_list = [talker_codes[i, :length] for i, length in enumerate(effective_lengths)]

        return Qwen3TTSGenerateOutput(sequences=talker_codes_list, **output_kwargs)
