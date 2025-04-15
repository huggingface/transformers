# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import is_torch_available

if is_torch_available():
    import torch
    import torch.nn.functional as F

from ..auto import AutoModel, AutoFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack

from ...audio_utils import make_nested_list_of_audios


class CsmProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {},
        "common_kwargs": {"return_tensors": "pt"},
    }


# TODO: @eustlb, AudioTokenizer, to be moved into audio_tokenization_conversational_speech_model.py
class CsmAudioTokenizer:
    def __init__(self):
        self.num_codebooks = 32
        self.codebook_padding_idx = 2050
        self.codebook_eos_idx = 0
        self.codec_model = AutoModel.from_pretrained("kyutai/mimi").cuda()
        self.codec_feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    def __call__(self, audio, **kwargs):
        codec_inputs = self.codec_feature_extractor(
            raw_audio=audio,
            sampling_rate=self.codec_feature_extractor.sampling_rate,
            return_tensors="pt"
        )
        codec_inputs = codec_inputs.to("cuda")
        codec_outputs = self.codec_model.encode(
            codec_inputs["input_values"],
            codec_inputs["padding_mask"]
        )

        return codec_outputs.audio_codes.transpose(1, -1).cpu() 


class CsmProcessor(ProcessorMixin):
    # attributes = ["audio_tokenizer", "tokenizer"]
    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]
    # audio_tokenizer_class = "CsmAudioTokenizer"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        # audio_tokenizer,
        tokenizer,
        chat_template=None,
    ):
        if not hasattr(tokenizer, "audio_token"):
            self.audio_token = "<|AUDIO|>"
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        else:
            self.audio_token = tokenizer.audio_token
            self.audio_token_id = tokenizer.audio_token_id

        if not hasattr(tokenizer, "audio_eos_token"):
            self.audio_eos_token = "<|audio_eos|>"
            self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(self.audio_eos_token)
        else:
            self.audio_eos_token = tokenizer.audio_eos_token
            self.audio_eos_token_id = tokenizer.audio_eos_token_id

        # TODO: @eustlb, AudioTokenizer
        self.audio_tokenizer = CsmAudioTokenizer()
        # self.num_codebooks = 32
        # self.codebook_padding_idx = 2050
        # self.codec_model = AutoModel.from_pretrained("kyutai/mimi").cuda()
        # self.codec_feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        
        # super().__init__(audio_tokenizer, tokenizer, chat_template=chat_template)
        super().__init__(tokenizer, chat_template=chat_template)
    
    def __call__(
        self,
        text,
        audio=None,
        **kwargs: Unpack[CsmProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            CsmProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        return_tensors = common_kwargs.pop("return_tensors", None) 
        if return_tensors != "pt":
            raise ValueError(
                f"{self.__class__.__name__} only supports `return_tensors='pt'`."
            )

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        n_audio_in_text = [t.count(self.audio_token) for t in text]

        n_audio_in_audio = 0
        if audio is not None:
            audio = make_nested_list_of_audios(audio)
            n_audio_in_audio = [len(audio_batch) for audio_batch in audio]

        if sum(n_audio_in_text) > 0 and n_audio_in_audio != n_audio_in_text:
            if audio is None:
                raise ValueError("No audio were provided, but there are audio tokens in the prompt")
            else:
                raise ValueError(
                    f"The number of audio tokens in each text ({n_audio_in_text}) should be the same as the "
                    f"number of provided audios ({n_audio_in_audio})."
                )

        if audio is not None:
            num_audio_tokens_list = [
                self.audio_tokenizer.codec_model.get_encoded_length(audio_array.shape[0])
                for audio_batch in audio for audio_array in audio_batch
            ]
            num_audio_tokens_list_copy = num_audio_tokens_list.copy()

            # expand the text to repeat the audio token for the corresponding number of frames
            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    num_audio_tokens = num_audio_tokens_list_copy.pop(0)
                    expanded_audio_token = self.audio_token * num_audio_tokens

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)
                    
                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)

            text = expanded_text

        data = {}
        encoding = self.tokenizer(text, **text_kwargs)
        data.update(encoding)

        # (BS, seq_len) -> (BS, seq_len, num_codebooks + 1)
        input_ids = F.pad(encoding["input_ids"].unsqueeze(-1), (32, 0), value=self.audio_tokenizer.codebook_padding_idx)
   
        if audio is not None:
            # =======================================
            # TODO: @eustlb, this should be batched !!! 
            # but requires making sure batched inference of the codec model works as intended
            audio_tokens_list = []
            for audio_batch in audio:
                for audio_array in audio_batch:
                    audio_tokens_list.append(self.audio_tokenizer(audio_array, **audio_kwargs)[0])
                
            max_audio_frames = max(el.shape[0] for el in audio_tokens_list)
            batched_audio_token_ids = torch.stack(
                [F.pad(el, (0, 0, 0, max_audio_frames - el.shape[0])) for el in audio_tokens_list]
            )
            # =======================================

            # batched_audio_codes is a tensor of shape (batch_size, max_audio_frames, num_codebooks)
            # We need to:
            # 1. Select only the valid frames for each audio (excluding padding frames)
            # 2. Place these audio tokens at the correct positions in input_ids where audio tokens are located
            num_audio_tokens = torch.tensor(num_audio_tokens_list, dtype=torch.long, device=batched_audio_token_ids.device)
            frames_mask = torch.arange(max_audio_frames, device=batched_audio_token_ids.device).expand(len(num_audio_tokens), -1) < num_audio_tokens.unsqueeze(-1)
            frames_mask = frames_mask.flatten()

            audio_token_idxs = (input_ids[:, :, -1] == self.audio_token_id).nonzero(as_tuple=True)
            input_ids[audio_token_idxs[0], audio_token_idxs[1], :self.audio_tokenizer.num_codebooks] = batched_audio_token_ids.view(-1, self.audio_tokenizer.num_codebooks)[frames_mask]
            data["input_ids"] = input_ids
        
        # audio_eos -> [codebook_eos_idx, codebook_eos_idx, ..., audio_token_id]
        audio_eos_idxs = (encoding["input_ids"] == self.audio_eos_token_id).nonzero(as_tuple=True)
        input_ids[audio_eos_idxs[0], audio_eos_idxs[1], :self.audio_tokenizer.num_codebooks] = self.audio_tokenizer.codebook_eos_idx
        input_ids[audio_eos_idxs[0], audio_eos_idxs[1], -1] = self.audio_token_id

        return BatchFeature(data=data, tensor_type=return_tensors)
    
    def decode(self, audio_token_ids, **kwargs):
        # token_ids: (BS, seq_len, num_codebooks + 1)
        # let's keep only the audio tokens
        audio_token_ids = audio_token_ids[:, :, :self.audio_tokenizer.num_codebooks]
        return self.audio_tokenizer.decode(audio_token_ids, **kwargs)


__all__ = ["CsmProcessor", "CsmAudioTokenizer"]