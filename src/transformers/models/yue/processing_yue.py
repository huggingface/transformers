# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for YuE"""

from ...processing_utils import AudioKwargs, BatchFeature, ProcessingKwargs, ProcessorMixin, Unpack

import re
import numpy as np
import torch
import torchaudio


class YuEAudioKwargs(AudioKwargs, total=False):
    eoa_token_id : int
    soa_token_id: int
    xcodec_marker_token_id: int
    start_of_reference_token_id: int
    end_of_reference_token_id: int
    generation: bool 



class YuEProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: YuEAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "truncation": False,
            "add_special_tokens": False, 
        },
			"audio_kwargs": {
			    "eoa_token_id": 50001,
			    "soa_token_id": 50000,
			    "xcodec_marker_token_id": 50008, # 32016
			    "start_of_reference_token_id": 50006,
			    "end_of_reference_token_id": 50007,
			    "prompt_start_time" : 0.0,
			    "prompt_end_time" :  5.0, #30.0,
                "codebook_size": 1024,
                "num_codebooks": 12,
                "global_offset": 45334,
                "fps": 50,
                "sample_rate":16000,
                }}
    



class YuEProcessor(ProcessorMixin):
    """
    Constructs a YuE processor which wraps a YuE tokenizer and a finetuned XCodec audio tokenizer into a single processor.

    [`YuEProcessor`] offers all the functionalities of [`YuETokenizer`] and [`XCodecModel`]. See the
    [`~YuEProcessor.__call__`] and [`~YuEProcessor.decode`] for more information.

    Args:
        tokenizer ([`YuETokenizer`]):
            The tokenizer is a required input.
        audio_tokenizer ([`XCodecModel`]):
            The audio tokenizer is a required input.
    """
    
    tokenizer_class = "YuETokenizer"
    audio_tokenizer_class = "XCodecModel"
    attributes = ["tokenizer", "audio_tokenizer"]

    def __init__(self, tokenizer, audio_tokenizer):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer

    def __call__(self, text=None, lyrics_segments=None, genre_tags=None, audio=None, return_tensors = None, **kwargs: Unpack[YuEProcessorKwargs],): #return_tensors="pt",
        output_kwargs = self._merge_kwargs(YuEProcessorKwargs, **kwargs)
        audio_kwargs = output_kwargs["audio_kwargs"]

        if lyrics_segments is None and text is None:
            raise ValueError("Either `lyrics_segments` or `text` must be provided.")

        #TODO : I should check that passed text has [chorus] [verse] tokens
        if lyrics_segments is None:
            lyrics_segments = self._split_lyrics_into_segments(text)

        #TODO : same thing check lyrics_segments has [chorus] [verse] tokens
        full_lyrics = "\n".join(lyrics_segments) 

        main_prompt = f"""Generate music from the given lyrics segment by segment.
                        [Genre] {genre_tags}
                        {full_lyrics}"""

        # tokenize main prompt with genre and full lyrics (this is head_ids)
        head_prompt_ids = self.tokenizer(main_prompt, **output_kwargs["text_kwargs"])["input_ids"]

        if audio is not None and self.audio_tokenizer is not None:
            head_prompt_ids= self._process_audio_prompt(head_prompt_ids, audio, audio_kwargs)

        # head_prompt_ids is used only in begenining tokenize each segment individually, they are used in the generation loop inside the stage 1 model
        lyrics_segments_ids = [self.tokenizer(segment, **output_kwargs["text_kwargs"])["input_ids"] for segment in lyrics_segments]

        return BatchFeature({"head_prompt_ids": head_prompt_ids, "lyrics_segments_ids": lyrics_segments_ids}) #,  tensor_type=None)


    @staticmethod
    def _split_lyrics_into_segments(lyrics):
        """Split lyrics into segments based on structure tags like [verse], [chorus], etc"""
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
        return structured_lyrics

    def _process_audio_prompt(self, text_ids, audio, audio_kwargs):
        target_sample_rate = audio_kwargs.pop("sample_rate", None)
        if isinstance(audio, str):
            raw_audio, sample_rate = torchaudio.load(audio)
        else:
            raw_audio, sample_rate = audio, target_sample_rate 
    
        if raw_audio.shape[0] > 1:
            # convert to mono if stereo
            raw_audio = torch.mean(raw_audio, dim=0, keepdim=True)

        if sample_rate != target_sample_rate:
            raw_audio = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(raw_audio)

        input_audio = raw_audio.unsqueeze(0)

        # maybe because xcodec doesn't support batching will loop element
        with torch.no_grad():
            audio_codes = self.audio_tokenizer.encode(input_audio, bandwidth=0.5).audio_codes

        # TODO: handle this better
        eoa_token_id = audio_kwargs.pop("eoa_token_id", None)
        soa_token_id = audio_kwargs.pop("soa_token_id", None)
        xcodec_marker_token_id = audio_kwargs.pop("xcodec_marker_token_id", None)
        prompt_start_time  = audio_kwargs.pop("prompt_start_time", None)
        prompt_end_time  = audio_kwargs.pop("prompt_end_time", None)
 
        # original yue takes only the codes of the first quantizer
        audio_codes_numpy = audio_codes[:, 0, :].cpu().numpy()
        audio_ids = self._offset_and_flatten_tokens(audio_codes_numpy, audio_kwargs)  
        start = int(prompt_start_time *50)
        end = int(prompt_end_time *50)
        audio_ids = audio_ids[start : end]
    
        # formating audio input
        audio_ids = [soa_token_id] + [xcodec_marker_token_id] + audio_ids + [eoa_token_id]
        start_of_reference = self.tokenizer("[start_of_reference]", add_special_tokens=False)["input_ids"]
        end_of_reference   = self.tokenizer("[end_of_reference]",   add_special_tokens=False)["input_ids"]
        audio_ids = start_of_reference + audio_ids + end_of_reference

        prompt_input_ids = text_ids  + audio_ids
        return prompt_input_ids


    def _offset_and_flatten_tokens(self, audio_codes, audio_kwargs):
        if audio_codes.ndim != 2 or audio_codes.shape[0] != 1:
            raise ValueError(f"Audio codes shape should be (1, T), got {audio_codes.shape}")
        
        #TODO handle this as well
        codebook_size = audio_kwargs.pop("codebook_size", None)
        global_offset = audio_kwargs.pop("global_offset", None)

        if audio_codes.max() >= codebook_size:
            raise ValueError(f"max(audio_codes)={audio_codes.max()}, codebook_size={codebook_size}")
        if audio_codes.min() < 0:
            raise ValueError(f"min(audio_codes)={audio_codes.min()}, must be >= 0")

        # apply offset to audio codes then flatten like original yue implementation
        # does  offset = global_offset + k * codebook_size for each quantizer k
        # for one quantizer k=0 so only global_offset is added
        # see https://github.com/multimodal-art-projection/YuE/blob/main/inference/codecmanipulator.py#L90 

        offset_codes = audio_codes.copy().astype(np.uint32)
        offset_codes[0] += global_offset
        flattened_tokens = offset_codes.flatten()

        return flattened_tokens.tolist()