import math
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from librosa import resample as librosa_resample
from pydub import AudioSegment
from ... import AutoTokenizer, WhisperFeatureExtractor
from ...processing_utils import ProcessorMixin

from .configuration_audioflamingo3 import MEDIA_TOKENS

__all__ = ["AudioFlamingo3Processor"]

class AudioFlamingo3Processor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"


    def __init__(self, feature_extractor=None, tokenizer=None):
        super().__init__(feature_extractor, tokenizer)


    @classmethod
    def from_pretrained(cls, path: str):
        fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')

        tok = AutoTokenizer.from_pretrained(
            os.path.join(path, "llm"),
            padding_side="right",
            use_fast=True,
            legacy=False,
        )

        # any post‑init tweaks, e.g. adding MEDIA_TOKENS
        tok.media_tokens = MEDIA_TOKENS
        for name, token in MEDIA_TOKENS.items():
            tok.add_tokens([token], special_tokens=True)

        # 4) return the processor
        return cls(fe, tok)


    def _tokenize_conversation(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = False,
        overrides: Optional[Dict[str, str]] = None,
        no_system_prompt: bool = False,
    ) -> torch.Tensor:
        # Normalize the conversation before tokenization
        for message in messages:
            message["value"] = message["value"].strip()

        conversation = []
        for m in messages:
            message = {}
            if m["from"] == "human":
                message["role"] = "user"
            elif m["from"] == "gpt":
                message["role"] = "assistant"
            else:
                raise ValueError(f"Unexpected sender '{m['from']}' in conversation entry.")

            message["content"] = m["value"]
            if overrides is not None and m["from"] in overrides:
                message["content"] = overrides[m["from"]]
            conversation.append(message)

        if no_system_prompt:
            conversation = [{"role": "system", "content": ""}] + conversation

        text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

        return self.tokenizer(text, return_tensors="pt").input_ids[0]


    def _get_num_windows(self, T, sr):
        window_length  = int(30.0 * sr)
        window_overlap = int(0.0 * sr)
        max_num_window = 20
        num_windows = 1
        if T <= window_length:
            num_windows = 1
            full_length = window_length
        elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
            num_windows = max_num_window
            full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
        else:
            num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
            full_length = num_windows * window_length - (num_windows - 1) * window_overlap
        
        return num_windows, full_length


    def _load_audio(self, file_path, target_sr=16000, duration=30.0, start=0.0):
        if file_path.endswith('.mp3'):
            audio = AudioSegment.from_file(file_path)
            if len(audio) > (start + duration) * 1000:
                audio = audio[start * 1000:(start + duration) * 1000]
            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            data = np.array(audio.get_array_of_samples())
            if audio.sample_width == 2:
                data = data.astype(np.float32) / np.iinfo(np.int16).max
            elif audio.sample_width == 4:
                data = data.astype(np.float32) / np.iinfo(np.int32).max
            else:
                raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

        else:
            with sf.SoundFile(file_path) as audio:
                original_sr = audio.samplerate
                channels = audio.channels
                max_frames = int((start + duration) * original_sr)
                audio.seek(int(start * original_sr))
                frames_to_read = min(max_frames, len(audio))
                data = audio.read(frames_to_read)
                if data.max() > 1 or data.min() < -1:
                    data = data / max(abs(data.max()), abs(data.min()))
            
            if original_sr != target_sr:
                if channels == 1:
                    data = librosa_resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
                else:
                    data = librosa_resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
            else:
                if channels != 1:
                    data = data.T[0]
        
        if data.min() >= 0:
            data = 2 * data / abs(data.max()) - 1.0
        else:
            data = data / max(abs(data.max()), abs(data.min()))
        
        assert len(data.shape) == 1, data.shape
        return data


    def _load_sound_mask(self, sound_file, sample_rate=16000, window_length=30.0, window_overlap=0.0, max_num_window=20, audio_start = 0.0):
        if sound_file is None:
            return None
        window_length  = int(window_length * sample_rate)
        window_overlap = int(window_overlap * sample_rate)
        max_num_window = int(max_num_window)
        duration = max_num_window * (window_length - window_overlap) + window_overlap

        sound_outputs = []
        audio_feature_masks = []
        audio_embed_masks = []

        try:
            audio_data = self._load_audio(sound_file, sample_rate, duration, audio_start) # already cuts to max duration
            T = len(audio_data)
            audio_data = audio_data.reshape(1, -1)
            num_windows, full_length = self._get_num_windows(T, sample_rate)

            int16_to_float32 = lambda x: (x / 32767.0).astype(np.float32)
            float32_to_int16 = lambda x: (np.clip(x, -1., 1.) * 32767.).astype(np.int16)

            audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
            for i in range(num_windows):
                audio_embed_mask = torch.zeros(750)
                start = i * (window_length - window_overlap)
                audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
                orig_length = audio_data_tensor_this.shape[1]
                audio_data_tensor_this = self.feature_extractor(audio_data_tensor_this.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt")
                sound_outputs.append(audio_data_tensor_this["input_features"])
                # calculate the mask for the input melspec to Whisper
                melspec_frames_this_window = int(math.ceil(orig_length / 160))
                feature_attention_mask = torch.zeros(3000, dtype=torch.int32)
                feature_attention_mask[:melspec_frames_this_window] = 1
                audio_feature_masks.append(feature_attention_mask.unsqueeze(0))
                # calculate the mask for the output embedding for use in AF3
                conv_lengths = (melspec_frames_this_window - 1) // 2 + 1
                output_embedding_lengths = (conv_lengths - 2) // 2 + 1
                audio_embed_mask[:output_embedding_lengths] = 1
                audio_embed_masks.append(audio_embed_mask)
        except:
            print("Error loading sound file: ", sound_file)
            sound_outputs.append(torch.zeros(1,128,3000))
            audio_feature_masks.append(torch.zeros(1, 3000, dtype=torch.int32))
            audio_embed_masks.append(torch.zeros(750))
        sound_outputs = torch.stack(sound_outputs, dim=0)
        audio_feature_masks = torch.stack(audio_feature_masks, dim=0)
        audio_embed_masks = torch.stack(audio_embed_masks, dim=0)
        return sound_outputs.numpy().tolist(), audio_feature_masks ,audio_embed_masks


    def __call__(self, text: str, audio_path: str):
        media = []
        media_meta = defaultdict(list)

        final_text = ""
        sound, audio_feature_masks,audio_embed_masks = self._load_sound_mask(audio_path)
        media.append(sound)
        media_meta["sound_feature_masks"].append(audio_feature_masks)
        media_meta["sound_embed_masks"].append(audio_embed_masks)
        final_text += MEDIA_TOKENS["sound"] * len(sound)
        final_text += text.replace(MEDIA_TOKENS["sound"], "").strip()
        
        conversation = [{"from": "human", "value": final_text}]
        input_ids = self._tokenize_conversation(conversation, add_generation_prompt=True).cuda().unsqueeze(0)

        sounds = torch.tensor(media).half()         
        media = [sound for sound in sounds]
        sound_feature_masks = torch.tensor(media_meta["sound_feature_masks"][0]).half()   
        media_meta["sound_feature_masks"] = [sound_mask for sound_mask in sound_feature_masks]
        sound_embed_masks = torch.tensor(media_meta["sound_embed_masks"][0]).half()   
        media_meta["sound_embed_masks"] = [sound_mask for sound_mask in sound_embed_masks]

        return input_ids, media, media_meta

    def decode(self, token_ids: torch.Tensor) -> str:
        result = [self.tokenizer.decode(output_ids, skip_special_tokens=True).strip() for output_ids in token_ids]
        return result[0] if len(result) == 1 else result
