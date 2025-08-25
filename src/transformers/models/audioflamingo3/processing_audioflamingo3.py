import math
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from ... import AutoTokenizer, WhisperFeatureExtractor
from ...processing_utils import ProcessorMixin
from ...utils.hub import snapshot_download
from ...utils import requires_backends

from .configuration_audioflamingo3 import MEDIA_TOKENS

__all__ = ["AudioFlamingo3Processor"]


class AudioFlamingo3Processor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        local_files_only: bool = False,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        # Resolve to local dir (supports HF repo id)
        root = os.path.expanduser(path)
        if not os.path.isdir(root):
            root = snapshot_download(
                repo_id=path,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )

        # Prefer "<root>/llm", fallback to "<root>/model/llm"
        llm_dir = os.path.join(root, "llm")
        if not os.path.isdir(llm_dir):
            alt = os.path.join(root, "model", "llm")
            llm_dir = alt if os.path.isdir(alt) else llm_dir
        if not os.path.isdir(llm_dir):
            raise FileNotFoundError(f"Tokenizer folder 'llm' not found under: {root}")

        # Load components
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        tok = AutoTokenizer.from_pretrained(
            llm_dir,
            padding_side="right",
            use_fast=True,
            legacy=False,
        )

        # Post-init tweaks
        tok.media_tokens = MEDIA_TOKENS
        for _, t in MEDIA_TOKENS.items():
            tok.add_tokens([t], special_tokens=True)

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
        window_length = int(30.0 * sr)
        window_overlap = int(0.0 * sr)
        max_num_window = 20
        num_windows = 1
        if T <= window_length:
            num_windows = 1
            full_length = window_length
        elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
            num_windows = max_num_window
            full_length = max_num_window * window_length - (max_num_window - 1) * window_overlap
        else:
            num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
            full_length = num_windows * window_length - (num_windows - 1) * window_overlap

        return num_windows, full_length

    def _load_audio(self, file_path, target_sr=16000, duration=30.0, start=0.0):
        requires_backends(self, ["librosa"])
        import librosa

        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim == 1: y = y[np.newaxis, :]
        C, N = y.shape

        if file_path.endswith(".mp3"):
            if (N / sr) * 1000.0 > (start + duration) * 1000.0:
                s0, s1 = int(start * sr), int((start + duration) * sr)
                y = y[:, max(0, s0):min(N, s1)]
            y = y.mean(axis=0) if C > 1 else y[0]
            if sr != target_sr and y.size: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            data = y.astype(np.float32, copy=False)
        else:
            s0, s1 = int(start * sr), min(N, int((start + duration) * sr))
            y = y[:, s0:s1] if s0 < N else y[:, 0:0]
            y = y[0] if y.ndim == 2 and y.shape[0] > 1 else y.squeeze(0)
            if sr != target_sr and y.size: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            data = y.astype(np.float64, copy=False)

        if data.size:
            dmin, dmax = data.min(), data.max()
            if dmin >= 0:
                m = abs(dmax) or 1.0
                data = 2 * data / m - 1.0
            else:
                m = max(abs(dmax), abs(dmin)) or 1.0
                data = data / m

        assert len(data.shape) == 1, data.shape
        return data

    def _load_sound_mask(self, sound_file, sample_rate=16000, window_length=30.0, window_overlap=0.0, max_num_window=20, audio_start=0.0):
        if sound_file is None:
            return None
        window_length = int(window_length * sample_rate)
        window_overlap = int(window_overlap * sample_rate)
        max_num_window = int(max_num_window)
        duration = max_num_window * (window_length - window_overlap) + window_overlap

        sound_outputs = []
        audio_feature_masks = []
        audio_embed_masks = []

        try:
            audio_data = self._load_audio(sound_file, sample_rate, duration, audio_start)  # already cuts to max duration
            T = len(audio_data)
            audio_data = audio_data.reshape(1, -1)
            num_windows, full_length = self._get_num_windows(T, sample_rate)

            int16_to_float32 = lambda x: (x / 32767.0).astype(np.float32)
            float32_to_int16 = lambda x: (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

            audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
            for i in range(num_windows):
                audio_embed_mask = torch.zeros(750)
                start = i * (window_length - window_overlap)
                audio_data_tensor_this = audio_data_tensor[:, start : start + window_length]
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
            sound_outputs.append(torch.zeros(1, 128, 3000))
            audio_feature_masks.append(torch.zeros(1, 3000, dtype=torch.int32))
            audio_embed_masks.append(torch.zeros(750))
        sound_outputs = torch.stack(sound_outputs, dim=0)
        audio_feature_masks = torch.stack(audio_feature_masks, dim=0)
        audio_embed_masks = torch.stack(audio_embed_masks, dim=0)
        return sound_outputs.numpy().tolist(), audio_feature_masks, audio_embed_masks

    def __call__(self, text: str, audio_path: str):
        media = []
        media_meta = defaultdict(list)

        final_text = ""
        sound, audio_feature_masks, audio_embed_masks = self._load_sound_mask(audio_path)
        media.append(sound)
        media_meta["sound_feature_masks"].append(audio_feature_masks)
        media_meta["sound_embed_masks"].append(audio_embed_masks)
        final_text += MEDIA_TOKENS["sound"] * len(sound)
        final_text += text.replace(MEDIA_TOKENS["sound"], "").strip()

        conversation = [{"from": "human", "value": final_text}]
        input_ids = self._tokenize_conversation(conversation, add_generation_prompt=True).cuda().unsqueeze(0)

        sounds = torch.tensor(media).half()
        media = [sound for sound in sounds]
        sound_feature_masks = media_meta["sound_feature_masks"][0].detach().clone().half()
        media_meta["sound_feature_masks"] = [sound_mask for sound_mask in sound_feature_masks]
        sound_embed_masks = media_meta["sound_embed_masks"][0].detach().clone().half()
        media_meta["sound_embed_masks"] = [sound_mask for sound_mask in sound_embed_masks]

        return input_ids, media, media_meta

    def decode(self, token_ids: torch.Tensor) -> str:
        result = [self.tokenizer.decode(output_ids, skip_special_tokens=True).strip() for output_ids in token_ids]
        return result[0] if len(result) == 1 else result
