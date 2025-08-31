import math
from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from ...processing_utils import ProcessorMixin

__all__ = ["AudioFlamingo3Processor"]


class AudioFlamingo3Processor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

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

    def _load_sound_mask(self, audio_data, sample_rate=16000, window_length=30.0, window_overlap=0.0, max_num_window=20, audio_start=0.0):
        if audio_data is None:
            return None
        window_length = int(window_length * sample_rate)
        window_overlap = int(window_overlap * sample_rate)
        max_num_window = int(max_num_window)
        duration = max_num_window * (window_length - window_overlap) + window_overlap

        sound_outputs = []
        audio_feature_masks = []
        audio_embed_masks = []

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

        sound_outputs = torch.stack(sound_outputs, dim=0)
        audio_feature_masks = torch.stack(audio_feature_masks, dim=0)
        audio_embed_masks = torch.stack(audio_embed_masks, dim=0)
        return sound_outputs.numpy().tolist(), audio_feature_masks, audio_embed_masks

    def __call__(self, text: str, audio_data):
        media = []
        media_meta = defaultdict(list)

        final_text = ""
        sound, audio_feature_masks, audio_embed_masks = self._load_sound_mask(audio_data)
        media.append(sound)
        media_meta["sound_feature_masks"].append(audio_feature_masks)
        media_meta["sound_embed_masks"].append(audio_embed_masks)
        final_text += "<sound>" * len(sound)
        final_text += text.replace("<sound>", "").strip()

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
