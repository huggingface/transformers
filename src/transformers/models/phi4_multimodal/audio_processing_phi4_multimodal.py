# Copyright 2025 The HuggingFace Inc. team.
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

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import mel_filter_bank
from ...feature_extraction_utils import BatchFeature


class Phi4MultimodalAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    preemphasis = 0.97
    n_fft = 512
    hop_length = 160
    win_length = 400
    n_mels = 80

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=self.n_mels,
            min_frequency=0,
            max_frequency=7690,
            sampling_rate=self.sample_rate,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

    def extract_spectrogram(self, audio, *, spectrogram_config):
        import torch

        waveform = torch.stack(audio, dim=0)
        device = waveform.device
        batch_size = waveform.shape[0]
        lengths = torch.tensor([a.shape[-1] for a in audio], device=device)

        # Unfold into frames
        frames = waveform.unfold(-1, self.win_length, self.hop_length)

        # Frame-level masking for padded inputs
        if batch_size > 1:
            frames = frames.clone()
            to_mask_batch_idxs = torch.arange(batch_size, device=device)[lengths != lengths.max()]
            if to_mask_batch_idxs.numel() > 0:
                batch_idxs_down = (lengths[to_mask_batch_idxs] - self.win_length) // self.hop_length + 1
                batch_idxs_up = (lengths[to_mask_batch_idxs] // self.hop_length) - 1
                offset_idx = batch_idxs_down.min()
                max_idx = batch_idxs_up.max()

                mask = torch.arange(max_idx - offset_idx, device=device).expand(to_mask_batch_idxs.shape[0], -1)
                mask = ((batch_idxs_down - offset_idx).unsqueeze(1) <= mask) & (
                    mask < (batch_idxs_up - offset_idx).unsqueeze(1)
                )
                mask = mask.unsqueeze(-1).expand(-1, -1, self.win_length)
                masked_frames = frames[to_mask_batch_idxs, offset_idx:max_idx].masked_fill_(mask, 0)
                frames[to_mask_batch_idxs, offset_idx:max_idx] = masked_frames

        # Pre-emphasis
        frames_prev = torch.roll(frames, 1, dims=-1)
        frames_prev[:, :, 0] = frames_prev[:, :, 1]
        frames = (frames - self.preemphasis * frames_prev) * 32768

        # Hamming window + FFT
        fft_window = torch.hamming_window(self.win_length, periodic=False, device=device, dtype=torch.float64)
        S = torch.fft.rfft(fft_window * frames.view(-1, self.win_length), n=self.n_fft, dim=1)
        S = S.view(batch_size, -1, S.shape[-1]).to(torch.complex64)

        spec = torch.abs(S)
        spec_power = spec**2

        # Mel filterbank + log
        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        log_spec = torch.clamp(spec_power @ mel_filters, min=1.0)
        log_spec = torch.log(log_spec)

        return [log_spec[i] for i in range(batch_size)]

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        import torch

        # Pad values to longest
        if padding:
            audio = self.pad_values(
                audio, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of
            )

        # Extract spectrogram
        features = self.extract_spectrogram(audio, spectrogram_config=None)

        # Pad features and stack
        max_feat_len = max(f.shape[0] for f in features)
        padded = []
        for f in features:
            if f.shape[0] < max_feat_len:
                pad_amount = max_feat_len - f.shape[0]
                f = torch.nn.functional.pad(f, (0, 0, 0, pad_amount), mode="constant", value=0.0)
            padded.append(f)

        output_key = self.model_input_names[0]
        stacked = torch.stack(padded, dim=0)
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["Phi4MultimodalAudioProcessor"]
