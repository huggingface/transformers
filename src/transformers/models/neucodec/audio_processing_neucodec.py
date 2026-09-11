# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch

from ..xcodec2.audio_processing_xcodec2 import Xcodec2AudioProcessor, Xcodec2AudioProcessorMixin


class NeuCodecAudioProcessorMixin(Xcodec2AudioProcessorMixin):
    """Same dual-encoder geometry as XCodec2: raw padded audio for the acoustic encoder, kaldi
    povey fbank features for the semantic one."""


class NeuCodecAudioProcessor(NeuCodecAudioProcessorMixin, Xcodec2AudioProcessor):
    def _finalize_output(self, output, audio_ranges=None, **kwargs):
        # XCodec2 trims each waveform to a whole number of hops *and* pads it by half a hop on
        # both sides before the fbank. NeuCodec's reference does neither: it pads the waveform up
        # to a hop multiple and feeds that straight in.
        # https://github.com/neuphonic/neucodec/blob/ed3e6cd/neucodec/model.py#L128
        #
        # The padding is per-utterance rather than to the batch's longest, because the CMVN below
        # runs over the whole padded signal — collating first would let a long clip skew a short
        # one's statistics.
        audio_values = output["audio_values"]
        padded_length = audio_values.shape[-1]

        features = []
        for i, (start, end) in enumerate(audio_ranges):
            # `end - start` counts the zero sample `_downmix_to_mono` appends, which the legacy
            # extractor also pads before rounding up to a hop multiple.
            valid_length = min(-(-(end - start) // self.hop_length) * self.hop_length, padded_length)
            waveform = audio_values[i, 0, :valid_length]
            f = self.compute_features([waveform], spectrogram_config=self.spectrogram_config)[0].transpose(-2, -1)
            f = (f - f.mean(0)) / torch.sqrt(f.var(0, unbiased=True) + 1e-7)
            features.append(f)

        frame_lengths = [f.shape[0] for f in features]
        max_frames = max(frame_lengths)
        # The legacy extractor pads to a stride multiple and then trims back to one, so the frame
        # count is always rounded *down* to the stride.
        if max_frames % self.stride:
            max_frames += self.stride - max_frames % self.stride
        batch = torch.stack(
            [
                torch.nn.functional.pad(f, (0, 0, 0, max_frames - f.shape[0]), value=self.feature_padding_value)
                for f in features
            ]
        )
        mask = self._get_mask([(0, length) for length in frame_lengths], max_frames)

        batch_size, num_frames, num_mel_bins = batch.shape
        output["audio_features"] = batch.reshape(batch_size, num_frames // self.stride, num_mel_bins * self.stride)
        output["audio_features_mask"] = (
            mask.reshape(batch_size, num_frames // self.stride, self.stride).min(dim=-1).values
        )
        return output


__all__ = ["NeuCodecAudioProcessor"]
