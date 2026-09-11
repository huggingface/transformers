# Copyright 2026 The HuggingFace Inc. team.
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

from ..xcodec2.audio_processing_xcodec2 import Xcodec2AudioProcessor, Xcodec2AudioProcessorMixin


class NeuCodecAudioProcessorMixin(Xcodec2AudioProcessorMixin):
    """Same dual-encoder geometry as XCodec2: raw padded audio for the acoustic encoder, kaldi
    povey fbank features for the semantic one."""

    def _preprocess(self, audio, *args, **kwargs):
        # The legacy semantic branch extracts each fbank from the original clip, independently
        # of any `max_length` truncation applied to the acoustic branch.
        return super()._preprocess(audio, *args, semantic_waveforms=audio, **kwargs)

    def _pad_semantic_waveform(self, waveform):
        # NeuCodec's reference feeds the hop-rounded clip straight to the fbank, without XCodec2's
        # half-hop context: https://github.com/neuphonic/neucodec/blob/ed3e6cd/neucodec/model.py#L128
        return waveform


class NeuCodecAudioProcessor(NeuCodecAudioProcessorMixin, Xcodec2AudioProcessor):
    pass


__all__ = ["NeuCodecAudioProcessor"]
