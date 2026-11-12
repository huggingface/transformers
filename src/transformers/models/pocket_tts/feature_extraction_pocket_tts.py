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

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, logging
from ...utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


@requires(backends=("torch",))
class PocketTTSFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Pocket TTS feature extractor, which prepares voice-prompt waveforms for the audio tokenizer's encoder.

    Each waveform optionally ends on a short pause (the original's `end_on_pause`: training prompts end inside the pause
    between two words, a prompt that stops on speech makes the model continue it and one that ends on a long silence
    delays the onset), then the batch is right-padded to a multiple of the tokenizer's frame size.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The number of channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used for padding.
        frame_size (`int`, *optional*, defaults to 1920):
            Number of audio samples per latent frame of the audio tokenizer (its `frame_size`).
        pad_to_multiple_of (`int`, *optional*, defaults to 1920):
            Pad the audio to a multiple of this value; a multiple of `frame_size` keeps every latent frame whole.
        end_on_pause (`bool`, *optional*, defaults to `True`):
            Whether to end each waveform on exactly `pause_duration` seconds of silence: the trailing silence (frames of
            `analysis_frame_duration` seconds more than `silence_floor_db` below the loudest one) is cut, the last
            `fade_out_duration` seconds of what remains are faded out, and `pause_duration` seconds of zeros appended.
        pause_duration (`float`, *optional*, defaults to 0.08):
            Duration of the silence appended by `end_on_pause`, in seconds.
        fade_out_duration (`float`, *optional*, defaults to 0.02):
            Duration of the fade-out applied by `end_on_pause` before the pause, in seconds.
        silence_floor_db (`float`, *optional*, defaults to 35.0):
            Level under the loudest analysis frame, in dB, below which `end_on_pause` treats trailing frames as silence.
        analysis_frame_duration (`float`, *optional*, defaults to 0.02):
            Duration of the frames `end_on_pause` measures the level of, in seconds.
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=24000,
        padding_value=0.0,
        frame_size=1920,
        pad_to_multiple_of=1920,
        end_on_pause=True,
        pause_duration=0.08,
        fade_out_duration=0.02,
        silence_floor_db=35.0,
        analysis_frame_duration=0.02,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.frame_size = frame_size
        self.pad_to_multiple_of = pad_to_multiple_of
        self.end_on_pause = end_on_pause
        self.pause_duration = pause_duration
        self.fade_out_duration = fade_out_duration
        self.silence_floor_db = silence_floor_db
        self.analysis_frame_duration = analysis_frame_duration

    def get_num_frames(self, padding_mask: "torch.Tensor") -> "torch.Tensor":
        """Number of latent frames the audio tokenizer's encoder computes from each waveform's samples (`latents_mask`)."""
        return (padding_mask.sum(-1) + self.frame_size - 1) // self.frame_size

    def _end_on_pause(self, waveform: "torch.Tensor") -> "torch.Tensor":
        frame_length = max(1, int(self.analysis_frame_duration * self.sampling_rate))
        num_frames = waveform.shape[-1] // frame_length
        if num_frames == 0:
            return waveform
        frame_rms = waveform[: num_frames * frame_length].reshape(num_frames, frame_length).square().mean(-1).sqrt()
        frame_db = 20 * torch.log10(frame_rms + 1e-12)
        loud_frames = torch.nonzero(frame_db > frame_db.max() - self.silence_floor_db).flatten()
        end = (int(loud_frames[-1]) + 1) * frame_length
        waveform = waveform[:end].clone()
        fade_length = min(int(self.fade_out_duration * self.sampling_rate), end)
        waveform[end - fade_length :] *= torch.linspace(1, 0, fade_length, dtype=waveform.dtype)
        pause = waveform.new_zeros(int(self.pause_duration * self.sampling_rate))
        return torch.cat([waveform, pause])

    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        end_on_pause: bool | None = None,
        padding: bool | str | PaddingStrategy | None = True,
        pad_to_multiple_of: int | None = None,
        max_length: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The mono waveform or batch of mono waveforms of the voice prompts.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            end_on_pause (`bool`, *optional*):
                Whether to end each waveform on a short pause, see the class docstring. Defaults to `self.end_on_pause`.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Padding strategy of the batch, see [`~SequenceFeatureExtractor.pad`].
            pad_to_multiple_of (`int`, *optional*):
                Pad the batch to a multiple of this value. Defaults to `self.pad_to_multiple_of`.
        """
        pad_to_multiple_of = self.pad_to_multiple_of if pad_to_multiple_of is None else pad_to_multiple_of
        end_on_pause = self.end_on_pause if end_on_pause is None else end_on_pause
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        audio = make_list_of_audio(audio)
        for idx, example in enumerate(audio):
            example = torch.as_tensor(example, dtype=torch.float32)
            if example.ndim != 1:
                raise ValueError(f"Audio should be mono, got shape: {tuple(example.shape)}")
            audio[idx] = self._end_on_pause(example) if end_on_pause else example

        output_values = BatchFeature({"input_values": audio})
        output_values = self.pad(
            output_values,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            max_length=max_length,
        )
        if return_attention_mask:
            output_values["padding_mask"] = output_values.pop("attention_mask")
        # (batch_size, audio_channels, num_samples), the audio tokenizer's input as `EncodecFeatureExtractor` returns it
        output_values["input_values"] = output_values["input_values"][:, None]
        return output_values


__all__ = ["PocketTTSFeatureExtractor"]
