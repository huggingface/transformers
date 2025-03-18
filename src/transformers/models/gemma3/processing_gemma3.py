# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
import re
from typing import List, Optional, Union, Tuple
from math import ceil

import numpy as np
import torch
import scipy
from torch.nn.utils.rnn import pad_sequence

from ...feature_extraction_utils import BatchFeature
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, AudioKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import to_py_obj, TensorType

from ...audio_utils import AudioInput

class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]
    do_convert_rgb: Optional[bool]


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }

def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank the same as SpeechLib FbankFC.
    Args:
        sample_rate (int): Sample rate in Hz. number > 0 [scalar]
        n_fft (int): FFT size. int > 0 [scalar]
        n_mel (int): Mel filter size. int > 0 [scalar]
        fmin (float): lowest frequency (in Hz). If None use 0.0.
            float >= 0 [scalar]
        fmax: highest frequency (in Hz). If None use sample_rate / 2.
            float >= 0 [scalar]
    Returns
        out (numpy.ndarray): Mel transform matrix
            [shape=(n_mels, 1 + n_fft/2)]
    """

    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negtive"
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)

    khi = max(khi, klo)

    # Spec 2: SpeechLib uses trianges in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix


class Gemma3AudioFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_audio_embeds", "audio_embed_sizes", "audio_attention_mask"]
    feature_extractor_type = "Gemma3AudioFeatureExtractor"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.compression_rate = audio_compression_rate
        self.qformer_compression_rate = audio_downsample_rate
        self.feat_stride = audio_feat_stride

        self._eightk_method = "fillzero"
        self._mel = speechlib_mel(sampling_rate, 512, feature_size, fmin=None, fmax=7690).T

        self._hamming400 = np.hamming(400)  # for 16k audio
        self._hamming200 = np.hamming(200)  # for 8k audio

    def duration_to_frames(self, duration):
        """duration in s, estimated frames"""
        frame_rate = 10

        num_frames = duration * 1000 // frame_rate
        return num_frames

    def __call__(
        self,
        audios: List[AudioInput],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        # Ref: https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py#L161
        returned_input_audio_embeds = []
        returned_audio_embed_sizes = []
        audio_frames_list = []

        for audio_data, sample_rate in audios:
            audio_embeds = self._extract_features(audio_data, sample_rate)
            audio_frames = len(audio_embeds) * self.feat_stride
            audio_embed_size = self._compute_audio_embed_size(audio_frames)
            returned_input_audio_embeds.append(torch.tensor(audio_embeds))
            returned_audio_embed_sizes.append(torch.tensor(audio_embed_size).long())
            audio_frames_list.append(audio_frames)

        returned_input_audio_embeds = pad_sequence(
            returned_input_audio_embeds, batch_first=True
        )
        returned_audio_embed_sizes = torch.stack(returned_audio_embed_sizes, dim=0)
        audio_frames = torch.tensor(audio_frames_list)
        returned_audio_attention_mask = torch.arange(0, audio_frames.max()).unsqueeze(0) < audio_frames.unsqueeze(1) if len(audios) > 1 else None

        data = {
            "input_audio_embeds": returned_input_audio_embeds,
            "audio_embed_sizes": returned_audio_embed_sizes,
        }
        if returned_audio_attention_mask is not None:
            data["audio_attention_mask"] = returned_audio_attention_mask

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _extract_spectrogram(self, wav, fs):
        """Extract spectrogram features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        if wav.ndim > 1:
            wav = np.squeeze(wav)

        # by default, we extract the mean if stereo
        if len(wav.shape) == 2:
            wav = wav.mean(1)

        # Resample to 16000 or 8000 if needed
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 16000)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 8000)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"Unsupported sample rate {fs}")

        if fs == 8000:
            if self._eightk_method == "resample":
                # Input audio is 8 kHz. Convert to 16 kHz before feature
                # extraction
                wav = scipy.signal.resample_poly(wav, 2, 1)
                fs = 16000
            # Do nothing here for fillzero method
        elif fs != 16000:
            # Input audio is not a supported sample rate.
            raise RuntimeError(f"Input data using an unsupported sample rate: {fs}")

        preemphasis = 0.97

        if fs == 8000:
            n_fft = 256
            win_length = 200
            hop_length = 80
            fft_window = self._hamming200
        elif fs == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            fft_window = self._hamming400

        # Spec 1: SpeechLib cut remaining sample insufficient for a hop
        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        # Here we don't use stride_tricks since the input array may not satisfy
        # memory layout requirement and we need writeable output
        # Here we only use list of views before copy to desination
        # so it is more efficient than broadcasting
        y_frames = np.array(
            [wav[_stride : _stride + win_length] for _stride in range(0, hop_length * n_batch, hop_length)],
            dtype=np.float32,
        )

        # Spec 2: SpeechLib applies preemphasis within each batch
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768

        S = np.fft.rfft(fft_window * y_frames, n=n_fft, axis=1).astype(np.complex64)

        if fs == 8000:
            # Need to pad the output to look like 16 kHz data but with zeros in
            # the 4 to 8 kHz bins.
            frames, bins = S.shape
            padarray = np.zeros((frames, bins))
            S = np.concatenate((S[:, 0:-1], padarray), axis=1)  # Nyquist bin gets set to zero

        spec = np.abs(S).astype(np.float32)
        return spec

    def _extract_features(self, wav, fs):
        """Extract log filterbank features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        spec = self._extract_spectrogram(wav, fs)
        spec_power = spec**2

        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)

        return log_fbank

    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.compression_rate
        remainder = audio_frames % self.compression_rate

        result = integer if remainder == 0 else integer + 1

        integer = result // self.qformer_compression_rate
        remainder = result % self.qformer_compression_rate
        result = integer if remainder == 0 else integer + 1  # qformer compression

        return result

class Gemma3Processor(ProcessorMixin):
    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_length"]
    image_processor_class = "AutoImageProcessor"
    feature_extractor_class = "Gemma3AudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        feature_extractor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"

        self.audio_token_id = tokenizer.audio_token_id
        self.boa_token = tokenizer.boa_token
        self.eoa_token = tokenizer.eoa_token
        self.audio_token = tokenizer.audio_token
        
        super().__init__(
            image_processor=image_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos=None,
        audios: AudioInput = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        if images is not None:
            batched_images = make_nested_list_of_images(images)

            image_inputs = self.image_processor(batched_images, **output_kwargs["images_kwargs"])

            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.boi_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            batch_num_crops = to_py_obj(image_inputs.pop("num_crops"))
            for batch_idx, (prompt, images, num_crops) in enumerate(zip(text, batched_images, batch_num_crops)):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(num_crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len(self.boi_token) :]
                        text[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            text = [prompt.replace(self.boi_token, self.full_image_sequence) for prompt in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
    
        audio_inputs = {}
        if audios is not None:
            def replace_tokens_sequentially(prompt, boa_token, full_audio_sequences):
                parts = prompt.split(boa_token)
                result = ""
                for i in range(len(parts) - 1):
                    result += parts[i]
                    if i < len(full_audio_sequences):
                        result += full_audio_sequences[i]
                    else:
                        result += boa_token
                result += parts[-1]
                return result

            audio_inputs = self.feature_extractor(audios[0])

            full_audio_sequences = []
            for i, embed_size in enumerate(audio_inputs.audio_embed_sizes):
                audio_tokens_expanded = "".join([self.audio_token] * embed_size)
                full_audio_sequence = f"\n\n{self.boa_token}{audio_tokens_expanded}{self.eoa_token}\n\n"
                full_audio_sequences.append(full_audio_sequence)
            text = [replace_tokens_sequentially(prompt, self.boa_token, full_audio_sequences) for prompt in text]

        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"], return_tensors="np")

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        array_ids = np.array(text_inputs["input_ids"])
        mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
        mm_token_type_ids[array_ids == self.image_token_id] = 1
        mm_token_type_ids[array_ids == self.audio_token_id] = 2
        text_inputs = {k: v.tolist() for k, v in text_inputs.items()}  # in case user requested list inputs
        text_inputs["token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs, **audio_inputs}, tensor_type=return_tensors)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

__all__ = ["Gemma3Processor", "Gemma3AudioFeatureExtractor"]
