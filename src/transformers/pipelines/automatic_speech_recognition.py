# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import subprocess
from typing import TYPE_CHECKING, Union

import numpy as np

from ..file_utils import is_torch_available
from ..utils import logging
from .base import ChunkPipeline


if TYPE_CHECKING:
    from ...feature_extraction_sequence_utils import SequenceFeatureExtractor

logger = logging.get_logger(__name__)

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_CTC_MAPPING, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename")
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


def apply_stride(tokens, stride):
    max_token_n = tokens.shape[-1]
    max_input_n = max(input_n for input_n, _, _ in stride)
    ratio = max_token_n / max_input_n
    for i, (input_n, left, right) in enumerate(stride):
        token_n = int(round(input_n * ratio))
        left_token = int(round(left / input_n * token_n))
        right_token = int(round((input_n - right) / input_n * token_n))
        # This is CTC to preseve decoding, we need to duplicate
        # next letter, and last letter

        first_letter = tokens[i, left_token]
        tokens[i, :left_token] = first_letter

        last_letter = tokens[i, right_token - 1]
        tokens[i, right_token:] = last_letter


def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for i in range(0, inputs_len, step):
        # add start and end paddings to the chunk
        chunk = inputs[i : i + chunk_len]
        processed = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        _stride_left = 0 if i == 0 else stride_left
        is_last = i + step >= inputs_len
        _stride_right = 0 if is_last else stride_right

        if chunk.shape[0] > _stride_left:
            yield {"is_last": is_last, "stride": (chunk.shape[0], _stride_left, _stride_right), **processed}


class AutomaticSpeechRecognitionPipeline(ChunkPipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats
    """

    def __init__(self, feature_extractor: Union["SequenceFeatureExtractor", str], *args, **kwargs):
        """
        Arguments:
            feature_extractor ([`SequenceFeatureExtractor`]):
                The feature extractor that will be used by the pipeline to encode waveform for the model.
            model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
                The model that will be used by the pipeline to make predictions. This needs to be a model inheriting
                from [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
            tokenizer ([`PreTrainedTokenizer`]):
                The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
                [`PreTrainedTokenizer`].
            chunk_length_s (`float`, *optional*, defaults to 0):
                The input length for in each chunk. If `0` then chunking is disabled (default). Only available for CTC
                models.
            stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`):
                The length of stride on the left and right of each chunk. Used only with `chunk_length_s > 0`. This
                enables the model to *see* more context and infer letters better than without this context but the
                pipeline discards the stride bits at the end to make the final reconstitution as perfect as possible.
            framework (`str`, *optional*):
                The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must
                be installed.

                If no framework is specified, will default to the one currently installed. If no framework is specified
                and both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no
                model is provided.
            device (`int`, *optional*, defaults to -1):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the
                model on the associated CUDA device id.
        """
        super().__init__(*args, **kwargs)

        self.feature_extractor = feature_extractor

        if self.framework == "tf":
            raise ValueError("The AutomaticSpeechRecognitionPipeline is only available in PyTorch.")

        self.check_model_type(dict(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items() + MODEL_FOR_CTC_MAPPING.items()))

        if self.model.__class__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.values():
            self.type = "seq2seq"
        elif (
            self.feature_extractor._processor_class
            and self.feature_extractor._processor_class.endswith("WithLM")
            and kwargs.get("decoder", None) is not None
        ):
            self.decoder = kwargs["decoder"]
            self.type = "ctc_with_lm"
        else:
            self.type = "ctc"

    def __call__(
        self,
        inputs: Union[np.ndarray, bytes, str],
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`] documentation for more
        information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str`):
                The inputs is either a raw waveform (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                at the correct sampling rate (no further check will be done) or a `str` that is the filename of the
                audio file, the file will be read at the correct sampling rate to get the waveform using *ffmpeg*. This
                requires *ffmpeg* to be installed on the system. If *inputs* is `bytes` it is supposed to be the
                content of an audio file and is interpreted by *ffmpeg* in the same way.

        Return:
            A `dict` with the following keys:

            - **text** (`str`) -- The recognized text.
        """
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        # No parameters on this pipeline right now
        preprocess_params = {}
        if "chunk_length_s" in kwargs:
            preprocess_params["chunk_length_s"] = kwargs["chunk_length_s"]
        if "stride_length_s" in kwargs:
            preprocess_params["stride_length_s"] = kwargs["stride_length_s"]
        return preprocess_params, {}, {}

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if isinstance(inputs, str):
            with open(inputs, "rb") as f:
                inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        if not isinstance(inputs, np.ndarray):
            raise ValueError("We expect a numpy ndarray as input")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            chunk_len = int(round(chunk_length_s * self.feature_extractor.sampling_rate))

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            stride_left = int(round(stride_length_s[0] * self.feature_extractor.sampling_rate))
            stride_right = int(round(stride_length_s[1] * self.feature_extractor.sampling_rate))

            if self.type != "ctc":
                raise ValueError(
                    "`chunk_length_s` is only valid for CTC models, use other chunking options for other models"
                )
            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            # make sure that
            for item in chunk_iter(inputs, self.feature_extractor, chunk_len, stride_left, stride_right):
                yield item
        else:
            processed = self.feature_extractor(
                inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
            )
            yield {"is_last": True, **processed}

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        if self.type == "seq2seq":
            encoder = self.model.get_encoder()
            # we need to pass `processed.get("attention_mask")` here since audio encoder
            # attention mask  length is different from expected text decoder `encoder_attention_mask` length
            # `generate` magic to create the mask automatically won't work, we basically need to help
            # it here.
            tokens = self.model.generate(
                encoder_outputs=encoder(**model_inputs), attention_mask=model_inputs.get("attention_mask")
            )
            out = {"tokens": tokens}
        elif self.type == "ctc_with_lm":
            outputs = self.model(**model_inputs)
            out = {"logits": outputs.logits}

        elif self.type == "ctc":
            stride = model_inputs.pop("stride", None)
            outputs = self.model(**model_inputs)
            tokens = outputs.logits.argmax(dim=-1)
            if stride is not None:
                if isinstance(stride, tuple):
                    stride = [stride]

                apply_stride(tokens, stride)
            out = {"tokens": tokens}
        else:
            logger.warning("This is an unknown class, treating it as CTC.")
            outputs = self.model(**model_inputs)
            tokens = outputs.logits.argmax(dim=-1)
            out = {"tokens": tokens}
        return {"is_last": is_last, **out}

    def postprocess(self, model_outputs):
        if self.type == "ctc_with_lm":
            logits = np.concatenate([outputs["logits"].numpy() for outputs in model_outputs], axis=1)
            logits = logits.squeeze(0)
            text = self.decoder.decode_beams(logits)[0][0]
        else:
            skip_special_tokens = self.type != "ctc"
            tokens = np.concatenate([outputs["tokens"].numpy() for outputs in model_outputs], axis=-1)
            tokens = tokens.squeeze(0)
            text = self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        return {"text": text}
