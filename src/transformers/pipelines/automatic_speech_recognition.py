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
from .base import Pipeline


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


class AutomaticSpeechRecognitionPipeline(Pipeline):
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
                from [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`]
                for TensorFlow.
            tokenizer ([`PreTrainedTokenizer`]):
                The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
                [`PreTrainedTokenizer`].
            modelcard (`str` or [`ModelCard`], *optional*):
                Model card attributed to the model for this pipeline.
            framework (`str`, *optional*):
                The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified
                framework must be installed.

                If no framework is specified, will default to the one currently installed. If no framework is specified
                and both frameworks are installed, will default to the framework of the `model`, or to PyTorch if
                no model is provided.
            device (`int`, *optional*, defaults to -1):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the
                model on the associated CUDA device id.
        """
        super().__init__(*args, **kwargs)

        self.feature_extractor = feature_extractor

        if self.framework == "tf":
            raise ValueError("The AutomaticSpeechRecognitionPipeline is only available in PyTorch.")

        self.check_model_type(dict(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items() + MODEL_FOR_CTC_MAPPING.items()))

    def __call__(
        self,
        inputs: Union[np.ndarray, bytes, str],
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`]
        documentation for more information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str`):
                The inputs is either a raw waveform (`np.ndarray` of shape (n, ) of type `np.float32` or
                `np.float64`) at the correct sampling rate (no further check will be done) or a `str` that is
                the filename of the audio file, the file will be read at the correct sampling rate to get the waveform
                using *ffmpeg*. This requires *ffmpeg* to be installed on the system. If *inputs* is `bytes` it is
                supposed to be the content of an audio file and is interpreted by *ffmpeg* in the same way.

        Return:
            A `dict` with the following keys:

            - **text** (`str`) -- The recognized text.
        """
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        # No parameters on this pipeline right now
        return {}, {}, {}

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            with open(inputs, "rb") as f:
                inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        if not isinstance(inputs, np.ndarray):
            raise ValueError("We expect a numpy ndarray as input")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

        processed = self.feature_extractor(
            inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        return processed

    def _forward(self, model_inputs):
        model_class = self.model.__class__
        if model_class in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.values():
            encoder = self.model.get_encoder()
            # we need to pass `processed.get("attention_mask")` here since audio encoder
            # attention mask  length is different from expected text decoder `encoder_attention_mask` length
            # `generate` magic to create the mask automatically won't work, we basically need to help
            # it here.
            tokens = self.model.generate(
                encoder_outputs=encoder(**model_inputs), attention_mask=model_inputs.get("attention_mask")
            )
            tokens = tokens.squeeze(0)
        elif model_class in MODEL_FOR_CTC_MAPPING.values():
            outputs = self.model(**model_inputs)
            tokens = outputs.logits.squeeze(0).argmax(dim=-1)
        else:
            logger.warning("This is an unknown class, treating it as CTC.")
            outputs = self.model(**model_inputs)
            tokens = outputs.logits.squeeze(0).argmax(dim=-1)
        return tokens

    def postprocess(self, model_outputs):
        skip_special_tokens = False if "CTC" in self.tokenizer.__class__.__name__ else True
        recognized_string = self.tokenizer.decode(model_outputs, skip_special_tokens=skip_special_tokens)
        return {"text": recognized_string}
