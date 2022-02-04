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
from collections import defaultdict
from typing import TYPE_CHECKING, Union

import numpy as np

from ..file_utils import is_torch_available
from ..utils import logging
from .audio_utils import ffmpeg_read
from .base import ChunkPipeline


if TYPE_CHECKING:
    from ...feature_extraction_sequence_utils import SequenceFeatureExtractor

logger = logging.get_logger(__name__)

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_CTC_MAPPING, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


def rescale_stride(tokens_or_logits, stride):
    """
    Rescales the stride values from audio space to tokens/logits space.

    (160_000, 16_000, 16_000) -> (2000, 200, 200) for instance.
    """
    # Shape is [B, SEQ] for tokens
    # [B, SEQ, V] for logits

    max_token_n = tokens_or_logits.shape[1]
    max_input_n = max(input_n for input_n, _, _ in stride)
    ratio = max_token_n / max_input_n
    new_strides = []
    for input_n, left, right in stride:
        token_n = int(round(input_n * ratio))
        left = int(round(left / input_n * token_n))
        right = int(round(right / input_n * token_n))
        new_stride = (token_n, left, right)
        new_strides.append(new_stride)

    return new_strides


def apply_stride(tokens, stride):
    new_stride = rescale_stride(tokens, stride)
    for i, (input_n, left, right) in enumerate(new_stride):
        left_token = left
        right_token = input_n - right
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
            model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
                The model that will be used by the pipeline to make predictions. This needs to be a model inheriting
                from [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
            tokenizer ([`PreTrainedTokenizer`]):
                The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
                [`PreTrainedTokenizer`].
            feature_extractor ([`SequenceFeatureExtractor`]):
                The feature extractor that will be used by the pipeline to encode waveform for the model.
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

        if self.model.__class__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.values():
            self.type = "seq2seq"
        elif (
            feature_extractor._processor_class
            and feature_extractor._processor_class.endswith("WithLM")
            and kwargs.get("decoder", None) is not None
        ):
            self.decoder = kwargs["decoder"]
            self.type = "ctc_with_lm"
        else:
            self.type = "ctc"

        if self.framework == "tf":
            raise ValueError("The AutomaticSpeechRecognitionPipeline is only available in PyTorch.")

        self.check_model_type(dict(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items() + MODEL_FOR_CTC_MAPPING.items()))

    def __call__(
        self,
        inputs: Union[np.ndarray, bytes, str],
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`] documentation for more
        information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is the filename of the audio file, the file will be read at the correct sampling rate
                      to get the waveform using *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be in the format `{"sampling_rate": int, "raw":
                      np.array}` with optionally a `"stride": (left: int, right: int)` than can ask the pipeline to
                      treat the first `left` samples and last `right` samples to be ignored in decoding (but used at
                      inference to provide more context to the model). Only use `stride` with CTC models.

        Return:
            `Dict`: A dictionary with the following keys:
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

        stride = None
        extra = {}
        if isinstance(inputs, dict):
            stride = inputs.pop("stride", None)
            _inputs = inputs.pop("raw")
            in_sampling_rate = inputs.pop("sampling_rate")
            extra = inputs
            inputs = _inputs
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                import torch
                from torchaudio import functional as F

                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
                ).numpy()
                ratio = self.feature_extractor.sampling_rate / in_sampling_rate
            else:
                ratio = 1
            if stride is not None:
                if stride[0] + stride[1] > inputs.shape[0]:
                    raise ValueError("Stride is too large for input")

                # Stride needs to get the chunk length here, it's going to get
                # swallowed by the `feature_extractor` later, and then batching
                # can add extra data in the inputs, so we need to keep track
                # of the original length in the stride so we can cut properly.
                stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))
        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
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

            if self.type not in {"ctc", "ctc_with_lm"}:
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
            if stride is not None:
                if self.model.__class__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.values():
                    raise ValueError("Stride is only usable with CTC models, try removing it")

                processed["stride"] = stride
            yield {"is_last": True, **processed, **extra}

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        if self.type == "seq2seq":
            encoder = self.model.get_encoder()
            # we need to pass `processed.get("attention_mask")` here since audio encoder
            # attention mask  length is different from expected text decoder `encoder_attention_mask` length
            # `generate` magic to create the mask automatically won't work, we basically need to help
            # it here.
            # Consume values so we can let extra information flow freely through
            # the pipeline (important for `partial` in microphone)
            input_features = model_inputs.pop("input_features")
            attention_mask = model_inputs.pop("attention_mask")
            tokens = self.model.generate(
                encoder_outputs=encoder(input_features=input_features, attention_mask=attention_mask),
                attention_mask=attention_mask,
            )
            out = {"tokens": tokens}
        elif self.type == "ctc_with_lm":
            stride = model_inputs.pop("stride", None)
            input_values = model_inputs.pop("input_values")
            attention_mask = model_inputs.pop("attention_mask", None)
            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            out = {"logits": logits}
            if stride is not None:
                # Send stride to `postprocess`.
                # it needs to be handled there where
                # the pieces are to be concatenated.
                if isinstance(stride, tuple):
                    out["stride"] = rescale_stride(logits, [stride])[0]
                else:
                    out["stride"] = rescale_stride(logits, stride)
        elif self.type == "ctc":
            stride = model_inputs.pop("stride", None)
            # Consume values so we can let extra information flow freely through
            # the pipeline (important for `partial` in microphone)
            input_values = model_inputs.pop("input_values")
            attention_mask = model_inputs.pop("attention_mask", None)
            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
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
        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}

    def postprocess(self, model_outputs):
        if self.type == "ctc_with_lm":
            final_logits = []
            for outputs in model_outputs:
                logits = outputs["logits"].numpy()
                stride = outputs.pop("stride", None)
                if stride is not None:
                    total_n, left, right = stride
                    # Total_n might be < logits.shape[1]
                    # because of padding, that's why
                    # we need to reconstruct this information
                    # This won't work with left padding (which doesn't exist right now)
                    right_n = total_n - right
                    logits = logits[:, left:right_n]
                final_logits.append(logits)
            logits = np.concatenate(final_logits, axis=1)
            logits = logits.squeeze(0)
            text = self.decoder.decode_beams(logits)[0][0]
        else:
            skip_special_tokens = self.type != "ctc"
            tokens = np.concatenate([outputs["tokens"].numpy() for outputs in model_outputs], axis=-1)
            tokens = tokens.squeeze(0)
            text = self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

        extra = defaultdict(list)
        for output in model_outputs:
            output.pop("tokens", None)
            output.pop("logits", None)
            for k, v in output.items():
                if k == "is_last":
                    continue
                extra[k].append(v)
        return {"text": text, **extra}
