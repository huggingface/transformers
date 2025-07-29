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
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import requests

from ..generation import GenerationConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import is_torch_available, is_torchaudio_available, is_torchcodec_available, logging
from .audio_utils import ffmpeg_read
from .base import ChunkPipeline


if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC

    from ..feature_extraction_sequence_utils import SequenceFeatureExtractor
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES


def rescale_stride(stride, ratio):
    """
    Rescales the stride values from audio space to tokens/logits space.

    (160_000, 16_000, 16_000) -> (2000, 200, 200) for instance.
    """
    # Shape is [B, SEQ] for tokens
    # [B, SEQ, V] for logits

    new_strides = []
    for input_n, left, right in stride:
        token_n = int(round(input_n * ratio))
        left = int(round(left / input_n * token_n))
        right = int(round(right / input_n * token_n))
        new_stride = (token_n, left, right)
        new_strides.append(new_stride)

    return new_strides


def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right, dtype=None):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        processed = feature_extractor(
            chunk,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if dtype is not None:
            processed = processed.to(dtype=dtype)
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        is_last = chunk_end_idx >= inputs_len
        _stride_right = 0 if is_last else stride_right

        chunk_len = chunk.shape[0]
        stride = (chunk_len, _stride_left, _stride_right)
        if chunk.shape[0] > _stride_left:
            yield {"is_last": is_last, "stride": stride, **processed}
        if is_last:
            break


def _find_longest_common_sequence(sequences, tokenizer):
    # TODO  Use a faster algorithm this can probably be done in O(n)
    # using suffix array.
    # It might be tedious to do because of fault tolerance.
    # We actually have a really good property which is that the total sequence
    # MUST be those subsequences in order.
    # Also the algorithm should be more tolerant to errors.
    sequence = [tok_id for tok_id in sequences[0][0].tolist() if tok_id not in tokenizer.all_special_ids]
    for new_seq in sequences[1:]:
        new_sequence = [tok_id for tok_id in new_seq[0].tolist() if tok_id not in tokenizer.all_special_ids]

        index = 0
        max_ = 0.0
        for i in range(1, len(new_sequence) + 1):
            # epsilon to favor long perfect matches
            eps = i / 10000.0
            matches = np.sum(np.array(sequence[-i:]) == np.array(new_sequence[:i]))
            matching = matches / i + eps
            if matches > 1 and matching > max_:
                index = i
                max_ = matching
        sequence.extend(new_sequence[index:])
    return np.array(sequence)


class AutomaticSpeechRecognitionPipeline(ChunkPipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats

    Unless the model you're using explicitly sets these generation parameters in its configuration files
    (`generation_config.json`), the following default values will be used:
    - max_new_tokens: 256
    - num_beams: 5

    Example:

    ```python
    >>> from transformers import pipeline

    >>> transcriber = pipeline(model="openai/whisper-base")
    >>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    {'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode waveform for the model.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        decoder (`pyctcdecode.BeamSearchDecoderCTC`, *optional*):
            [PyCTCDecode's
            BeamSearchDecoderCTC](https://github.com/kensho-technologies/pyctcdecode/blob/2fd33dc37c4111417e08d89ccd23d28e9b308d19/pyctcdecode/decoder.py#L180)
            can be passed for language model boosted decoding. See [`Wav2Vec2ProcessorWithLM`] for more information.
        chunk_length_s (`float`, *optional*, defaults to 0):
            The input length for in each chunk. If `chunk_length_s = 0` then chunking is disabled (default).

            <Tip>

            For more information on how to effectively use `chunk_length_s`, please have a look at the [ASR chunking
            blog post](https://huggingface.co/blog/asr-chunking).

            </Tip>

        stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`):
            The length of stride on the left and right of each chunk. Used only with `chunk_length_s > 0`. This enables
            the model to *see* more context and infer letters better than without this context but the pipeline
            discards the stride bits at the end to make the final reconstitution as perfect as possible.

            <Tip>

            For more information on how to effectively use `stride_length_s`, please have a look at the [ASR chunking
            blog post](https://huggingface.co/blog/asr-chunking).

            </Tip>

        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed. If no framework is specified, will default to the one currently installed. If no framework is
            specified and both frameworks are installed, will default to the framework of the `model`, or to PyTorch if
            no model is provided.
        device (Union[`int`, `torch.device`], *optional*):
            Device ordinal for CPU/GPU supports. Setting this to `None` will leverage CPU, a positive will run the
            model on the associated CUDA device id.
        torch_dtype (Union[`int`, `torch.dtype`], *optional*):
            The data-type (dtype) of the computation. Setting this to `None` will use float32 precision. Set to
            `torch.float16` or `torch.bfloat16` to use half-precision in the respective dtypes.

    """

    _pipeline_calls_generate = True
    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = True
    _load_tokenizer = True
    # Make sure the docstring is updated when the default generation config is changed
    _default_generation_config = GenerationConfig(
        max_new_tokens=256,
        num_beams=5,  # follows openai's whisper implementation
    )

    def __init__(
        self,
        model: "PreTrainedModel",
        feature_extractor: Union["SequenceFeatureExtractor", str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        decoder: Optional[Union["BeamSearchDecoderCTC", str]] = None,
        device: Union[int, "torch.device"] = None,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        # set the model type so we can check we have the right pre- and post-processing parameters
        if model.config.model_type == "whisper":
            self.type = "seq2seq_whisper"
        elif model.__class__.__name__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values():
            self.type = "seq2seq"
        elif (
            feature_extractor._processor_class
            and feature_extractor._processor_class.endswith("WithLM")
            and decoder is not None
        ):
            self.decoder = decoder
            self.type = "ctc_with_lm"
        else:
            self.type = "ctc"

        super().__init__(model, tokenizer, feature_extractor, device=device, torch_dtype=torch_dtype, **kwargs)

    def __call__(self, inputs: Union[np.ndarray, bytes, str, dict], **kwargs: Any) -> list[dict[str, Any]]:
        """
        Transcribe the audio sequence(s) given as inputs to text. See the [`AutomaticSpeechRecognitionPipeline`]
        documentation for more information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is either the filename of a local audio file, or a public URL address to download the
                      audio file. The file will be read at the correct sampling rate to get the waveform using
                      *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be in the format `{"sampling_rate": int, "raw":
                      np.array}` with optionally a `"stride": (left: int, right: int)` than can ask the pipeline to
                      treat the first `left` samples and last `right` samples to be ignored in decoding (but used at
                      inference to provide more context to the model). Only use `stride` with CTC models.
            return_timestamps (*optional*, `str` or `bool`):
                Only available for pure CTC models (Wav2Vec2, HuBERT, etc) and the Whisper model. Not available for
                other sequence-to-sequence models.

                For CTC models, timestamps can take one of two formats:
                    - `"char"`: the pipeline will return timestamps along the text for every character in the text. For
                        instance, if you get `[{"text": "h", "timestamp": (0.5, 0.6)}, {"text": "i", "timestamp": (0.7,
                        0.9)}]`, then it means the model predicts that the letter "h" was spoken after `0.5` and before
                        `0.6` seconds.
                    - `"word"`: the pipeline will return timestamps along the text for every word in the text. For
                        instance, if you get `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text": "there", "timestamp":
                        (1.0, 1.5)}]`, then it means the model predicts that the word "hi" was spoken after `0.5` and
                        before `0.9` seconds.

                For the Whisper model, timestamps can take one of two formats:
                    - `"word"`: same as above for word-level CTC timestamps. Word-level timestamps are predicted
                        through the *dynamic-time warping (DTW)* algorithm, an approximation to word-level timestamps
                        by inspecting the cross-attention weights.
                    - `True`: the pipeline will return timestamps along the text for *segments* of words in the text.
                        For instance, if you get `[{"text": " Hi there!", "timestamp": (0.5, 1.5)}]`, then it means the
                        model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
                        Note that a segment of text refers to a sequence of one or more words, rather than individual
                        words as with word-level timestamps.
            generate_kwargs (`dict`, *optional*):
                The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a
                complete overview of generate, check the [following
                guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

        Return:
            `Dict`: A dictionary with the following keys:
                - **text** (`str`): The recognized text.
                - **chunks** (*optional(, `list[Dict]`)
                    When using `return_timestamps`, the `chunks` will become a list containing all the various text
                    chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text":
                    "there", "timestamp": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
                    `"".join(chunk["text"] for chunk in output["chunks"])`.
        """
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(
        self,
        chunk_length_s=None,
        stride_length_s=None,
        ignore_warning=None,
        decoder_kwargs=None,
        return_timestamps=None,
        return_language=None,
        generate_kwargs=None,
    ):
        # No parameters on this pipeline right now
        preprocess_params = {}
        if chunk_length_s is not None:
            if self.type in ["seq2seq", "seq2seq_whisper"] and not ignore_warning:
                type_warning = (
                    "Using `chunk_length_s` is very experimental with seq2seq models. The results will not necessarily"
                    " be entirely accurate and will have caveats. More information:"
                    " https://github.com/huggingface/transformers/pull/20104. Ignore this warning with pipeline(...,"
                    " ignore_warning=True)."
                )
                if self.type == "seq2seq_whisper":
                    type_warning += (
                        " To use Whisper for long-form transcription, use rather the model's `generate` method directly "
                        "as the model relies on it's own chunking mechanism (cf. Whisper original paper, section 3.8. "
                        "Long-form Transcription)."
                    )
                logger.warning(type_warning)
            preprocess_params["chunk_length_s"] = chunk_length_s
        if stride_length_s is not None:
            preprocess_params["stride_length_s"] = stride_length_s

        forward_params = defaultdict(dict)
        if generate_kwargs is not None:
            forward_params.update(generate_kwargs)

        postprocess_params = {}
        if decoder_kwargs is not None:
            postprocess_params["decoder_kwargs"] = decoder_kwargs

        # in some models like whisper, the generation config has a `return_timestamps` key
        if hasattr(self, "generation_config") and hasattr(self.generation_config, "return_timestamps"):
            return_timestamps = return_timestamps or self.generation_config.return_timestamps

        if return_timestamps is not None:
            # Check whether we have a valid setting for return_timestamps and throw an error before we perform a forward pass
            if self.type == "seq2seq" and return_timestamps:
                raise ValueError("We cannot return_timestamps yet on non-CTC models apart from Whisper!")
            if self.type == "ctc_with_lm" and return_timestamps != "word":
                raise ValueError("CTC with LM can only predict word level timestamps, set `return_timestamps='word'`")
            if self.type == "ctc" and return_timestamps not in ["char", "word"]:
                raise ValueError(
                    "CTC can either predict character level timestamps, or word level timestamps. "
                    "Set `return_timestamps='char'` or `return_timestamps='word'` as required."
                )
            if self.type == "seq2seq_whisper" and return_timestamps == "char":
                raise ValueError(
                    "Whisper cannot return `char` timestamps, only word level or segment level timestamps. "
                    "Use `return_timestamps='word'` or `return_timestamps=True` respectively."
                )
            forward_params["return_timestamps"] = return_timestamps
            postprocess_params["return_timestamps"] = return_timestamps
        if return_language is not None:
            if self.type != "seq2seq_whisper":
                raise ValueError("Only Whisper can return language for now.")
            postprocess_params["return_language"] = return_language

        if getattr(self, "assistant_model", None) is not None:
            forward_params["assistant_model"] = self.assistant_model
        if getattr(self, "assistant_tokenizer", None) is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        stride = None
        extra = {}

        if is_torch_available():
            import torch

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cpu().numpy()

        if is_torchcodec_available():
            import torchcodec

            if isinstance(inputs, torchcodec.decoders.AudioDecoder):
                _audio_samples = inputs.get_all_samples()
                _array = _audio_samples.data
                inputs = {"array": _array, "sampling_rate": _audio_samples.sample_rate}

        if isinstance(inputs, dict):
            stride = inputs.pop("stride", None)
            # Accepting `"array"` which is the key defined in `datasets` for
            # better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to AutomaticSpeechRecognitionPipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array or torch tensor representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            extra = inputs
            inputs = _inputs
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AutomaticSpeechRecognitionPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )

                inputs = F.resample(
                    torch.from_numpy(inputs) if isinstance(inputs, np.ndarray) else inputs,
                    in_sampling_rate,
                    in_sampling_rate,
                    self.feature_extractor.sampling_rate,
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
            raise TypeError(f"We expect a numpy ndarray or torch tensor as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            # XXX: Carefully, this variable will not exist in `seq2seq` setting.
            # Currently chunking is not possible at this level for `seq2seq` so
            # it's ok.
            align_to = getattr(self.model.config, "inputs_to_logits_ratio", 1)
            chunk_len = int(round(chunk_length_s * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_left = int(round(stride_length_s[0] * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_right = int(round(stride_length_s[1] * self.feature_extractor.sampling_rate / align_to) * align_to)

            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            for item in chunk_iter(
                inputs, self.feature_extractor, chunk_len, stride_left, stride_right, self.torch_dtype
            ):
                yield {**item, **extra}
        else:
            if self.type == "seq2seq_whisper" and inputs.shape[0] > self.feature_extractor.n_samples:
                processed = self.feature_extractor(
                    inputs,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    truncation=False,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            else:
                if self.type == "seq2seq_whisper" and stride is None:
                    processed = self.feature_extractor(
                        inputs,
                        sampling_rate=self.feature_extractor.sampling_rate,
                        return_tensors="pt",
                        return_token_timestamps=True,
                        return_attention_mask=True,
                    )
                    extra["num_frames"] = processed.pop("num_frames")
                else:
                    processed = self.feature_extractor(
                        inputs,
                        sampling_rate=self.feature_extractor.sampling_rate,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
            if self.torch_dtype is not None:
                processed = processed.to(dtype=self.torch_dtype)
            if stride is not None:
                if self.type == "seq2seq":
                    raise ValueError("Stride is only usable with CTC models, try removing it !")

                processed["stride"] = stride
            yield {"is_last": True, **processed, **extra}

    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):
        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        num_frames = model_inputs.pop("num_frames", None)
        is_last = model_inputs.pop("is_last")

        if stride is not None and num_frames is not None:
            raise ValueError("num_frames must be used only when stride is None")

        if self.type in {"seq2seq", "seq2seq_whisper"}:
            # Consume values so we can let extra information flow freely through
            # the pipeline (important for `partial` in microphone)
            if "input_features" in model_inputs:
                inputs = model_inputs.pop("input_features")
            elif "input_values" in model_inputs:
                inputs = model_inputs.pop("input_values")
            else:
                raise ValueError(
                    "Seq2Seq speech recognition model requires either a "
                    f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
                )

            # custom processing for Whisper timestamps and word-level timestamps
            return_timestamps = return_timestamps or getattr(self.generation_config, "return_timestamps", False)
            if return_timestamps and self.type == "seq2seq_whisper":
                generate_kwargs["return_timestamps"] = bool(return_timestamps)
                if return_timestamps == "word":
                    generate_kwargs["return_token_timestamps"] = True
                    generate_kwargs["return_segments"] = True

            # User-defined `generation_config` passed to the pipeline call take precedence
            if "generation_config" not in generate_kwargs:
                generate_kwargs["generation_config"] = self.generation_config

            main_input_name = self.model.main_input_name if hasattr(self.model, "main_input_name") else "inputs"
            generate_kwargs = {
                main_input_name: inputs,
                "attention_mask": attention_mask,
                **generate_kwargs,
            }
            tokens = self.model.generate(**generate_kwargs)

            # whisper longform generation stores timestamps in "segments"
            if return_timestamps == "word" and self.type == "seq2seq_whisper":
                if "segments" not in tokens:
                    out = {"tokens": tokens["sequences"], "token_timestamps": tokens["token_timestamps"]}
                else:
                    token_timestamps = [
                        torch.cat([segment["token_timestamps"] for segment in segment_list])
                        for segment_list in tokens["segments"]
                    ]
                    out = {"tokens": tokens["sequences"], "token_timestamps": token_timestamps}
            else:
                out = {"tokens": tokens}
            if self.type == "seq2seq_whisper":
                if stride is not None:
                    out["stride"] = stride

        else:
            inputs = {
                self.model.main_input_name: model_inputs.pop(self.model.main_input_name),
                "attention_mask": attention_mask,
            }
            outputs = self.model(**inputs)
            logits = outputs.logits

            if self.type == "ctc_with_lm":
                out = {"logits": logits}
            else:
                out = {"tokens": logits.argmax(dim=-1)}
            if stride is not None:
                # Send stride to `postprocess`.
                # it needs to be handled there where
                # the pieces are to be concatenated.
                ratio = 1 / self.model.config.inputs_to_logits_ratio
                if isinstance(stride, tuple):
                    out["stride"] = rescale_stride([stride], ratio)[0]
                else:
                    out["stride"] = rescale_stride(stride, ratio)
        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}

    def postprocess(
        self, model_outputs, decoder_kwargs: Optional[dict] = None, return_timestamps=None, return_language=None
    ):
        # Optional return types
        optional = {}

        final_items = []
        key = "logits" if self.type == "ctc_with_lm" else "tokens"
        stride = None
        for outputs in model_outputs:
            if self.framework == "pt" and outputs[key].dtype in (torch.bfloat16, torch.float16):
                items = outputs[key].to(torch.float32).numpy()
            else:
                items = outputs[key].numpy()
            stride = outputs.get("stride", None)
            if stride is not None and self.type in {"ctc", "ctc_with_lm"}:
                total_n, left, right = stride
                # Total_n might be < logits.shape[1]
                # because of padding, that's why
                # we need to reconstruct this information
                # This won't work with left padding (which doesn't exist right now)
                right_n = total_n - right
                items = items[:, left:right_n]
            final_items.append(items)

        if stride and self.type == "seq2seq":
            items = _find_longest_common_sequence(final_items, self.tokenizer)
        elif self.type == "seq2seq_whisper":
            time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
            # Send the chunking back to seconds, it's easier to handle in whisper
            sampling_rate = self.feature_extractor.sampling_rate
            for output in model_outputs:
                if "stride" in output:
                    chunk_len, stride_left, stride_right = output["stride"]
                    # Go back in seconds
                    chunk_len /= sampling_rate
                    stride_left /= sampling_rate
                    stride_right /= sampling_rate
                    output["stride"] = chunk_len, stride_left, stride_right

            text, optional = self.tokenizer._decode_asr(
                model_outputs,
                return_timestamps=return_timestamps,
                return_language=return_language,
                time_precision=time_precision,
            )
        else:
            items = np.concatenate(final_items, axis=1)
            items = items.squeeze(0)

        if self.type == "ctc_with_lm":
            if decoder_kwargs is None:
                decoder_kwargs = {}
            beams = self.decoder.decode_beams(items, **decoder_kwargs)
            text = beams[0][0]
            if return_timestamps:
                # Simply cast from pyctcdecode format to wav2vec2 format to leverage
                # pre-existing code later
                chunk_offset = beams[0][2]
                offsets = []
                for word, (start_offset, end_offset) in chunk_offset:
                    offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
        elif self.type != "seq2seq_whisper":
            skip_special_tokens = self.type != "ctc"
            text = self.tokenizer.decode(items, skip_special_tokens=skip_special_tokens)
            if return_timestamps:
                offsets = self.tokenizer.decode(
                    items, skip_special_tokens=skip_special_tokens, output_char_offsets=True
                )["char_offsets"]
                if return_timestamps == "word":
                    offsets = self.tokenizer._get_word_offsets(offsets, self.tokenizer.replace_word_delimiter_char)

        if return_timestamps and self.type not in {"seq2seq", "seq2seq_whisper"}:
            chunks = []
            for item in offsets:
                start = item["start_offset"] * self.model.config.inputs_to_logits_ratio
                start /= self.feature_extractor.sampling_rate

                stop = item["end_offset"] * self.model.config.inputs_to_logits_ratio
                stop /= self.feature_extractor.sampling_rate

                chunks.append({"text": item[return_timestamps], "timestamp": (start, stop)})
            optional["chunks"] = chunks

        extra = defaultdict(list)
        for output in model_outputs:
            output.pop("tokens", None)
            output.pop("logits", None)
            output.pop("is_last", None)
            output.pop("stride", None)
            output.pop("token_timestamps", None)
            for k, v in output.items():
                extra[k].append(v)
        return {"text": text, **optional, **extra}
