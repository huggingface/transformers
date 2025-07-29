# coding=utf-8
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
"""Processor class for Llasa."""

from typing import Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import is_torch_available


if is_torch_available():
    import torch


class LlasaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "tokenize": True,
            "continue_final_message": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class LlasaProcessor(ProcessorMixin):
    r"""
    Constructs a Llasa processor which wraps a [`DacFeatureExtractor`], [`LlasaTokenizer`], and a [`XCodec2Model`] into
    a single processor. It inherits, the audio feature extraction, tokenizer, and audio encode/decode functio-
    nalities. See [`~LlasaProcessor.__call__`], [`~LlasaProcessor.encode`], and [`~LlasaProcessor.decode`] for more
    information.

    Args:
        tokenizer (`LlasaTokenizer`):
            An instance of [`LlasaTokenizer`]. The tokenizer is a required input.
        audio_tokenizer (`XCodec2Model`):
            An instance of [`XCodec2Model`] used to encode/decode audio into/from codebooks. It is is a required input.
    """

    tokenizer_class = "LlasaTokenizer"
    # TODO use "audio_tokenizer_class" when merged https://github.com/huggingface/transformers/pull/37868
    # audio_tokenizer_class = "XCodec2Model"
    attributes = ["tokenizer"]
    optional_attributes = ["audio_codec"]

    def __init__(self, tokenizer, audio_codec):
        super().__init__(tokenizer, audio_codec=audio_codec)

    def __call__(
        self,
        text: Union[str, list[str]],
        **kwargs: Unpack[LlasaProcessorKwargs],
    ):
        """
        Main method to prepare text(s) to be fed as input to the model. The `text` argument is passed formatted into a
        chat template with start and end tokens for text understanding and speech generation, and then tokenized with
        [`~LlasaTokenizer.__call__`].
        """
        if not is_torch_available():
            raise ValueError(
                "The `LlasaProcessor` relies on the `audio_tokenizer` which requires `torch` but we couldn't "
                "find it in your environment. You can install torch via `pip install torch`."
            )

        if text is None:
            raise ValueError("You need to specify the `text` input to process.")

        output_kwargs = self._merge_kwargs(
            LlasaProcessorKwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        data = {}
        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # Within chat template
        chats = []
        for t in text:
            formatted_text = f"{self.tokenizer.llasa_token['text_understanding_start']}{t}{self.tokenizer.llasa_token['text_understanding_end']}"
            chats.append(
                [
                    {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                    {"role": "assistant", "content": self.tokenizer.llasa_token["speech_generation_start"]},
                ]
            )
        input_ids = self.tokenizer.apply_chat_template(chats, **text_kwargs)
        data.update({"input_ids": input_ids})
        data.update({"input_offset": [len(input_id) for input_id in input_ids]})
        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(
        self,
        decoder_input_ids: "torch.Tensor",
        input_offset: list[int],
        **kwargs: Unpack[LlasaProcessorKwargs],
    ) -> list["torch.Tensor"]:
        """
        Decodes a batch of audio codebook sequences into their respective audio waveforms via the
        `audio_tokenizer`. See [`~XCodec2Model.decode`] for more information.

        TODO: do proper batch decoding, atm looping over inputs

        Args:
            decoder_input_ids (`torch.Tensor`): The complete output sequence of the decoder.
        """
        audios = []
        for i in range(decoder_input_ids.shape[0]):
            # extract generated audio and remove end token
            generated_ids = decoder_input_ids[i, input_offset[i] : -1]

            # to speech tokens
            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = self.extract_speech_ids(speech_tokens)

            # decode speech tokens to waveform
            speech_tokens = torch.tensor(speech_tokens).to(self.audio_codec.device).unsqueeze(0).unsqueeze(0)
            audios.append(self.audio_codec.decode_code(speech_tokens)[0, 0, :])
        return audios

    def decode(
        self,
        decoder_input_ids: "torch.Tensor",
        input_offset: int,
        **kwargs: Unpack[LlasaProcessorKwargs],
    ) -> "torch.Tensor":
        """
        Decodes a single sequence of audio codebooks into the respective audio waveform via the
        `audio_tokenizer`. See [`~XCodec2Model.decode`] and [`~LlasaProcessor.batch_decode`] for more information.

        Prepare text for the model by adding start and end tokens with chat template.
        https://github.com/zhenye234/LLaSA_training/blob/ef5c2605927190ba40656d09b3a9e10df6631149/train_tts.py#L114
        Easier to see in their example on their model card: https://huggingface.co/HKUSTAudio/Llasa-1B#how-to-use
        """
        if decoder_input_ids.shape[0] != 1:
            raise ValueError(
                f"Expecting a single output to be decoded but received {decoder_input_ids.shape[0]} samples instead."
            )
        if isinstance(input_offset, list):
            if len(input_offset) != 1:
                raise ValueError(
                    f"Expecting a single input offset to be decoded but received {len(input_offset)} offsets instead."
                )
        else:
            input_offset = [input_offset]

        return self.batch_decode(decoder_input_ids, input_offset, **kwargs)[0]

    def extract_speech_ids(self, speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith("<|s_") and token_str.endswith("|>"):
                # TODO fix hardcoded integers
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                raise ValueError(f"Unexpected token: {token_str}")
        return speech_ids


__all__ = ["LlasaProcessor"]
