# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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
"""HiggsAudioProcessor."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import is_soundfile_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from .modeling_higgs_audio import _ceil_to_nearest

if is_soundfile_available():
    import soundfile as sf


logger = logging.get_logger(__name__)


class HiggsAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"pad_token_id": 128001},
        "audio_kwargs": {
            "sampling_rate": 24000,
            "audio_in_token": "<|AUDIO|>",
            "audio_out_token": "<|AUDIO_OUT|>",
            "audio_in_token_idx": 128015,
            "audio_out_token_idx": 128016,
            "audio_stream_bos_id": 1024,
            "audio_stream_eos_id": 1025,
            "audio_num_codebooks": 8,
            "audio_codebook_size": 1024,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@dataclass
class HiggsAudioChatSample:
    # Shape (seq_len,): The input text tokens.
    input_ids: "torch.LongTensor"

    # Shape (seq_len,): The label ids.
    label_ids: "torch.LongTensor"

    # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    audio_ids_concat: "torch.LongTensor"

    # Here `audio_seq_len` is the length of the concatenated audio tokens.
    # Shape (num_audios,): The start index of each audio token in the concatenated audio tokens.
    audio_ids_start: "torch.LongTensor"

    # Shape (total_wv_length,): The concatenated audio waveforms for audio-in features.
    audio_waveforms_concat: "torch.Tensor"

    # Shape (num_audios,): The start index of each audio waveform in the concatenated audio waveforms.
    audio_waveforms_start: "torch.LongTensor"

    # Shape (num_audios,): The sampling rate of the audio waveforms.
    audio_sample_rate: "torch.Tensor"

    # Shape (num_audios,) -1 means unknown speaker: The speaker indices for each audio.
    audio_speaker_indices: "torch.LongTensor"

    # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    audio_label_ids_concat: Optional["torch.LongTensor"] = None

    def get_audio_codes(self, idx):
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_ids_concat[:, code_start:code_end]

    def get_audio_codes_labels(self, idx):
        if self.audio_label_ids_concat is None:
            return None
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_label_ids_concat[:, code_start:code_end]

    def get_wv(self, idx):
        wv_start = self.audio_waveforms_start[idx]
        sr = self.audio_sample_rate[idx]
        if idx < len(self.audio_waveforms_start) - 1:
            wv_end = self.audio_waveforms_start[idx + 1]
        else:
            wv_end = self.audio_waveforms_concat.shape[-1]
        return self.audio_waveforms_concat[wv_start:wv_end], sr


@dataclass
class HiggsAudioBatchInput:
    # shape (bsz, seq_len).
    input_ids: "torch.LongTensor"

    # shape (bsz, seq_len).
    attention_mask: "torch.Tensor"

    # shape (num_codebooks, audio_out_total_length).
    audio_out_ids: Optional["torch.LongTensor"]

    # shape (num_audio_out,)
    audio_out_ids_start: Optional["torch.LongTensor"]

    # Currently, we concatenate audio segments along dim 0 to handle variadic audio segment length.
    # For example:
    #   audio_out_ids_start = [0, 2, 4, 8]
    # The first two audio segments come from the same sample in a batch,
    # and the other two come from different samples.
    # shape (num_codebooks, audio_in_total_length).
    audio_in_ids: Optional["torch.LongTensor"]

    # shape (num_audio_in,)
    audio_in_ids_start: Optional["torch.LongTensor"]

    # shape (bsz, seq_len).
    label_ids: Optional["torch.LongTensor"]

    # shape (num_codebooks, audio_out_total_length).
    label_audio_ids: Optional["torch.LongTensor"]


@dataclass
class HiggsAudioResponse:
    audio: Optional[np.ndarray] = None
    generated_text: str = ""
    sampling_rate: Optional[int] = None


def build_delay_pattern_mask(
    input_ids: "torch.LongTensor",
    bos_token_id: int,
    pad_token_id: int,
):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=5. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  P,  P,  P]
    - [ B,  *,  *,  *,  *,  *,  P,  P]
    - [ B,  B,  *,  *,  *,  *,  *,  P]
    - [ B,  B,  B,  *,  *,  *,  *,  *]

    where B indicates the delay token id, P is the special padding token id and `*` indicates that the original audio token.

    Now let's consider the case where we have a sequence of audio tokens to condition on.
    The audio tokens were originally in the following non-delayed form:

    - [a, b]
    - [c, d]
    - [e, f]
    - [g, h]

    After conversion, we get the following delayed form:
    - [a, b, -1, -1, -1]
    - [B, c,  d, -1, -1]
    - [B, B,  e,  f, -1]
    - [B, B,  B,  g,  h]

    Note that we have a special token `-1` that indicates it should be replaced by a new token we see in the generation phase.
    In that case, we should override the `-1` tokens in auto-regressive generation.

    Args:
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape (bsz, num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids, input_ids_with_gen_mask


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


class HiggsAudioProcessor(ProcessorMixin):
    r"""
    Constructs a Higgs Audio processor which wraps a [`DacFeatureExtractor`], a [`AutoTokenizer`],
    and a [`HiggsAudioTokenizer`] into a single processor. It inherits, the audio feature extraction, tokenizer,
    and audio encode/decode functionalities.
    See [`~HiggsAudioProcessor.__call__`] and [`~HiggsAudioProcessor.decode`] for more information.

    Args:
        feature_extractor (`DacFeatureExtractor`):
            An instance of [`DacFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
        audio_tokenizer (`HiggsAudioTokenizer`):
            An instance of [`HiggsAudioTokenizer`]. The audio tokenizer is a required input.
        chat_template (`str`, *optional*):
            A template string for chat formatting when combining text and audio interactions.
    """

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "HiggsAudioTokenizer"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer,
        chat_template=None,
    ):
        if chat_template is None:
            chat_template = self.default_chat_template

        super().__init__(
            feature_extractor,
            tokenizer,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template,
        )

    def __call__(
        self,
        text: Union[str, list[str]],
        audio: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[HiggsAudioProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare inputs for the model from text and audio.

        Args:
            text: Single text string or list of text strings (chat format)
            audio: Optional audio input as numpy array(s)
            output_labels: Whether to generate label tokens for training
            **kwargs: Additional arguments

        Returns:
            BatchFeature of model inputs including tokenized text and processed audio
        """
        if not is_torch_available():
            raise ValueError(
                "The `HiggsAudioProcessor` requires `torch` but we couldn't "
                "find it in your environment. You can install torch via `pip install torch`."
            )
        # Handle text input
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            HiggsAudioProcessorKwargs,
            **kwargs,
        )

        common_kwargs = output_kwargs["common_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        audio_in_token = audio_kwargs["audio_in_token"]
        audio_out_token = audio_kwargs["audio_out_token"]

        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        # If text is already a rendered string (single item), use it directly
        num_audio_tokens = []
        for rendered_text in text:
            # Extract audio count from the rendered text for validation
            num_audio_token = rendered_text.count(audio_in_token) + rendered_text.count(audio_out_token)
            num_audio_tokens.append(num_audio_token)

        # Validate audio input count
        if audio is not None:
            num_audios = 1 if isinstance(audio, np.ndarray) else len(audio)
            if sum(num_audio_tokens) != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {audio_in_token} and {audio_out_token} token{'s' if num_audio_tokens > 1 else ''} "
                    f"in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
                )

        # Process audio and expand audio tokens if needed
        audio_ids_list = []

        if audio is not None:
            # Ensure audio is a list
            if isinstance(audio, np.ndarray):
                audio = [audio]

            # Process each audio
            for i, audio_data in enumerate(audio):
                # Assume audio_data comes with the correct sample rate or resample
                if hasattr(audio_data, "shape"):
                    processed_audio = audio_data
                else:
                    processed_audio = np.array(audio_data)

                # Generate audio tokens
                input_values = self.feature_extractor(processed_audio)
                input_values = torch.tensor(
                    input_values["input_values"][0], device=self.audio_tokenizer.device
                ).unsqueeze(0)
                audio_ids = self.audio_tokenizer.encode(input_values)
                audio_ids_list.append(audio_ids[0].squeeze(0).cpu())

        # Create samples
        samples = []
        cumsum_audio_tokens = np.cumsum([0] + num_audio_tokens)
        for index, rendered_text in enumerate(text):
            input_tokens = self.tokenizer.encode(rendered_text, add_special_tokens=False)

            sliced_audio_ids_list = audio_ids_list[cumsum_audio_tokens[index] : cumsum_audio_tokens[index + 1]]

            # Prepare audio data for sample
            if sliced_audio_ids_list:
                audio_ids_start = torch.tensor(
                    np.cumsum([0] + [ids.shape[1] for ids in sliced_audio_ids_list])[:-1], dtype=torch.long
                )
                audio_ids_concat = torch.cat(sliced_audio_ids_list, dim=1)
            else:
                audio_ids_start = None
                audio_ids_concat = None

            # Create dataset sample
            sample = HiggsAudioChatSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=torch.LongTensor(input_tokens) if output_labels else None,
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            samples.append(sample)

        # Use collator to process the sample
        batch_data = self.process_sample(
            samples,
            audio_in_token_idx=audio_kwargs["audio_in_token_idx"],
            audio_out_token_idx=audio_kwargs["audio_out_token_idx"],
            audio_stream_bos_id=audio_kwargs["audio_stream_bos_id"],
            audio_stream_eos_id=audio_kwargs["audio_stream_eos_id"],
            pad_token_id=text_kwargs["pad_token_id"],
            audio_num_codebooks=audio_kwargs["audio_num_codebooks"],
        )
        inputs = asdict(batch_data) if hasattr(batch_data, "__dict__") else batch_data._asdict()
        inputs = {k: v for k, v in inputs.items() if v is not None}

        return BatchFeature(data=inputs, tensor_type=return_tensors)

    @staticmethod
    def process_sample(
        batch: list[HiggsAudioChatSample],
        audio_in_token_idx=128015,
        audio_out_token_idx=128016,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        pad_token_id=128001,
        audio_num_codebooks=8,
        pad_left=False,
        round_to=1,
        return_audio_in_tokens=False,
        mask_audio_out_token_label=True,
        disable_audio_codes_transform=False,
    ):
        """Collate the input data with support for long audio processing."""

        label_ids = None
        label_audio_ids = None
        if all(ele.label_ids is None for ele in batch):
            return_labels = False
        else:
            return_labels = True

        processed_batch = batch

        # Get the max sequence length based on processed batch
        max_seq_length = _ceil_to_nearest(max([len(sample.input_ids) for sample in processed_batch]), round_to)

        # Get the ids for audio-in and audio-out for each batch
        audio_in_ids_l = []
        audio_out_ids_l = []
        audio_in_label_ids_l = None
        audio_out_label_ids_l = None

        if return_labels:
            audio_out_no_train_flag = []  # Whether the audio-out data should be trained on or not.

        # Process the audio inputs and outputs
        for i in range(len(processed_batch)):
            audio_in_mask = processed_batch[i].input_ids == audio_in_token_idx
            audio_out_mask = processed_batch[i].input_ids == audio_out_token_idx
            audio_ids = torch.ones_like(processed_batch[i].input_ids)
            audio_ids[audio_in_mask ^ audio_out_mask] = torch.cumsum(audio_ids[audio_in_mask ^ audio_out_mask], 0) - 1
            audio_in_ids = audio_ids[audio_in_mask]
            audio_out_ids = audio_ids[audio_out_mask]

            if return_labels:
                audio_out_no_train_flag.append(processed_batch[i].label_ids[audio_out_mask] < 0)
                if mask_audio_out_token_label:
                    processed_batch[i].label_ids[audio_out_mask] = -100

            # Process audio inputs
            if return_audio_in_tokens:
                audio_in_ids_l.extend(
                    [processed_batch[i].get_audio_codes(idx)[:audio_num_codebooks, :] for idx in audio_in_ids]
                )
                if processed_batch[i].audio_label_ids_concat is not None:
                    if audio_in_label_ids_l is None:
                        audio_in_label_ids_l = []
                    audio_in_label_ids_l.extend(
                        [
                            processed_batch[i].get_audio_codes_labels(idx)[:audio_num_codebooks, :]
                            for idx in audio_in_ids
                        ]
                    )

            audio_out_ids_l.extend(
                [processed_batch[i].get_audio_codes(idx)[:audio_num_codebooks, :] for idx in audio_out_ids]
            )

            if processed_batch[i].audio_label_ids_concat is not None:
                if audio_out_label_ids_l is None:
                    audio_out_label_ids_l = []
                audio_out_label_ids_l.extend(
                    [processed_batch[i].get_audio_codes_labels(idx)[:audio_num_codebooks, :] for idx in audio_out_ids]
                )

        if return_labels:
            audio_out_no_train_flag = torch.cat(audio_out_no_train_flag, dim=0)

        # Process audio input tokens
        if len(audio_in_ids_l) > 0:
            # Append audio-stream-bos and eos tokens
            new_audio_in_ids_l = []
            for ele in audio_in_ids_l:
                if disable_audio_codes_transform:
                    # Do not add audio-stream-bos or eos tokens.
                    audio_codes = ele
                else:
                    audio_codes = torch.cat(
                        [
                            torch.full((ele.shape[0], 1), audio_stream_bos_id, dtype=torch.long),
                            ele,
                            torch.full((ele.shape[0], 1), audio_stream_eos_id, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    audio_codes = build_delay_pattern_mask(
                        audio_codes.unsqueeze(0),
                        bos_token_id=audio_stream_bos_id,
                        pad_token_id=audio_stream_eos_id,
                    )[0].squeeze(0)
                new_audio_in_ids_l.append(audio_codes)
            audio_in_ids = torch.cat(new_audio_in_ids_l, dim=1).long()
            audio_in_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_codes.shape[1] for audio_codes in new_audio_in_ids_l[:-1]]), dim=0
            )
        else:
            audio_in_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_in_ids_start = torch.zeros(0, dtype=torch.long)

        # Process audio output tokens
        if len(audio_out_ids_l) > 0:
            new_audio_out_ids_l = []
            label_audio_ids_l = []
            for idx, ele in enumerate(audio_out_ids_l):
                if disable_audio_codes_transform:
                    # Do not add audio-stream-bos or eos tokens.
                    audio_codes = ele
                    if return_labels:
                        label_audio_ids = audio_out_label_ids_l[idx]
                else:
                    audio_codes = torch.cat(
                        [
                            torch.full((ele.shape[0], 1), audio_stream_bos_id, dtype=torch.long),
                            ele,
                            torch.full((ele.shape[0], 1), audio_stream_eos_id, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    if return_labels:
                        label_audio_ids = torch.cat(
                            [
                                torch.full((ele.shape[0], 1), -100, dtype=torch.long),
                                ele,
                                torch.full((ele.shape[0], 1), audio_stream_eos_id, dtype=torch.long),
                            ],
                            dim=1,
                        )

                    audio_codes = build_delay_pattern_mask(
                        audio_codes.unsqueeze(0),
                        bos_token_id=audio_stream_bos_id,
                        pad_token_id=audio_stream_eos_id,
                    )[0].squeeze(0)
                    if return_labels:
                        label_audio_ids = build_delay_pattern_mask(
                            label_audio_ids.unsqueeze(0),
                            bos_token_id=-100,
                            pad_token_id=-100,
                        )[0].squeeze(0)
                new_audio_out_ids_l.append(audio_codes)

                if return_labels:
                    if audio_out_no_train_flag[idx]:
                        label_audio_ids[:] = -100
                    label_audio_ids_l.append(label_audio_ids)

            audio_out_ids = torch.cat(new_audio_out_ids_l, dim=1).long()
            if return_labels:
                label_audio_ids = torch.cat(label_audio_ids_l, dim=1).long()
            audio_out_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_codes.shape[1] for audio_codes in new_audio_out_ids_l[:-1]]), dim=0
            )
        else:
            audio_out_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_out_ids_start = torch.zeros(0, dtype=torch.long)
            if return_labels:
                label_audio_ids = torch.zeros((0, 0), dtype=torch.long)

        # Handle padding for input ids and attention mask
        if pad_left:
            input_ids = torch.stack(
                [
                    F.pad(ele.input_ids, (max_seq_length - len(ele.input_ids), 0), value=pad_token_id)
                    for ele in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(ele.label_ids, (max_seq_length - len(ele.label_ids), 0), value=-100)
                        for ele in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(torch.ones_like(ele.input_ids), (max_seq_length - len(ele.input_ids), 0), value=0)
                    for ele in processed_batch
                ]
            )
        else:
            input_ids = torch.stack(
                [
                    F.pad(ele.input_ids, (0, max_seq_length - len(ele.input_ids)), value=pad_token_id)
                    for ele in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(ele.label_ids, (0, max_seq_length - len(ele.label_ids)), value=-100)
                        for ele in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(torch.ones_like(ele.input_ids), (0, max_seq_length - len(ele.input_ids)), value=0)
                    for ele in processed_batch
                ]
            )

        if not return_audio_in_tokens:
            audio_in_ids = None
            audio_in_ids_start = None

        # Apply audio_num_codebooks limit if specified
        if audio_num_codebooks is not None:
            if audio_in_ids is not None:
                audio_in_ids = audio_in_ids[:audio_num_codebooks]
            if audio_out_ids is not None:
                audio_out_ids = audio_out_ids[:audio_num_codebooks]
            if label_audio_ids is not None:
                label_audio_ids = label_audio_ids[:audio_num_codebooks]

        return HiggsAudioBatchInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids,
        )

    def batch_decode(
        self,
        decoder_text_ids: "torch.Tensor",
        decoder_audio_ids: list["torch.Tensor"],
        prompt_token_length: int,
    ) -> list["torch.Tensor"]:
        raise NotImplementedError("Higgs Audio currently only supports single sample generation")

    def get_prompt_len(
        self,
        model_inputs_ids: "torch.Tensor",
    ) -> int:
        """Utility function to get the input prompt length."""
        return model_inputs_ids.shape[1]

    # Copied from transformers.models.dia.processing_dia.DiaProcessor.save_audio with Dia->HiggsAudio
    def save_audio(
        self,
        audio: AudioInput,
        saving_path: Union[str, Path, list[Union[str, Path]]],
        **kwargs: Unpack[HiggsAudioProcessorKwargs],
    ):
        # TODO: @eustlb, this should be in AudioProcessor
        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files.")

        # ensure correct audio input
        audio = make_list_of_audio(audio)

        # ensure correct saving path
        if isinstance(saving_path, (str, Path)):
            saving_path = [saving_path]
        elif not (isinstance(saving_path, (list, tuple)) and all(isinstance(p, (str, Path)) for p in saving_path)):
            raise ValueError("Invalid input path. Please provide a string, or a list of strings")

        if len(audio) != len(saving_path):
            raise ValueError("The number of audio and saving paths must be the same")

        output_kwargs = self._merge_kwargs(
            HiggsAudioProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        sampling_rate = audio_kwargs["sampling_rate"]

        for audio_value, p in zip(audio, saving_path):
            if isinstance(audio_value, torch.Tensor):
                audio_value = audio_value.cpu().float().numpy()
            sf.write(p, audio_value, sampling_rate)

    def decode(
        self,
        decoder_text_ids: "torch.Tensor",
        decoder_audio_ids: list["torch.Tensor"],
        prompt_token_length: int,
        **kwargs: Unpack[HiggsAudioProcessorKwargs],
    ) -> HiggsAudioResponse:
        """
        Decodes a single sequence of audio codebooks into the respective audio waveform via the
        `audio_tokenizer`.
        """
        output_kwargs = self._merge_kwargs(
            HiggsAudioProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        audio_codebook_size = audio_kwargs["audio_codebook_size"]

        if len(decoder_audio_ids) > 0:
            wv_list = []
            for output_audio in decoder_audio_ids:
                vq_code = revert_delay_pattern(output_audio).clip(0, audio_codebook_size - 1)[:, 1:-1]
                wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0][0, 0]
                wv_list.append(wv_numpy.detach().cpu().numpy())
            wv_numpy = np.concatenate(wv_list)
        else:
            wv_numpy = None

        # We only support one request at a time now
        generated_text_tokens = decoder_text_ids.cpu().numpy()[prompt_token_length:]
        generated_text = self.tokenizer.decode(generated_text_tokens)

        return HiggsAudioResponse(
            audio=wv_numpy,
            generated_text=generated_text,
            sampling_rate=self.audio_tokenizer.sampling_rate,
        )

    @property
    def default_chat_template(self):
        """
        ChatML template that handles multimodal messages with text and audio content.

        For each message:
        - Formats with role headers using <|start_header_id|>role<|end_header_id|>
        - Handles text content directly
        - Replaces audio content with appropriate tokens:
          - User/system audio: <|audio_bos|><|AUDIO|><|audio_eos|>
          - Assistant audio: <|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>
        - Adds recipient information for assistant messages when present
        - Ends messages with <|eot_id|> or <|eom_id|> for consecutive assistant turns

        Example:
        ```python
        messages = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Hello'},
                {'type': 'audio', 'audio_url': 'audio.wav'}
            ]},
            {'role': 'assistant', 'content': 'Hi there!', 'recipient': 'user'}
        ]
        ```
        """
        return (
            "{% set loop_total = messages|length %}"
            "{% for message in messages %}"
            "{% if loop.first %}"
            "<|begin_of_text|>"
            "{% endif %}"
            "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
            # Handle recipient for assistant messages
            "{% if message.get('recipient') and message['role'] == 'assistant' %}"
            "{{ message['recipient'] }}<|recipient|>"
            "{% endif %}"
            # Process content
            "{% if message['content'] is string %}"
            "{{ message['content'] }}"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'text' %}"
            "{{ content['text'] }}"
            "{% elif content['type'] == 'audio' %}"
            "{% if message['role'] in ['user', 'system'] %}"
            "<|audio_bos|><|AUDIO|><|audio_eos|>"
            "{% elif message['role'] == 'assistant' %}"
            "<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>"
            "{% endif %}"
            "{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            # Add appropriate ending token
            "{% set next_idx = loop.index %}"
            "{% if message['role'] == 'assistant' and next_idx < loop_total and messages[next_idx]['role'] == 'assistant' %}"
            "<|eom_id|>"
            "{% else %}"
            "<|eot_id|>"
            "{% endif %}"
            "{% endfor %}"
            # Add generation prompt if needed
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{% endif %}"
        )


__all__ = ["HiggsAudioProcessor"]
