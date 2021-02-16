# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Speech2TextTransformer."""
import json
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import sentencepiece


try:
    import torchaudio.compliance.kaldi as ta_kaldi
except ImportError:
    raise ImportError("Please install `torchaudio` to enable fbank feature extraction")

from ...tokenization_utils import BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TensorType
from ...utils import logging


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "s2t_librispeech_transformer_small": "https://huggingface.co/valhalla/s2t_librispeech_transformer_small/resolve/main/vocab.json",
    },
    "spm_file": {
        "s2t_librispeech_transformer_small": "https://huggingface.co/valhalla/s2t_librispeech_transformer_small/resolve/main/sentencepiece.bpe.model"
    },
}


class Speech2TextTransformerTokenizer(PreTrainedTokenizer):
    """
    Construct an Speech2TextTransformer tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        spm_file,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        **kwargs,
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        self.sp_model = load_spm(spm_file)

        self.transforms = [UtteranceCMVN()]

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_vocab(self) -> Dict:
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d
        self.sp_model = load_spm(self.spm_file)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_dir = Path(save_directory)
        assert save_dir.is_dir(), f"{save_directory} should be a directory"
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        save_json(self.encoder, vocab_save_path)

        if not spm_save_path.exists():
            copyfile(self.spm_file, spm_save_path)

        return (str(vocab_save_path), str(spm_save_path))

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sample_rate: int,
        num_mel_bins: int = 80,
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        return_attention_mask: bool = True,
        **kwargs
    ):
        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        # make sure input is in list format
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        features = [self._extract_fbank_features(waveform, sample_rate, num_mel_bins) for waveform in raw_speech]
        for transform in self.transforms:
            features = [transform(feature) for feature in features]

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        padded_inputs = self._pad_frames(
            features, padding_strategy, max_length, pad_to_multiple_of, return_attention_mask
        )
        padded_inputs = BatchEncoding(padded_inputs, tensor_type=return_tensors)

        return padded_inputs

    def _extract_fbank_features(
        self,
        waveform,
        sample_rate: int,
        num_mel_bins: int = 80,
    ):
        waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        features = ta_kaldi.fbank(waveform, num_mel_bins=num_mel_bins, sample_frequency=sample_rate)
        return features.numpy()

    def _pad_frames(self, features, padding_strategy, max_length, pad_to_multiple_of, return_attention_mask):
        cur_max_length = max(feature.shape[0] for feature in features)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = cur_max_length

        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            input_features = np.zeros((len(features), max_length, features[0].shape[1]))
            for i, v in enumerate(features):
                input_features[i, : v.shape[0]] = v

            if return_attention_mask:
                attention_mask = np.not_equal(input_features, 0).astype(np.long).tolist()

            input_features = input_features.tolist()
        else:
            features = features
            if return_attention_mask:
                attention_mask = [np.not_equal(feature, 0).astype(np.long).tolist() for feature in features]

            input_features = [feature.tolist() for feature in features]

        encoded_inputs = {"input_features": input_features}
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask

        return encoded_inputs


class UtteranceCMVN:
    """Utterance-level CMVN (cepstral mean and variance normalization)"""

    def __init__(self, norm_means=True, norm_vars=True):
        self.norm_means, self.norm_vars = norm_means, norm_vars

    def __repr__(self):
        return self.__class__.__name__ + f"(norm_means={self.norm_means}, norm_vars={self.norm_vars})"

    def __call__(self, x):
        mean = x.mean(axis=0)
        square_sums = (x ** 2).sum(axis=0)

        if self.norm_means:
            x = np.subtract(x, mean)
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        return x


def load_spm(path: str) -> sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(str(path))
    return spm


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
