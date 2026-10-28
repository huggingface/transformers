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
"""
Write the Transformers files of Nemotron 3.5 Transcribe for NVIDIA's NeMo export (`examples/speechlm2/to_hf.py`:
`config.json`, `llm_backbone/config.json`, `model.safetensors` and the tokenizer files).

The weights are not converted: the model loads NVIDIA's `model.safetensors` as is (NeMo key layout, mixed bfloat16 and
float32), through the `nemo_speechlm` entry of `src/transformers/conversion_mapping.py`, so that NeMo, vLLM and
Transformers share one checkpoint. This script only writes the small files, from the export's text files and the
safetensors header and metadata (no tensor is read but two constants of the word timestamp head):
- `config.json`: NVIDIA's `config.json` with the keys of the Transformers config appended;
- `tokenizer_config.json`: NVIDIA's file with the processor's padding defaults, nothing else changed (`tokenizer.json`
  and `generation_config.json` are NVIDIA's, unchanged);
- `processor_config.json`, `chat_template.jinja`;
- `alignment_tokenizer/`: the word timestamp head's SentencePiece model, converted to a `tokenizers` tokenizer.

`--output_dir` is an overlay: its files go next to `model.safetensors` (the NeMo files they replace are rewritten from
them). `--check_dir` loads a directory holding both, checks that every tensor is loaded and transcribes a sample.

    python src/transformers/models/nemotron3_5_transcribe/convert_nemotron3_5_transcribe_to_hf.py \
        --nemo_dir /path/to/nemo_export --output_dir /path/to/overlay [--check_dir /path/to/overlay_with_weights]

Converting the alignment tokenizer requires `sentencepiece` and `protobuf`.
"""

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import AddedToken, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from transformers import (
    AutoTokenizer,
    Nemotron3_5TranscribeConfig,
    Nemotron3_5TranscribeForConditionalGeneration,
    Nemotron3_5TranscribeProcessor,
    NemotronAsrStreamingFeatureExtractor,
    PreTrainedTokenizerFast,
)
from transformers.audio_utils import load_audio
from transformers.convert_slow_tokenizer import SpmConverter, generate_merges, import_protobuf


SANITY_CHECK_AUDIO_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
)

# The checkpoint's chat template, which renders string contents only, also takes lists of content parts (what the
# processor builds from audio items): rendered as NeMo's training prompt (`SALMDataset`, the prompt contract of NeMo's
# vLLM plugin), the text parts joined by a space, then one `<|audio|>` per audio part. Without a loop over the content,
# so that vLLM keeps passing string contents, which the plugin requires.
RENDER_CONTENT = """{#- a list of content parts is rendered as NeMo's training prompt (`SALMDataset`, the vLLM plugin's prompt
contract): the text parts joined by a space, then one `<|audio|>` per audio part, separated by a space -#}
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- set text_parts = content | selectattr("type", "in", ["text", "input_text", "output_text"]) | list -%}
        {%- set text = text_parts | map(attribute="text") | join(" ") -%}
        {%- set num_audio = content | selectattr("type", "in", ["audio", "audio_url", "input_audio"]) | list | length -%}
        {%- set locators = (["<|audio|>"] * num_audio) | join(" ") -%}
        {%- if num_audio == 0 -%}
            {{- text_parts | map(attribute="text") | join("\\n") -}}
        {%- elif text.count("<|audio|>") > 0 -%}
            {%- if text.count("<|audio|>") != num_audio -%}
                {{- raise_exception("The text has " ~ text.count("<|audio|>") ~ " <|audio|> locators for " ~ num_audio ~ " audio parts.") -}}
            {%- endif -%}
            {%- set head = text[:(text | length) - (locators | length)] if text.endswith(locators) else text -%}
            {%- if text != locators and text != head.rstrip() ~ " " ~ locators -%}
                {{- raise_exception("An <|audio|> locator in the text must be final and preceded by one space.") -}}
            {%- endif -%}
            {{- text -}}
        {%- elif text -%}
            {{- text ~ " " ~ locators -}}
        {%- else -%}
            {{- locators -}}
        {%- endif -%}
    {%- endif -%}
{%- endmacro -%}
"""

CHAT_TEMPLATE_REPLACEMENTS = [
    (
        '{%- set system_message = messages[0]["content"] %}',
        '{%- set system_message = render_content(messages[0]["content"]) %}',
    ),
    (
        "{%- set content = message.content | string %}",
        "{%- set content = render_content(message.content) %}",
    ),
]

# Defaults of NeMo `ParallelExpertEncoder` for the windows of its online inference and the diarization speaker cache,
# which checkpoint configs usually omit. In encoder frames.
NEMO_ONLINE_INFERENCE_DEFAULTS = {
    "online_inference_length": 500,
    "chunk_left_context": 50,
    "chunk_right_context": 50,
    "diar_fifo_len": 40,
    "diar_spkcache_update_period": 300,
    "diar_spkcache_len": 188,
}
# Defaults of NeMo `TransformerEncoder`, for the options a checkpoint config may omit.
NEMO_ENCODER_DEFAULTS = {
    "subsampling": "feature_stacking",
    "qkv_bias": False,
    "qk_norm": False,
    "pre_block_norm": True,
    "self_attention_model": "rel_pos",
    "rotary_fraction": 1.0,
    "xscaling": False,
    "feat_out": -1,
}


def check_encoder(name: str, encoder_config: dict, **expected):
    """Checks a NeMo `TransformerEncoder` config against the options the HF encoders implement."""
    expected = {
        "subsampling": "feature_stacking",
        "qkv_bias": False,
        "pre_block_norm": True,
        "self_attention_model": "rope",
        "xscaling": False,
        "feat_out": -1,
        **expected,
    }
    for key, value in expected.items():
        # an option missing from the config takes the NeMo default
        actual = encoder_config.get(key, NEMO_ENCODER_DEFAULTS[key])
        if actual != value:
            raise ValueError(f"Unsupported `{name}.{key}={actual}`, expected {value}.")


def read_word_timestamp_metadata(nemo_dir: Path) -> dict:
    """The word timestamp head's decoder config, stored in the safetensors metadata of the export."""
    with safe_open(nemo_dir / "model.safetensors", framework="pt") as weights:
        metadata = weights.metadata()
    if metadata.get("ctc_timestamp_format") != "nemo_ctc_timestamp_artifact_v1":
        raise ValueError(f"Unsupported word timestamp head format `{metadata.get('ctc_timestamp_format')}`.")
    return json.loads(metadata["ctc_timestamp_decoder_config"])


def convert_word_timestamp_config(nemo_dir: Path) -> dict:
    """Checks NeMo's `TransformerCTCDecoder` config against the head the model implements and converts it."""
    decoder = read_word_timestamp_metadata(nemo_dir)
    # the options the head implements, with the `TransformerCTCDecoder` defaults for the ones a config may omit
    expected = {
        "use_transformer": (True, False),
        "qkv_bias": (False, False),
        "qk_norm": (True, False),
        "pre_block_norm": (True, True),
        "self_attention_model": ("rope", "rel_pos"),
        "attn_mode": ("full", "full"),
        "xscaling": (False, False),
        "residual": (True, False),
        "learnable_residual_scale": (False, False),
        "add_blank": (True, True),
        "upsampling_factor": (2, 1),
        "upsampling_mode": ("conv_transpose", "conv_transpose"),
    }
    for name, (value, default) in expected.items():
        if decoder.get(name, default) != value:
            raise ValueError(
                f"Unsupported `ctc_timestamp_decoder_config.{name}={decoder.get(name, default)}`, expected {value}."
            )
    with safe_open(nemo_dir / "model.safetensors", framework="pt") as weights:
        residual_scale = weights.get_tensor("ctc_timestamp.decoder.residual_scale")
    # the head reads the scale from its config, the checkpoint buffer is left unloaded
    if residual_scale.ndim != 0 or residual_scale != torch.tensor(
        decoder["residual_scale"], dtype=residual_scale.dtype
    ):
        raise ValueError(
            f"The `residual_scale` tensor ({residual_scale}) differs from the config value ({decoder['residual_scale']})."
        )
    return {
        "hidden_size": decoder["feat_in"],
        "intermediate_size": int(decoder["feat_in"] * decoder["ff_expansion"]),
        "num_hidden_layers": decoder["n_layers"],
        "num_attention_heads": decoder["n_heads"],
        "max_position_embeddings": decoder["pos_emb_max_len"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": decoder["rope_base"],
            "partial_rotary_factor": decoder["rotary_fraction"],
        },
        # the classes are the alignment tokenizer pieces, then the blank
        "vocab_size": decoder["num_classes"] + 1,
        "blank_token_id": decoder["num_classes"],
        "residual_scale": decoder["residual_scale"],
        "upsampling_factor": decoder["upsampling_factor"],
    }


def convert_config(nemo_dir: Path, audio_token_id: int) -> Nemotron3_5TranscribeConfig:
    nemo_config = json.loads((nemo_dir / "config.json").read_text())
    encoder_config = nemo_config["pe_encoder_config"]
    asr = encoder_config["asr_encoder_cfg"]
    diarization = encoder_config["diarization_model_cfg"]

    expected = {
        "asr_encoder_type": "transformer",
        "asr_normalize_type": "per_feature",
        "diar_normalize_type": "NA",
        "speaker_feature_mode": "thresholded",
        "spk_kernel_scale": 1.0,
    }
    for name, value in expected.items():
        if encoder_config[name] != value:
            raise ValueError(f"Unsupported `pe_encoder_config.{name}={encoder_config[name]}`, expected {value}.")
    diarization_encoder = diarization["encoder"]
    check_encoder("asr_encoder_cfg", asr, qk_norm=True)
    # the diarization audio encoder has no query/key norm and rotates the whole head
    check_encoder("diarization_model_cfg.encoder", diarization_encoder, qk_norm=False, rotary_fraction=1.0)
    if diarization.get("transformer_encoder") is not None:
        raise ValueError("Unsupported `diarization_model_cfg.transformer_encoder`, the Sortformer must have none.")
    if diarization.get("async_streaming", False) or not diarization["sortformer_modules"]["use_learnable_sil_emb"]:
        raise ValueError(
            "Unsupported Sortformer streaming: it must be synchronous with a learnable silence embedding."
        )
    online = {name: encoder_config.get(name, value) for name, value in NEMO_ONLINE_INFERENCE_DEFAULTS.items()}

    asr_config = {
        "num_mel_bins": asr["feat_in"],
        "hidden_size": asr["d_model"],
        "intermediate_size": int(asr["d_model"] * asr["ff_expansion"]),
        "num_hidden_layers": asr["n_layers"],
        "num_attention_heads": asr["n_heads"],
        "subsampling_factor": asr["subsampling_factor"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": asr.get("rope_base", 10000.0),
            "partial_rotary_factor": asr.get("rotary_fraction", 1.0),
        },
    }
    sortformer = diarization["sortformer_modules"]
    diarization_config = {
        "audio_config": {
            "num_mel_bins": diarization_encoder["feat_in"],
            "hidden_size": diarization_encoder["d_model"],
            "intermediate_size": int(diarization_encoder["d_model"] * diarization_encoder["ff_expansion"]),
            "num_hidden_layers": diarization_encoder["n_layers"],
            "num_attention_heads": diarization_encoder["n_heads"],
            "subsampling_factor": diarization_encoder["subsampling_factor"],
        },
        "head_config": {
            "hidden_size": sortformer["tf_d_model"],
            "num_speakers": sortformer["num_spks"],
            "audio_hidden_size": sortformer["fc_d_model"],
            "subsampling_factor": diarization_encoder["subsampling_factor"],
        },
        "streaming_config": {
            "speaker_cache_length": online["diar_spkcache_len"],
            "speaker_cache_silence_frames_per_speaker": sortformer["spkcache_sil_frames_per_spk"],
            "prediction_score_threshold": sortformer["pred_score_threshold"],
            "latest_frames_score_boost": sortformer["scores_boost_latest"],
            "strong_boost_rate": sortformer["strong_boost_rate"],
            "weak_boost_rate": sortformer["weak_boost_rate"],
            "min_positive_scores_rate": sortformer["min_pos_scores_rate"],
            "num_speakers": sortformer["num_spks"],
            "subsampling_factor": diarization_encoder["subsampling_factor"],
        },
        "chunk_length": online["online_inference_length"],
        "chunk_left_context": online["chunk_left_context"],
        "chunk_right_context": online["chunk_right_context"],
        "fifo_length": online["diar_fifo_len"],
        "speaker_cache_update_period": online["diar_spkcache_update_period"],
    }

    text_config = json.loads((nemo_dir / "llm_backbone" / "config.json").read_text())
    # NeMo serializes the infinite upper bound of the Mamba time step as `{"__float__": "Infinity"}`.
    text_config["time_step_limit"] = [
        float(value) if not isinstance(value, dict) else math.inf for value in text_config["time_step_limit"]
    ]
    # the multi-token prediction head is not converted
    text_config["num_nextn_predict_layers"] = 0
    for name in ("architectures", "auto_map", "_name_or_path", "transformers_version"):
        text_config.pop(name, None)

    config = Nemotron3_5TranscribeConfig(
        audio_config={
            "asr_config": asr_config,
            "diarization_config": diarization_config,
            "speaker_activity_threshold": encoder_config["speaker_activity_threshold"],
        },
        text_config=text_config,
        audio_token_id=audio_token_id,
        word_timestamp_config=convert_word_timestamp_config(nemo_dir),
        dtype="bfloat16",
    )
    config.architectures = ["Nemotron3_5TranscribeForConditionalGeneration"]
    return config


def write_config(nemo_dir: Path, output_dir: Path, config: Nemotron3_5TranscribeConfig):
    """
    NeMo and vLLM read the same `config.json`: NVIDIA's file is kept as is (`model_type` and `architectures` are
    already the Transformers ones), the keys of the Transformers config it does not have appended, in NVIDIA's
    formatting for a minimal diff.
    """
    nemo_config = json.loads((nemo_dir / "config.json").read_text())
    for name in ("model_type", "architectures"):
        if nemo_config.get(name) != getattr(config, name):
            raise ValueError(
                f"`{name}` of NVIDIA's `config.json` is {nemo_config.get(name)}, not {getattr(config, name)}."
            )
    config_dict = json.loads(config.to_json_string(use_diff=True))
    nemo_config.update({name: value for name, value in config_dict.items() if name not in nemo_config})
    (output_dir / "config.json").write_text(json.dumps(nemo_config, indent=2, ensure_ascii=False) + "\n")


def convert_chat_template(chat_template: str) -> str:
    """The checkpoint's chat template with `render_content` for the contents of user and system turns."""
    for old, new in CHAT_TEMPLATE_REPLACEMENTS:
        if chat_template.count(old) != 1:
            raise ValueError(
                f"Unexpected chat template: `{old}` found {chat_template.count(old)} times instead of once."
            )
        chat_template = chat_template.replace(old, new)
    position = chat_template.index("{%- set enable_thinking")
    return chat_template[:position] + RENDER_CONTENT + chat_template[position:]


def convert_tokenizer(nemo_dir: Path, output_dir: Path, vocab_size: int):
    """
    Writes NVIDIA's `tokenizer_config.json` with the processor's padding defaults, everything else as NVIDIA wrote it,
    the chat template included (Transformers reads the processor's `chat_template.jinja` first). `tokenizer.json` is
    NVIDIA's: `<|audio|>` is appended after the language model's vocabulary, with no embedding row (the model embeds it
    as token 0 before replacing it with the audio), as NeMo, vLLM and SGLang expect it.
    """
    tokenizer_config = json.loads((nemo_dir / "tokenizer_config.json").read_text())
    chat_template = convert_chat_template(tokenizer_config["chat_template"])
    # older exports store the audio token at the top level, newer ones under `extra_special_tokens`
    audio_token = tokenizer_config.get("audio_token") or tokenizer_config["extra_special_tokens"]["audio_token"]
    # the processor's call defaults, read from the tokenizer's config by `ProcessorMixin`
    tokenizer_config.update(padding=True, padding_side="left")
    # in NVIDIA's formatting, for a minimal diff
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.copy(nemo_dir / "tokenizer.json", Path(tmp_dir) / "tokenizer.json")
        shutil.copy(output_dir / "tokenizer_config.json", Path(tmp_dir) / "tokenizer_config.json")
        tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    past_vocabulary = [token_id for token_id in range(vocab_size, len(tokenizer)) if token_id != audio_token_id]
    if past_vocabulary:
        raise ValueError(f"Only `{audio_token}` may be past the {vocab_size} embeddings, not {past_vocabulary}.")
    return tokenizer, audio_token_id, chat_template


class AlignmentTokenizerConverter(SpmConverter):
    """
    Converts the word timestamp head's SentencePiece model, a BPE model with `add_dummy_prefix`, no extra whitespace
    removal, no byte fallback and 48 user-defined tag pieces (`<ev:…>`, `<spk:…>`, `<sty:…>`).

    SentencePiece adds its dummy prefix before matching the tags (`"<ev:laughter> ok"` -> `▁ <ev:laughter> ▁ok`), while
    a `tokenizers` tokenizer matches added tokens in the raw text, before any normalization. The prefix is therefore
    added by the pre-tokenizer to the first part of the text only (as SentencePiece adds it to the text only, not after
    each tag), which gives SentencePiece's pieces for any text that does not start with a tag, a whitespace or `▁`.
    For any text, encoding `" " + text` gives SentencePiece's pieces of `text`.
    """

    handle_byte_fallback = False

    def __init__(self, model_proto: bytes):
        self.original_tokenizer = None
        self.proto = import_protobuf().ModelProto()
        self.proto.ParseFromString(model_proto)
        trainer, normalizer = self.proto.trainer_spec, self.proto.normalizer_spec
        # `ModelType.BPE`
        if trainer.model_type != 2 or trainer.byte_fallback or not normalizer.add_dummy_prefix:
            raise ValueError("The alignment tokenizer must be a SentencePiece BPE model with a dummy prefix.")
        if normalizer.remove_extra_whitespaces or not normalizer.escape_whitespaces:
            raise ValueError("The alignment tokenizer must keep and escape every whitespace.")
        if any(piece.type == 3 for piece in self.proto.pieces):
            raise ValueError("The alignment tokenizer must have no control pieces.")

    def tokenizer(self, proto):
        vocab_scores = self.vocab(proto)
        vocab = {piece: index for index, (piece, _) in enumerate(vocab_scores)}
        tokenizer = Tokenizer(
            BPE(
                vocab,
                generate_merges(vocab, vocab_scores),
                unk_token=proto.trainer_spec.unk_piece,
                fuse_unk=True,
                byte_fallback=False,
            )
        )
        # the user-defined pieces are matched before the rest of the text, as SentencePiece does
        tokenizer.add_tokens(
            [AddedToken(piece.piece, normalized=False, special=False) for piece in proto.pieces if piece.type == 4]
        )
        return tokenizer

    def normalizer(self, proto):
        # whitespaces are kept, so no strip and no merge of consecutive spaces
        return normalizers.Sequence(
            [normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap), normalizers.Replace(" ", "▁")]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme="first")

    def decoder(self, replacement, add_prefix_space):
        return decoders.Metaspace(replacement=replacement, prepend_scheme="first")


def convert_alignment_tokenizer(nemo_dir: Path) -> PreTrainedTokenizerFast:
    """
    Converts the word timestamp head's SentencePiece model, stored as a uint8 tensor of the checkpoint, and checks it
    against the `sentencepiece` library.
    """
    import sentencepiece

    with safe_open(nemo_dir / "model.safetensors", framework="pt") as weights:
        model_proto = weights.get_tensor("ctc_timestamp.tokenizer.model_proto").numpy().tobytes()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=AlignmentTokenizerConverter(model_proto).converted(),
        unk_token="<unk>",
        add_prefix_space=True,
        clean_up_tokenization_spaces=False,
        # `<unk>` in a text is text, as for SentencePiece
        split_special_tokens=True,
        model_max_length=int(1e30),
    )

    reference = sentencepiece.SentencePieceProcessor(model_proto=model_proto)
    pieces = [reference.id_to_piece(index) for index in range(reference.get_piece_size())]
    # the head's output column `i` is the piece `i`
    vocabulary = read_word_timestamp_metadata(nemo_dir)["vocabulary"]
    if vocabulary != pieces or tokenizer.convert_ids_to_tokens(list(range(len(pieces)))) != pieces:
        raise ValueError("The alignment tokenizer pieces differ from the word timestamp head's vocabulary.")
    if len(tokenizer) != len(pieces):
        raise ValueError(f"Expected {len(pieces)} tokens, got {len(tokenizer)}.")

    tags = tuple(piece.piece for piece in AlignmentTokenizerConverter(model_proto).proto.pieces if piece.type == 4)
    texts = pieces + [f"{tag} ok" for tag in tags] + [f"<spk:0> hi {tag} there" for tag in tags]
    texts += ["<ev:laughter> ok", "<spk:0> hi", "hi<ev:breath>there", "it's 42 dollars", "hello—world", "a  b "]
    texts += ["你好 世界", "안녕하세요 세계", "こんにちは 世界"]
    fixtures = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "nemotron3_5_transcribe"
    for path in sorted(fixtures.glob("expected_results_*.json")):
        for transcript in json.loads(path.read_text())["transcriptions"]:
            texts += [transcript, *transcript.split()]
    mismatches = []
    for text in texts:
        expected = reference.encode(text)
        if tokenizer.encode(" " + text, add_special_tokens=False) != expected:
            mismatches.append((" " + text, expected))
        if not text[:1].isspace() and not text.startswith(("▁", *tags)):
            if tokenizer.encode(text, add_special_tokens=False) != expected:
                mismatches.append((text, expected))
    if mismatches:
        raise ValueError(f"{len(mismatches)} texts are tokenized unlike SentencePiece, e.g. {mismatches[:5]}.")
    print(f"Alignment tokenizer: {len(texts)} texts tokenized as by SentencePiece.")
    return tokenizer


def write_processor(
    nemo_dir: Path, output_dir: Path, tokenizer, alignment_tokenizer, chat_template: str
) -> Nemotron3_5TranscribeProcessor:
    nemo_config = json.loads((nemo_dir / "config.json").read_text())
    preprocessor = nemo_config["perception"]["preprocessor"]
    sampling_rate = preprocessor["sample_rate"]
    # the checkpoint stores the mel filter bank and the STFT window in a reduced precision and computes the features
    # with those rounded constants: the extractor rounds its own float32 constants the same way
    with safe_open(nemo_dir / "model.safetensors", framework="pt") as weights:
        filterbank_dtype = weights.get_slice("perception.preprocessor.featurizer.fb").get_dtype()
    feature_extractor = NemotronAsrStreamingFeatureExtractor(
        feature_size=preprocessor["features"],
        sampling_rate=sampling_rate,
        hop_length=int(preprocessor["window_stride"] * sampling_rate),
        n_fft=preprocessor["n_fft"],
        win_length=int(preprocessor["window_size"] * sampling_rate),
        return_attention_mask=True,
        mel_filters_dtype={"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[filterbank_dtype],
    )
    processor = Nemotron3_5TranscribeProcessor(
        feature_extractor,
        tokenizer,
        alignment_tokenizer,
        chat_template=chat_template,
        # `asr_config.subsampling_factor` of the converted config
        subsampling_factor=nemo_config["pe_encoder_config"]["asr_encoder_cfg"]["subsampling_factor"],
    )
    # the tokenizer files are NVIDIA's, edited by `convert_tokenizer`: only the processor's own files are written
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor.save_pretrained(tmp_dir)
        for name in ("processor_config.json", "chat_template.jinja"):
            shutil.copy(Path(tmp_dir) / name, output_dir / name)
        shutil.copytree(Path(tmp_dir) / "alignment_tokenizer", output_dir / "alignment_tokenizer", dirs_exist_ok=True)
    return processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", required=True, help="Directory of the NeMo `to_hf.py` export.")
    parser.add_argument("--output_dir", required=True, help="Directory of the files to put next to the weights.")
    parser.add_argument(
        "--check_dir",
        default=None,
        help="Directory with the output files and NVIDIA's `model.safetensors`, loaded to check the conversion.",
    )
    args = parser.parse_args()

    nemo_dir, output_dir = Path(args.nemo_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = json.loads((nemo_dir / "llm_backbone" / "config.json").read_text())["vocab_size"]
    tokenizer, audio_token_id, chat_template = convert_tokenizer(nemo_dir, output_dir, vocab_size)
    write_config(nemo_dir, output_dir, convert_config(nemo_dir, audio_token_id))
    processor = write_processor(nemo_dir, output_dir, tokenizer, convert_alignment_tokenizer(nemo_dir), chat_template)

    if args.check_dir is None:
        return
    model, loading_info = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
        args.check_dir, device_map="auto", output_loading_info=True
    )
    audio = load_audio(SANITY_CHECK_AUDIO_URL, sampling_rate=processor.feature_extractor.sampling_rate)
    inputs = processor.apply_transcription_request(audio).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("Greedy transcription:", processor.decode(outputs[:, inputs["input_ids"].shape[1] :])[0])
    # the checkpoint tensors the model does not use are ignored through `_keys_to_ignore_on_load_unexpected`
    for name, keys in loading_info.items():
        if keys:
            raise ValueError(f"{name}: {sorted(keys)}")


if __name__ == "__main__":
    main()
