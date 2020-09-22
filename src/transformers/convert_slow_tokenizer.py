import json
import os
from typing import Dict, List, Tuple

from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.normalizers import NFKD, Lowercase, Precompiled, Replace, Sequence, StripAccents
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.pre_tokenizers import Sequence as PSequence
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

# from transformers.tokenization_openai import OpenAIGPTTokenizer
from transformers.utils import sentencepiece_model_pb2 as model


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models.
    https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        # Get SentencePiece
        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in vocab.keys():
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges


def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


def get_proto(filename: str):
    m = model.ModelProto()
    m.ParseFromString(open(filename, "rb").read())
    return m


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab_path = "vocab.txt"
        self.original_tokenizer.save_vocabulary(vocab_path)
        tokenizer = Tokenizer(WordPiece(vocab_path, unk_token=str(self.original_tokenizer.unk_token)))

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(self.original_tokenizer.unk_token)) is not None:
            tokenizer.add_special_tokens([str(self.original_tokenizer.unk_token)])
        if tokenizer.token_to_id(str(self.original_tokenizer.sep_token)) is not None:
            tokenizer.add_special_tokens([str(self.original_tokenizer.sep_token)])
        if tokenizer.token_to_id(str(self.original_tokenizer.cls_token)) is not None:
            tokenizer.add_special_tokens([str(self.original_tokenizer.cls_token)])
        if tokenizer.token_to_id(str(self.original_tokenizer.pad_token)) is not None:
            tokenizer.add_special_tokens([str(self.original_tokenizer.pad_token)])
        if tokenizer.token_to_id(str(self.original_tokenizer.mask_token)) is not None:
            tokenizer.add_special_tokens([str(self.original_tokenizer.mask_token)])

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        sep_token_id = tokenizer.token_to_id(str(self.original_tokenizer.sep_token))
        if sep_token_id is None:
            raise TypeError("sep_token not found in the vocabulary")
        cls_token_id = tokenizer.token_to_id(str(self.original_tokenizer.cls_token))
        if cls_token_id is None:
            raise TypeError("cls_token not found in the vocabulary")

        tokenizer.post_processor = processors.BertProcessing(
            (str(self.original_tokenizer.sep_token), sep_token_id),
            (str(self.original_tokenizer.cls_token), cls_token_id),
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class GPT2Converter(Converter):
    def converted(self) -> Tokenizer:
        save_directory = "./"
        vocab_file, merges_file = self.original_tokenizer.save_vocabulary(save_directory)

        tokenizer = Tokenizer(
            BPE(
                vocab_file,
                merges_file,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


class RobertaConverter(Converter):
    def converted(self) -> Tokenizer:
        save_directory = "./"
        ot = self.original_tokenizer
        vocab_file, merges_file = ot.save_vocabulary(save_directory)

        tokenizer = Tokenizer(
            BPE(
                vocab_file,
                merges_file,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(ot.sep_token, ot.sep_token_id),
            cls=(ot.cls_token, ot.cls_token_id),
            add_prefix_space=ot.add_prefix_space,
            trim_offsets=True,  # True by default on Roberta (historical)
        )

        return tokenizer


class SpmConverter(Converter):
    def __init__(self, *args):
        super().__init__(*args)
        self.proto = get_proto(self.original_tokenizer.vocab_file)

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab = self.vocab(proto)
        unk_id = self.unk_id(proto)
        filename = self.original_tokenizer.vocab_file

        if model_type == 1:
            data = {"unk_id": unk_id, "vocab": vocab}

            out_vocab_filename = f"{filename}.json"
            try:
                with open(out_vocab_filename, "w") as f:
                    json.dump(data, f, indent=4)

                tokenizer = Tokenizer(Unigram(out_vocab_filename))
            finally:
                os.remove(out_vocab_filename)
        elif model_type == 2:
            vocab, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            # Open output files and let's extract model information
            out_vocab_filename = f"{filename}.vocab"
            out_merge_filename = f"{filename}.merge"
            try:
                with open(out_vocab_filename, "w") as vocab_f:
                    json.dump(vocab, vocab_f)
                try:
                    with open(out_merge_filename, "w") as merges_f:
                        # Save content
                        merges_f.writelines(map(lambda x: f"{x[0]} {x[1]}{os.linesep}", merges))
                    tokenizer = Tokenizer(
                        BPE(
                            out_vocab_filename,
                            out_merge_filename,
                            unk_token=proto.trainer_spec.unk_piece,
                        )
                    )
                finally:
                    os.remove(out_merge_filename)
            finally:
                os.remove(out_vocab_filename)
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        return Precompiled(precompiled_charsmap)

    def post_processor(self, tokenizer):
        return None

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        tokenizer.normalizer = self.normalizer(self.proto)

        replacement = "▁"
        add_prefix_space = True
        tokenizer.pre_tokenizer = PSequence(
            [
                WhitespaceSplit(),
                Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)
        post_processor = self.post_processor(tokenizer)
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


class AlbertConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        normalizers = [Replace("``", '"'), Replace("''", '"')]
        if not self.original_tokenizer.keep_accents:
            normalizers.append(NFKD())
            normalizers.append(StripAccents())
        if self.original_tokenizer.do_lower_case:
            normalizers.append(Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        normalizers.append(Precompiled(precompiled_charsmap))
        return Sequence(normalizers)

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["[CLS]", "$0", "[SEP]"],
            seq_b=["$1", "[SEP]"],
            special_tokens=[
                ("[CLS]", tokenizer.get_vocab()["[CLS]"]),
                ("[SEP]", tokenizer.get_vocab()["[SEP]"]),
            ],
        )


class CamembertConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>NOTUSED", 0.0),
            ("<pad>", 0.0),
            ("</s>NOTUSED", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces]
        return vocab

    def unk_id(self, proto):
        # See vocab unk position
        return 3

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["<s>", "$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("<s>", tokenizer.get_vocab()["<s>"]),
                ("</s>", tokenizer.get_vocab()["</s>"]),
            ],
        )


class MBartConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [
            ("ar_AR", 0.0),
            ("cs_CZ", 0.0),
            ("de_DE", 0.0),
            ("en_XX", 0.0),
            ("es_XX", 0.0),
            ("et_EE", 0.0),
            ("fi_FI", 0.0),
            ("fr_XX", 0.0),
            ("gu_IN", 0.0),
            ("hi_IN", 0.0),
            ("it_IT", 0.0),
            ("ja_XX", 0.0),
            ("kk_KZ", 0.0),
            ("ko_KR", 0.0),
            ("lt_LT", 0.0),
            ("lv_LV", 0.0),
            ("my_MM", 0.0),
            ("ne_NP", 0.0),
            ("nl_XX", 0.0),
            ("ro_RO", 0.0),
            ("ru_RU", 0.0),
            ("si_LK", 0.0),
            ("tr_TR", 0.0),
            ("vi_VN", 0.0),
            ("zh_CN", 0.0),
        ]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["$0", "</s>", "en_XX"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("en_XX", tokenizer.get_vocab()["en_XX"]),
                ("</s>", tokenizer.get_vocab()["</s>"]),
            ],
        )


class XLMRobertaConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["<s>", "$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("<s>", tokenizer.get_vocab()["<s>"]),
                ("</s>", tokenizer.get_vocab()["</s>"]),
            ],
        )


class XLNetConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        normalizers = [Replace("``", '"'), Replace("''", '"')]
        if not self.original_tokenizer.keep_accents:
            normalizers.append(NFKD())
            normalizers.append(StripAccents())
        if self.original_tokenizer.do_lower_case:
            normalizers.append(Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        normalizers.append(Precompiled(precompiled_charsmap))
        return Sequence(normalizers)

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["$0", "<sep>", "<cls>"],
            seq_b=["$1", "<sep>"],
            special_tokens=[
                ("<sep>", tokenizer.get_vocab()["<sep>"]),
                ("<cls>", tokenizer.get_vocab()["<cls>"]),
            ],
        )


class ReformerConverter(SpmConverter):
    pass


class PegasusConverter(SpmConverter):
    offset = 103

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0),
            (self.original_tokenizer.eos_token, 0),
        ]
        vocab += [(f"unk_{i}", -100) for i in range(2, 2 + self.offset)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id + self.offset

    def post_processor(self, tokenizer):
        eos = self.original_tokenizer.eos_token
        return TemplateProcessing(
            seq_a=["$0", eos],
            seq_b=["$1", eos],
            special_tokens=[
                (eos, tokenizer.get_vocab()[eos]),
            ],
        )


class T5Converter(SpmConverter):
    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("</s>", tokenizer.get_vocab()["</s>"]),
            ],
        )


class OpenAIGPTConverter(Converter):
    def converted(self) -> Tokenizer:
        save_directory = "./"
        # self.original_tokenizer: OpenAIGPTTokenizer = self.original_tokenizer
        vocab_file, merges_file = self.original_tokenizer.save_vocabulary(save_directory)

        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab_file,
                merges_file,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
            )
        )

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        return tokenizer


CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "BertTokenizer": BertConverter,
    "CamembertTokenizer": CamembertConverter,
    "GPT2Tokenizer": GPT2Converter,
    "MBartTokenizer": MBartConverter,
    "OpenAIGPTTokenizer": OpenAIGPTConverter,
    "PegasusTokenizer": PegasusConverter,
    "ReformerTokenizer": ReformerConverter,
    "RobertaTokenizer": RobertaConverter,
    "T5Tokenizer": T5Converter,
    "XLMRobertaTokenizer": XLMRobertaConverter,
    "XLNetTokenizer": XLNetConverter,
}


def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    converter_class = CONVERTERS[transformer_tokenizer.__class__.__name__]
    return converter_class(transformer_tokenizer).converted()
