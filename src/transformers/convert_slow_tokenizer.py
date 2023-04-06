# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
Utilities to convert slow tokenizers in their fast tokenizers counterparts.

All the conversions are grouped here to gather SentencePiece dependencies outside of the fast tokenizers files and
allow to make our dependency on SentencePiece optional.
"""

import warnings
from typing import Dict, List, Tuple

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece

from .utils import requires_backends


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        requires_backends(self, "sentencepiece")
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        if vocab_scores is not None:
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            vocab_scores, reverse = vocab, False

        # Merges
        merges = []
        for piece_l in vocab.keys():
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_score = vocab_scores.get(merge, None)
                if piece_score:
                    merges += [(piece_l, piece_r, piece_score)]
        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]
        return vocab, merges


def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

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

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class SplinterConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

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

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        question = str(self.original_tokenizer.question_token)
        dot = "."
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        question_token_id = self.original_tokenizer.question_token_id
        dot_token_id = self.original_tokenizer.convert_tokens_to_ids(".")

        if self.original_tokenizer.padding_side == "right":
            pair = f"{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1"
        else:
            pair = f"{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1"

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=pair,
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
                (question, question_token_id),
                (dot, dot_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class FunnelConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

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

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:2 $A:0 {sep}:0",  # token_type_id is 2 for Funnel transformer
            pair=f"{cls}:2 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class MPNetConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

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

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 {sep}:0 $B:1 {sep}:1",  # MPNet uses two [SEP] tokens
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class OpenAIGPTConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        return tokenizer


class GPT2Converter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        if self.original_tokenizer.add_bos_token:
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",
                pair=f"{bos}:0 $A:0 $B:1",
                special_tokens=[
                    (bos, bos_token_id),
                ],
            )
        else:
            # XXX trim_offsets=False actually means this post_processor doesn't
            # really do anything.
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer


class HerbertConverter(Converter):
    def converted(self) -> Tokenizer:
        tokenizer_info_str = "#version:"
        token_suffix = "</w>"

        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        if tokenizer_info_str in merges[0][0]:
            merges = merges[1:]

        tokenizer = Tokenizer(
            BPE(
                vocab,
                merges,
                dropout=None,
                unk_token=self.original_tokenizer.unk_token,
                end_of_word_suffix=token_suffix,
            )
        )

        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix=token_suffix)
        tokenizer.post_processor = processors.BertProcessing(
            sep=(self.original_tokenizer.sep_token, self.original_tokenizer.sep_token_id),
            cls=(self.original_tokenizer.cls_token, self.original_tokenizer.cls_token_id),
        )

        return tokenizer


class RobertaConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
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


class RoFormerConverter(Converter):
    def converted(self) -> Tokenizer:
        from .models.roformer.tokenization_utils import JiebaPreTokenizer

        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class DebertaConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )

        return tokenizer


class SpmConverter(Converter):
    def __init__(self, *args):
        requires_backends(self, "protobuf")

        super().__init__(*args)

        from .utils import sentencepiece_model_pb2 as model_pb2

        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        unk_id = self.unk_id(proto)

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if not precompiled_charsmap:
            return normalizers.Sequence([normalizers.Replace(Regex(" {2,}"), " ")])
        else:
            return normalizers.Sequence(
                [normalizers.Precompiled(precompiled_charsmap), normalizers.Replace(Regex(" {2,}"), " ")]
            )

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    def post_processor(self):
        return None

    def decoder(self, replacement, add_prefix_space):
        return decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
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
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class BarthezConverter(SpmConverter):
    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class CamembertConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>NOTUSED", 0.0),
            ("<pad>", 0.0),
            ("</s>NOTUSED", 0.0),
            ("<unk>", 0.0),
            ("<unk>NOTUSED", -100),
        ]
        # We down-grade the original SentencePiece by -100 to avoid using it and use our added token instead
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[1:]]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        # See vocab unk position
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class DebertaV2Converter(SpmConverter):
    def pre_tokenizer(self, replacement, add_prefix_space):
        list_pretokenizers = []
        if self.original_tokenizer.split_by_punct:
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior="isolated"))
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space))
        return pre_tokenizers.Sequence(list_pretokenizers)

    def normalizer(self, proto):
        list_normalizers = []
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        list_normalizers.append(normalizers.Strip())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))

        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
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
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="$A </s> en_XX",
            pair="$A $B </s> en_XX",
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class MBart50Converter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # fmt: off
        vocab += [("ar_AR", 0.0), ("cs_CZ", 0.0), ("de_DE", 0.0), ("en_XX", 0.0), ("es_XX", 0.0), ("et_EE", 0.0), ("fi_FI", 0.0), ("fr_XX", 0.0), ("gu_IN", 0.0), ("hi_IN", 0.0), ("it_IT", 0.0), ("ja_XX", 0.0), ("kk_KZ", 0.0), ("ko_KR", 0.0), ("lt_LT", 0.0), ("lv_LV", 0.0), ("my_MM", 0.0), ("ne_NP", 0.0), ("nl_XX", 0.0), ("ro_RO", 0.0), ("ru_RU", 0.0), ("si_LK", 0.0), ("tr_TR", 0.0), ("vi_VN", 0.0), ("zh_CN", 0.0), ("af_ZA", 0.0), ("az_AZ", 0.0), ("bn_IN", 0.0), ("fa_IR", 0.0), ("he_IL", 0.0), ("hr_HR", 0.0), ("id_ID", 0.0), ("ka_GE", 0.0), ("km_KH", 0.0), ("mk_MK", 0.0), ("ml_IN", 0.0), ("mn_MN", 0.0), ("mr_IN", 0.0), ("pl_PL", 0.0), ("ps_AF", 0.0), ("pt_XX", 0.0), ("sv_SE", 0.0), ("sw_KE", 0.0), ("ta_IN", 0.0), ("te_IN", 0.0), ("th_TH", 0.0), ("tl_XX", 0.0), ("uk_UA", 0.0), ("ur_PK", 0.0), ("xh_ZA", 0.0), ("gl_ES", 0.0), ("sl_SI", 0.0)]
        # fmt: on
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="en_XX $A </s>",
            pair="en_XX $A $B </s>",
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class NllbConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [
            # fmt: off
            ('ace_Arab', 0.0), ('ace_Latn', 0.0), ('acm_Arab', 0.0), ('acq_Arab', 0.0), ('aeb_Arab', 0.0), ('afr_Latn', 0.0), ('ajp_Arab', 0.0), ('aka_Latn', 0.0), ('amh_Ethi', 0.0), ('apc_Arab', 0.0), ('arb_Arab', 0.0), ('ars_Arab', 0.0), ('ary_Arab', 0.0), ('arz_Arab', 0.0), ('asm_Beng', 0.0), ('ast_Latn', 0.0), ('awa_Deva', 0.0), ('ayr_Latn', 0.0), ('azb_Arab', 0.0), ('azj_Latn', 0.0), ('bak_Cyrl', 0.0), ('bam_Latn', 0.0), ('ban_Latn', 0.0), ('bel_Cyrl', 0.0), ('bem_Latn', 0.0), ('ben_Beng', 0.0), ('bho_Deva', 0.0), ('bjn_Arab', 0.0), ('bjn_Latn', 0.0), ('bod_Tibt', 0.0), ('bos_Latn', 0.0), ('bug_Latn', 0.0), ('bul_Cyrl', 0.0), ('cat_Latn', 0.0), ('ceb_Latn', 0.0), ('ces_Latn', 0.0), ('cjk_Latn', 0.0), ('ckb_Arab', 0.0), ('crh_Latn', 0.0), ('cym_Latn', 0.0), ('dan_Latn', 0.0), ('deu_Latn', 0.0), ('dik_Latn', 0.0), ('dyu_Latn', 0.0), ('dzo_Tibt', 0.0), ('ell_Grek', 0.0), ('eng_Latn', 0.0), ('epo_Latn', 0.0), ('est_Latn', 0.0), ('eus_Latn', 0.0), ('ewe_Latn', 0.0), ('fao_Latn', 0.0), ('pes_Arab', 0.0), ('fij_Latn', 0.0), ('fin_Latn', 0.0), ('fon_Latn', 0.0), ('fra_Latn', 0.0), ('fur_Latn', 0.0), ('fuv_Latn', 0.0), ('gla_Latn', 0.0), ('gle_Latn', 0.0), ('glg_Latn', 0.0), ('grn_Latn', 0.0), ('guj_Gujr', 0.0), ('hat_Latn', 0.0), ('hau_Latn', 0.0), ('heb_Hebr', 0.0), ('hin_Deva', 0.0), ('hne_Deva', 0.0), ('hrv_Latn', 0.0), ('hun_Latn', 0.0), ('hye_Armn', 0.0), ('ibo_Latn', 0.0), ('ilo_Latn', 0.0), ('ind_Latn', 0.0), ('isl_Latn', 0.0), ('ita_Latn', 0.0), ('jav_Latn', 0.0), ('jpn_Jpan', 0.0), ('kab_Latn', 0.0), ('kac_Latn', 0.0), ('kam_Latn', 0.0), ('kan_Knda', 0.0), ('kas_Arab', 0.0), ('kas_Deva', 0.0), ('kat_Geor', 0.0), ('knc_Arab', 0.0), ('knc_Latn', 0.0), ('kaz_Cyrl', 0.0), ('kbp_Latn', 0.0), ('kea_Latn', 0.0), ('khm_Khmr', 0.0), ('kik_Latn', 0.0), ('kin_Latn', 0.0), ('kir_Cyrl', 0.0), ('kmb_Latn', 0.0), ('kon_Latn', 0.0), ('kor_Hang', 0.0), ('kmr_Latn', 0.0), ('lao_Laoo', 0.0), ('lvs_Latn', 0.0), ('lij_Latn', 0.0), ('lim_Latn', 0.0), ('lin_Latn', 0.0), ('lit_Latn', 0.0), ('lmo_Latn', 0.0), ('ltg_Latn', 0.0), ('ltz_Latn', 0.0), ('lua_Latn', 0.0), ('lug_Latn', 0.0), ('luo_Latn', 0.0), ('lus_Latn', 0.0), ('mag_Deva', 0.0), ('mai_Deva', 0.0), ('mal_Mlym', 0.0), ('mar_Deva', 0.0), ('min_Latn', 0.0), ('mkd_Cyrl', 0.0), ('plt_Latn', 0.0), ('mlt_Latn', 0.0), ('mni_Beng', 0.0), ('khk_Cyrl', 0.0), ('mos_Latn', 0.0), ('mri_Latn', 0.0), ('zsm_Latn', 0.0), ('mya_Mymr', 0.0), ('nld_Latn', 0.0), ('nno_Latn', 0.0), ('nob_Latn', 0.0), ('npi_Deva', 0.0), ('nso_Latn', 0.0), ('nus_Latn', 0.0), ('nya_Latn', 0.0), ('oci_Latn', 0.0), ('gaz_Latn', 0.0), ('ory_Orya', 0.0), ('pag_Latn', 0.0), ('pan_Guru', 0.0), ('pap_Latn', 0.0), ('pol_Latn', 0.0), ('por_Latn', 0.0), ('prs_Arab', 0.0), ('pbt_Arab', 0.0), ('quy_Latn', 0.0), ('ron_Latn', 0.0), ('run_Latn', 0.0), ('rus_Cyrl', 0.0), ('sag_Latn', 0.0), ('san_Deva', 0.0), ('sat_Beng', 0.0), ('scn_Latn', 0.0), ('shn_Mymr', 0.0), ('sin_Sinh', 0.0), ('slk_Latn', 0.0), ('slv_Latn', 0.0), ('smo_Latn', 0.0), ('sna_Latn', 0.0), ('snd_Arab', 0.0), ('som_Latn', 0.0), ('sot_Latn', 0.0), ('spa_Latn', 0.0), ('als_Latn', 0.0), ('srd_Latn', 0.0), ('srp_Cyrl', 0.0), ('ssw_Latn', 0.0), ('sun_Latn', 0.0), ('swe_Latn', 0.0), ('swh_Latn', 0.0), ('szl_Latn', 0.0), ('tam_Taml', 0.0), ('tat_Cyrl', 0.0), ('tel_Telu', 0.0), ('tgk_Cyrl', 0.0), ('tgl_Latn', 0.0), ('tha_Thai', 0.0), ('tir_Ethi', 0.0), ('taq_Latn', 0.0), ('taq_Tfng', 0.0), ('tpi_Latn', 0.0), ('tsn_Latn', 0.0), ('tso_Latn', 0.0), ('tuk_Latn', 0.0), ('tum_Latn', 0.0), ('tur_Latn', 0.0), ('twi_Latn', 0.0), ('tzm_Tfng', 0.0), ('uig_Arab', 0.0), ('ukr_Cyrl', 0.0), ('umb_Latn', 0.0), ('urd_Arab', 0.0), ('uzn_Latn', 0.0), ('vec_Latn', 0.0), ('vie_Latn', 0.0), ('war_Latn', 0.0), ('wol_Latn', 0.0), ('xho_Latn', 0.0), ('ydd_Hebr', 0.0), ('yor_Latn', 0.0), ('yue_Hant', 0.0), ('zho_Hans', 0.0), ('zho_Hant', 0.0), ('zul_Latn', 0.0)
            # fmt: on
        ]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="eng_Latn $A </s>",
            pair="eng_Latn $A $B </s>",
            special_tokens=[
                ("eng_Latn", self.original_tokenizer.convert_tokens_to_ids("eng_Latn")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
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
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class XLNetConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="$A:0 <sep>:0 <cls>:2",
            pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
            special_tokens=[
                ("<sep>", self.original_tokenizer.convert_tokens_to_ids("<sep>")),
                ("<cls>", self.original_tokenizer.convert_tokens_to_ids("<cls>")),
            ],
        )


class ReformerConverter(SpmConverter):
    pass


class RemBertConverter(SpmConverter):
    # Inspired from AlbertConverter
    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class BertGenerationConverter(SpmConverter):
    pass


class PegasusConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
        ]

        if self.original_tokenizer.mask_token_sent is not None:
            vocab += [(self.original_tokenizer.mask_token_sent, 0.0)]

        if (
            self.original_tokenizer.mask_token is not None
            and self.original_tokenizer.mask_token_id < self.original_tokenizer.offset
        ):
            vocab += [(self.original_tokenizer.mask_token, 0.0)]

        vocab += [(f"<unk_{i}>", -100.0) for i in range(2, self.original_tokenizer.offset)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id + self.original_tokenizer.offset

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )

    def post_processor(self):
        eos = self.original_tokenizer.eos_token
        special_tokens = [
            (eos, self.original_tokenizer.eos_token_id),
        ]
        return processors.TemplateProcessing(single=["$A", eos], pair=["$A", "$B", eos], special_tokens=special_tokens)


class T5Converter(SpmConverter):
    def vocab(self, proto):
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        vocab += [(f"<extra_id_{i}>", 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class WhisperConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        prefix_token_ids = self.original_tokenizer.prefix_tokens
        prefixes = self.original_tokenizer.convert_ids_to_tokens(prefix_token_ids)
        eos = self.original_tokenizer.eos_token
        eos_token_id = self.original_tokenizer.eos_token_id
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

        return tokenizer


class BigBirdConverter(SpmConverter):
    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class CLIPConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                    behavior="removed",
                    invert=True,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()

        # Hack to have a ByteLevel and TemplaceProcessor
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(self.original_tokenizer.eos_token, self.original_tokenizer.eos_token_id),
            cls=(self.original_tokenizer.bos_token, self.original_tokenizer.bos_token_id),
            add_prefix_space=False,
            trim_offsets=False,
        )
        return tokenizer


class LayoutLMv2Converter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = True
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

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class BlenderbotConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A:0 {ot.eos_token}:0",
            special_tokens=[
                (ot.eos_token, ot.eos_token_id),
            ],
        )

        return tokenizer


class XGLMConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        # fmt: off
        vocab += [("<madeupword0>", 0.0), ("<madeupword1>", 0.0), ("<madeupword2>", 0.0), ("<madeupword3>", 0.0), ("<madeupword4>", 0.0), ("<madeupword5>", 0.0), ("<madeupword6>", 0.0)]
        # fmt: on
        return vocab

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self):
        return processors.TemplateProcessing(
            single="</s> $A",
            pair="</s> $A </s> </s> $B",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class LlamaConverter(SpmConverter):
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(content=" ", left=1),
            ]
        )

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        if model_type == 1:
            raise RuntimeError("Llama is supposed to be a BPE model!")
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens(
                [
                    AddedToken("<unk>", normalized=True),
                    AddedToken("<s>", normalized=True),
                    AddedToken("</s>", normalized=True),
                ]
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        return normalizers.Sequence(
            [
                normalizers.Prepend(prepend="▁"),
                normalizers.Replace(pattern=" ", content="▁"),
            ]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A",
            pair="<s> $A $B",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
            ],
        )


class MarkupLMConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                unk_token=self.original_tokenizer.unk_token,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls} $A {sep}",
            pair=f"{cls} $A {sep} $B {sep}",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )

        return tokenizer


SLOW_TO_FAST_CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "BartTokenizer": RobertaConverter,
    "BarthezTokenizer": BarthezConverter,
    "BertTokenizer": BertConverter,
    "BigBirdTokenizer": BigBirdConverter,
    "BlenderbotTokenizer": BlenderbotConverter,
    "CamembertTokenizer": CamembertConverter,
    "CLIPTokenizer": CLIPConverter,
    "CodeGenTokenizer": GPT2Converter,
    "ConvBertTokenizer": BertConverter,
    "DebertaTokenizer": DebertaConverter,
    "DebertaV2Tokenizer": DebertaV2Converter,
    "DistilBertTokenizer": BertConverter,
    "DPRReaderTokenizer": BertConverter,
    "DPRQuestionEncoderTokenizer": BertConverter,
    "DPRContextEncoderTokenizer": BertConverter,
    "ElectraTokenizer": BertConverter,
    "FNetTokenizer": AlbertConverter,
    "FunnelTokenizer": FunnelConverter,
    "GPT2Tokenizer": GPT2Converter,
    "HerbertTokenizer": HerbertConverter,
    "LayoutLMTokenizer": BertConverter,
    "LayoutLMv2Tokenizer": BertConverter,
    "LayoutLMv3Tokenizer": RobertaConverter,
    "LayoutXLMTokenizer": XLMRobertaConverter,
    "LongformerTokenizer": RobertaConverter,
    "LEDTokenizer": RobertaConverter,
    "LxmertTokenizer": BertConverter,
    "MarkupLMTokenizer": MarkupLMConverter,
    "MBartTokenizer": MBartConverter,
    "MBart50Tokenizer": MBart50Converter,
    "MPNetTokenizer": MPNetConverter,
    "MobileBertTokenizer": BertConverter,
    "MvpTokenizer": RobertaConverter,
    "NllbTokenizer": NllbConverter,
    "OpenAIGPTTokenizer": OpenAIGPTConverter,
    "PegasusTokenizer": PegasusConverter,
    "RealmTokenizer": BertConverter,
    "ReformerTokenizer": ReformerConverter,
    "RemBertTokenizer": RemBertConverter,
    "RetriBertTokenizer": BertConverter,
    "RobertaTokenizer": RobertaConverter,
    "RoFormerTokenizer": RoFormerConverter,
    "SqueezeBertTokenizer": BertConverter,
    "T5Tokenizer": T5Converter,
    "WhisperTokenizer": WhisperConverter,
    "XLMRobertaTokenizer": XLMRobertaConverter,
    "XLNetTokenizer": XLNetConverter,
    "SplinterTokenizer": SplinterConverter,
    "XGLMTokenizer": XGLMConverter,
    "LlamaTokenizer": LlamaConverter,
}


def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__

    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance."
            " No converter was found. Currently available slow->fast convertors:"
            f" {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    return converter_class(transformer_tokenizer).converted()
