from typing import Dict, List, Tuple

from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel

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
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

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
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

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

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab, unk_id))
        elif model_type == 2:
            vocab, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            tokenizer = Tokenizer(
                BPE(
                    vocab,
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
        return normalizers.Precompiled(precompiled_charsmap)

    def post_processor(self):
        return None

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        tokenizer.normalizer = self.normalizer(self.proto)

        replacement = "▁"
        add_prefix_space = True
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)
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
        list_normalizers = [normalizers.Replace("``", '"'), normalizers.Replace("''", '"')]
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
            seq_a=["[CLS]", "$0", "[SEP]"],
            seq_b=["$1", "[SEP]"],
            special_tokens=[
                ("[CLS]", self.original_tokenizer.get_vocab()["[CLS]"]),
                ("[SEP]", self.original_tokenizer.get_vocab()["[SEP]"]),
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
        # We down-grade the original SentencePiece by -100 to avoid using it and use our added token instead
        vocab += [(piece.piece, piece.score if i != 0 else piece.score - 100) for i, piece in enumerate(proto.pieces)]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        # See vocab unk position
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            seq_a=["<s>", "$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("<s>", self.original_tokenizer.get_vocab()["<s>"]),
                ("</s>", self.original_tokenizer.get_vocab()["</s>"]),
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
            seq_a=["$0", "</s>", "en_XX"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("en_XX", self.original_tokenizer.get_vocab()["en_XX"]),
                ("</s>", self.original_tokenizer.get_vocab()["</s>"]),
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
            seq_a=["<s>", "$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("<s>", self.original_tokenizer.get_vocab()["<s>"]),
                ("</s>", self.original_tokenizer.get_vocab()["</s>"]),
            ],
        )


class XLNetConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        list_normalizers = [normalizers.Replace("``", '"'), normalizers.Replace("''", '"')]
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
            seq_a=["$0", "<sep>", "<cls>"],
            seq_b=["$1", "<sep>"],
            special_tokens=[
                ("<sep>", self.original_tokenizer.get_vocab()["<sep>"]),
                ("<cls>", self.original_tokenizer.get_vocab()["<cls>"]),
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

    def post_processor(self):
        eos = self.original_tokenizer.eos_token
        return processors.TemplateProcessing(
            seq_a=["$0", eos],
            seq_b=["$1", eos],
            special_tokens=[
                (eos, self.original_tokenizer.get_vocab()[eos]),
            ],
        )


class T5Converter(SpmConverter):
    def vocab(self, proto):
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        vocab += [("<extra_id_{}>".format(i), 0.0)
                    for i in range(num_extra_ids - 1, -1, -1)]
        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            seq_a=["$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.get_vocab()["</s>"]),
            ],
        )


CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "BertTokenizer": BertConverter,
    "BartTokenizer": RobertaConverter,
    "CamembertTokenizer": CamembertConverter,
    "DistilBertTokenizer": BertConverter,
    "DPRReaderTokenizer": BertConverter,
    "DPRQuestionEncoderTokenizer": BertConverter,
    "DPRContextEncoderTokenizer": BertConverter,
    "GPT2Tokenizer": GPT2Converter,
    "LxmertTokenizer": BertConverter,
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
