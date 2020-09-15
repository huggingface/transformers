import logging
from typing import List, Optional

from datasets import Dataset, Split, load_dataset

from utils_ner import TokenClassificationTask


logger = logging.getLogger(__name__)


class NER(TokenClassificationTask):
    def __init__(self):
        super().__init__(source_column="tokens", target_column="labels")

    def get_dataset(self, split: Split) -> Dataset:
        return load_dataset("germeval_14", split=split)

    def get_labels(self, path: Optional[str] = None) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return [
                "B-LOC",
                "B-LOCderiv",
                "B-LOCpart",
                "B-ORG",
                "B-ORGderiv",
                "B-ORGpart",
                "B-OTH",
                "B-OTHderiv",
                "B-OTHpart",
                "B-PER",
                "B-PERderiv",
                "B-PERpart",
                "I-LOC",
                "I-LOCderiv",
                "I-LOCpart",
                "I-ORG",
                "I-ORGderiv",
                "I-ORGpart",
                "I-OTH",
                "I-OTHderiv",
                "I-OTHpart",
                "I-PER",
                "I-PERderiv",
                "I-PERpart",
                "O",
            ]


class Chunk(TokenClassificationTask):
    def __init__(self):
        super().__init__(source_column="word", target_column="chunk")

    def get_dataset(self, split: Split) -> Dataset:
        return load_dataset("./conll2003.py", split=split)

    def get_labels(self, path: Optional[str] = None) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return [
                "O",
                "B-ADVP",
                "B-INTJ",
                "B-LST",
                "B-PRT",
                "B-NP",
                "B-SBAR",
                "B-VP",
                "B-ADJP",
                "B-CONJP",
                "B-PP",
                "I-ADVP",
                "I-INTJ",
                "I-LST",
                "I-PRT",
                "I-NP",
                "I-SBAR",
                "I-VP",
                "I-ADJP",
                "I-CONJP",
                "I-PP",
            ]


class POS(TokenClassificationTask):
    def __init__(self):
        super().__init__(source_column="form", target_column="upos")

    def get_dataset(self, split: Split) -> Dataset:
        return load_dataset("./ud_english_ewt.py", split=split)

    def get_labels(self, path: Optional[str] = None) -> List[str]:
        if path:
            with open(path, "r") as f:
                return f.read().splitlines()
        else:
            return [
                "ADJ",
                "ADP",
                "ADV",
                "AUX",
                "CCONJ",
                "DET",
                "INTJ",
                "NOUN",
                "NUM",
                "PART",
                "PRON",
                "PROPN",
                "PUNCT",
                "SCONJ",
                "SYM",
                "VERB",
                "X",
            ]
