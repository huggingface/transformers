from dataclasses import dataclass, field

from transformers import TFTrainingArguments, add_start_docstrings


@dataclass
@add_start_docstrings(TFTrainingArguments.__doc__)
class TFSeq2SeqTrainingArguments(TFTrainingArguments):
    """
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
