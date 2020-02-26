from enum import Enum

from transformers.pipelines import Pipeline, PipelineDataFormat, pipeline


class ModelType(str, Enum):
    BERT = "bert"
    GPT = "gpt"
    GPT2 = "gpt2"
    TRANSFORMER_XL = "transfo_xl"
    XLNET = "xlnet"
    XLM = "xlm"


class PipelineTask(str, Enum):
    FEATURE_EXTRACTION = "feature-extraction"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    NER = "ner"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"


class SupportedFormat(str, Enum):
    INFER = "infer"
    JSON = "json"
    CSV = "csv"
    PIPE = "pipe"
