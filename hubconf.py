dependencies = ['torch', 'tqdm', 'boto3', 'requests', 'regex']

from hubconfs.bert_hubconf import (
    bertTokenizer,
    bertModel,
    bertForNextSentencePrediction,
    bertForPreTraining,
    bertForMaskedLM,
    bertForSequenceClassification,
    bertForMultipleChoice,
    bertForQuestionAnswering,
    bertForTokenClassification
)
from hubconfs.gpt_hubconf import (
    openAIGPTTokenizer,
    openAIGPTModel,
    openAIGPTLMHeadModel,
    openAIGPTDoubleHeadsModel
)
from hubconfs.gpt2_hubconf import (
    gpt2Tokenizer,
    gpt2Model,
    gpt2LMHeadModel,
    gpt2DoubleHeadsModel
)
from hubconfs.transformer_xl_hubconf import (
    transformerXLTokenizer,
    transformerXLModel,
    transformerXLLMHeadModel
)
