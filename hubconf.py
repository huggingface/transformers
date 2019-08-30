dependencies = ['torch', 'tqdm', 'boto3', 'requests', 'regex', 'sentencepiece', 'sacremoses']

from hubconfs.automodels_hubconf import (
    config,
    model,
    modelForQuestionAnswering,
    modelForSequenceClassification,
    modelWithLMHead,
    tokenizer,
)
