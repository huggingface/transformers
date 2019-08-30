dependencies = ['torch', 'tqdm', 'boto3', 'requests', 'regex']

from hubconfs.automodels_hubconf import (
    autoConfig,
    autoModel,
    autoModelForQuestionAnswering,
    autoModelForSequenceClassification,
    autoModelWithLMHead,
    autoTokenizer,
)