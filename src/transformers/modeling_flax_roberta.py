import jax
import numpy as np
import jax.numpy as jnp
from transformers import BertConfig

from transformers import RobertaConfig
from transformers.modeling_flax_bert import FlaxBertModel


class FlaxRobertaModel(FlaxBertModel):
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
        "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
        "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
        "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
        "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
        "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
        "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
    }

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config: BertConfig, state: dict, **kwargs):

        super().__init__(config, state, **kwargs)

        if config.pad_token_id is None:
            config.pad_token_id = 1

    @property
    def config(self) -> RobertaConfig:
        return self._config

    def __call__(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        @jax.jit
        def predict(input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
            if token_type_ids is None:
                token_type_ids = np.ones_like(input_ids)

            if position_ids is None:
                position_ids = np.arange(
                    self.config.pad_token_id + 1,
                    np.atleast_2d(input_ids).shape[-1] + self.config.pad_token_id + 1
                )

            if attention_mask is None:
                attention_mask = np.ones_like(input_ids)

            return self.model(
                jnp.array(input_ids, dtype='i4'),
                jnp.array(token_type_ids, dtype='i4'),
                jnp.array(position_ids, dtype='i4'),
                jnp.array(attention_mask, dtype='i4')
            )

        return predict(input_ids, token_type_ids, position_ids, attention_mask)
