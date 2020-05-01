import numpy as np

from transformers import BertConfig, RobertaConfig
from transformers.modeling_flax_bert import FlaxBertModel
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class FlaxRobertaModel(FlaxBertModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config: BertConfig, state: dict, seed: int = 0, **kwargs):

        super().__init__(config, state, seed, **kwargs)

        if config.pad_token_id is None:
            config.pad_token_id = 1

    @property
    def config(self) -> RobertaConfig:
        return self._config

    def __call__(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        if position_ids is None:
            position_ids = np.arange(
                self.config.pad_token_id + 1, np.atleast_2d(input_ids).shape[-1] + self.config.pad_token_id + 1
            )

        return super().__call__(input_ids, token_type_ids, position_ids, attention_mask)
