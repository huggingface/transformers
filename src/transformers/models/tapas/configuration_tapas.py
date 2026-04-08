# Copyright 2020 Google Research and The HuggingFace Inc. team.
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
TAPAS configuration. Based on the BERT configuration with added parameters.

Hyperparameters are taken from run_task_main.py and hparam_utils.py of the original implementation. URLS:

- https://github.com/google-research/tapas/blob/master/tapas/run_task_main.py
- https://github.com/google-research/tapas/blob/master/tapas/utils/hparam_utils.py

"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/tapas-base-finetuned-sqa")
@strict
class TapasConfig(PreTrainedConfig):
    r"""
    type_vocab_sizes (`list[int]`, *optional*, defaults to `[3, 256, 256, 2, 256, 256, 10]`):
        The vocabulary sizes of the `token_type_ids` passed when calling [`TapasModel`].
    positive_label_weight (`float`, *optional*, defaults to 10.0):
        Weight for positive labels.
    num_aggregation_labels (`int`, *optional*, defaults to 0):
        The number of aggregation operators to predict.
    aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
        Importance weight for the aggregation loss.
    use_answer_as_supervision (`bool`, *optional*):
        Whether to use the answer as the only supervision for aggregation examples.
    answer_loss_importance (`float`, *optional*, defaults to 1.0):
        Importance weight for the regression loss.
    use_normalized_answer_loss (`bool`, *optional*, defaults to `False`):
        Whether to normalize the answer loss by the maximum of the predicted and expected value.
    huber_loss_delta (`float`, *optional*):
        Delta parameter used to calculate the regression loss.
    temperature (`float`, *optional*, defaults to 1.0):
        Value used to control (OR change) the skewness of cell logits probabilities.
    aggregation_temperature (`float`, *optional*, defaults to 1.0):
        Scales aggregation logits to control the skewness of probabilities.
    use_gumbel_for_cells (`bool`, *optional*, defaults to `False`):
        Whether to apply Gumbel-Softmax to cell selection.
    use_gumbel_for_aggregation (`bool`, *optional*, defaults to `False`):
        Whether to apply Gumbel-Softmax to aggregation selection.
    average_approximation_function (`string`, *optional*, defaults to `"ratio"`):
        Method to calculate the expected average of cells in the weak supervision case. One of `"ratio"`,
        `"first_order"` or `"second_order"`.
    cell_selection_preference (`float`, *optional*):
        Preference for cell selection in ambiguous cases. Only applicable in case of weak supervision for
        aggregation (WTQ, WikiSQL). If the total mass of the aggregation probabilities (excluding the "NONE"
        operator) is higher than this hyperparameter, then aggregation is predicted for an example.
    answer_loss_cutoff (`float`, *optional*):
        Ignore examples with answer loss larger than cutoff.
    max_num_rows (`int`, *optional*, defaults to 64):
        Maximum number of rows.
    max_num_columns (`int`, *optional*, defaults to 32):
        Maximum number of columns.
    average_logits_per_cell (`bool`, *optional*, defaults to `False`):
        Whether to average logits per cell.
    select_one_column (`bool`, *optional*, defaults to `True`):
        Whether to constrain the model to only select cells from a single column.
    allow_empty_column_selection (`bool`, *optional*, defaults to `False`):
        Whether to allow not to select any column.
    init_cell_selection_weights_to_zero (`bool`, *optional*, defaults to `False`):
        Whether to initialize cell selection weights to 0 so that the initial probabilities are 50%.
    reset_position_index_per_cell (`bool`, *optional*, defaults to `True`):
        Whether to restart position indexes at every cell (i.e. use relative position embeddings).
    disable_per_token_loss (`bool`, *optional*, defaults to `False`):
        Whether to disable any (strong or weak) supervision on cells.
    aggregation_labels (`dict[int, label]`, *optional*):
        The aggregation labels used to aggregate the results. For example, the WTQ models have the following
        aggregation labels: `{0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}`
    no_aggregation_label_index (`int`, *optional*):
        If the aggregation labels are defined and one of these labels represents "No aggregation", this should be
        set to its index. For example, the WTQ models have the "NONE" aggregation label at index 0, so that value
        should be set to 0 for these models.

    Example:

    ```python
    >>> from transformers import TapasModel, TapasConfig

    >>> # Initializing a default (SQA) Tapas configuration
    >>> configuration = TapasConfig()
    >>> # Initializing a model from the configuration
    >>> model = TapasModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "tapas"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 1024
    type_vocab_sizes: list[int] | tuple[int, ...] = (3, 256, 256, 2, 256, 256, 10)
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    positive_label_weight: float = 10.0
    num_aggregation_labels: int = 0
    aggregation_loss_weight: float = 1.0
    use_answer_as_supervision: bool | None = None
    answer_loss_importance: float = 1.0
    use_normalized_answer_loss: bool = False
    huber_loss_delta: float | None = None
    temperature: float = 1.0
    aggregation_temperature: float = 1.0
    use_gumbel_for_cells: bool = False
    use_gumbel_for_aggregation: bool = False
    average_approximation_function: str = "ratio"
    cell_selection_preference: float | None = None
    answer_loss_cutoff: float | int | None = None
    max_num_rows: int = 64
    max_num_columns: int = 32
    average_logits_per_cell: bool = False
    select_one_column: bool = True
    allow_empty_column_selection: bool = False
    init_cell_selection_weights_to_zero: bool = False
    reset_position_index_per_cell: bool = True
    disable_per_token_loss: bool = False
    aggregation_labels: dict | None = None
    no_aggregation_label_index: int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.aggregation_labels, dict):
            self.aggregation_labels = {int(k): v for k, v in self.aggregation_labels.items()}
        super().__post_init__(**kwargs)


__all__ = ["TapasConfig"]
