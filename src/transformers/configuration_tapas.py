# coding=utf-8
# Copyright (...) and The HuggingFace Inc. team.
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
""" TAPAS configuration. Inherits from BERT configuration and adds additional hyperparameters."""


from .configuration_bert import BertConfig

TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP = {"tapas-base": "", "tapas-large": ""}  # to be added  # to be added


class TapasConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.TapasModel`.
    It is used to instantiate a TAPAS model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the TAPAS `tapas-base-finetuned-sqa` architecture. Configuration objects
    inherit from :class:`~transformers.PreTrainedConfig` and can be used to control the model outputs.
    Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Hyperparameters additional to BERT are taken from run_task_main.py and hparam_utils.py of the original implementation.
    Original implementation available at https://github.com/google-research/tapas/tree/master.

    Args:
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`List[int]`, `optional`, defaults to [3, 256, 256, 2, 256, 256, 10]):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.TapasModel`.
        positive_weight (:obj:`float`, `optional`, defaults to 10.0):
            Weight for positive labels.
        num_aggregation_labels (:obj:`int`, `optional`, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_importance (:obj:`float`, `optional`, defaults to 1.0):
            Importance weight for the aggregation loss.
        use_answer_as_supervision (:obj:`bool`, `optional`, defaults to :obj:`None`):
            Whether to use the answer as the only supervision for aggregation examples.
        answer_loss_importance (:obj:`float`, `optional`, defaults to 1.0):
            Importance weight for the regression loss.
        use_normalized_answer_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Normalize loss by max of predicted and expected value.
        huber_loss_delta: (:obj:`float`, `optional`, defaults to None):
            Delta parameter used to calculate the regression loss.
        temperature: (:obj:`float`, `optional`, defaults to 1.0):
            Scales cell logits to control the skewness of probabilities.
        agg_temperature: (:obj:`float`, `optional`, defaults to 1.0):
            Scales aggregation logits to control the skewness of probabilities.
        use_gumbel_for_cells: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Applies Gumbel-Softmax to cell selection.
        use_gumbel_for_agg: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Applies Gumbel-Softmax to aggregation selection.
        average_approximation_function: (:obj:`string`, `optional`, defaults to :obj:`"ratio"`):
            Method to calculate expected average of cells in the relaxed case.
        cell_select_pref: (:obj:`float`, `optional`, defaults to None):
            Preference for cell selection in ambiguous cases.
        answer_loss_cutoff: (:obj:`float`, `optional`, defaults to None):
            Ignore examples with answer loss larger than cutoff.
        max_num_rows: (:obj:`int`, `optional`, defaults to 64):
            Maximum number of rows.
        max_num_columns: (:obj:`int`, `optional`, defaults to 32):
            Maximum number of columns.
        average_logits_per_cell: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to average logits per cell.
        select_one_column: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to constrain the model to only select cells from a single column.
        allow_empty_column_selection: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Allow not to select any column.
        init_cell_selection_weights_to_zero: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to initialize cell selection weights to 0 so that the initial probabilities are 50%.
        reset_position_index_per_cell: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Restart position indexes at every cell.
        disable_per_token_loss: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Disable any (strong or weak) supervision on cells.
        span_prediction: (:obj:`string`, `optional`, defaults to :obj:`"none"`):
            Span selection mode to use. Currently only "none" is supported. 

    Example::
        >>> from transformers import TapasModel, TapasConfig
        >>> # Initializing a Tapas configuration
        >>> configuration = TapasConfig()
        >>> # Initializing a model from the configuration
        >>> model = TapasModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "tapas"

    def __init__(
        self,
        max_position_embeddings=1024,
        type_vocab_size=[3, 256, 256, 2, 256, 256, 10],
        positive_weight=10.0,
        num_aggregation_labels=0,
        aggregation_loss_importance=1.0,
        use_answer_as_supervision=None,
        answer_loss_importance=1.0,
        use_normalized_answer_loss=False,
        huber_loss_delta=None,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function="ratio",
        cell_select_pref=None,
        answer_loss_cutoff=None,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=False,
        select_one_column=True,
        allow_empty_column_selection=False,
        init_cell_selection_weights_to_zero=False,
        reset_position_index_per_cell=True,
        disable_per_token_loss=False,
        span_prediction="none",
        **kwargs
    ):

        super().__init__(max_position_embeddings=max_position_embeddings, type_vocab_size=type_vocab_size, **kwargs)

        # Fine-tuning task arguments
        self.positive_weight = positive_weight
        self.num_aggregation_labels = num_aggregation_labels
        self.aggregation_loss_importance = aggregation_loss_importance
        self.use_answer_as_supervision = use_answer_as_supervision
        self.answer_loss_importance = answer_loss_importance
        self.use_normalized_answer_loss = use_normalized_answer_loss
        self.huber_loss_delta = huber_loss_delta
        self.temperature = temperature
        self.agg_temperature = agg_temperature
        self.use_gumbel_for_cells = use_gumbel_for_cells
        self.use_gumbel_for_agg = use_gumbel_for_agg
        self.average_approximation_function = average_approximation_function
        self.cell_select_pref = cell_select_pref
        self.answer_loss_cutoff = answer_loss_cutoff
        self.max_num_rows = max_num_rows
        self.max_num_columns = max_num_columns
        self.average_logits_per_cell = average_logits_per_cell
        self.select_one_column = select_one_column
        self.allow_empty_column_selection = allow_empty_column_selection
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
        self.reset_position_index_per_cell = reset_position_index_per_cell
        self.disable_per_token_loss = disable_per_token_loss
        self.span_prediction = span_prediction
