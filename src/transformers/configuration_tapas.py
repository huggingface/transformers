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
""" TAPAS configuration. Currently only the 4 fine-tuning tasks (SQA, WTQ, WIKISQL and
    WIKISQL_SUPERVISED) are supported. By default, intializes to SQA configuration. 
"""


import logging

from .configuration_bert import BertConfig


logger = logging.getLogger(__name__)

TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tapas-base": "", # to be added
    "tapas-large": "" # to be added
}

class TapasConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.TapasModel. 
    It is used to instantiate a TAPAS model according to the specified arguments, defining the model 
    architecture. Instantiating a configuration with the defaults will yield a similar configuration 
    to that of the TAPAS `tapas-base-finetuned-sqa (URL to be added) architecture. Configuration objects 
    inherit from :class:`~transformers.PreTrainedConfig` and can be used to control the model outputs. 
    Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    """
    
    model_type = "tapas"

    def __init__(
        self, 
        max_position_embeddings=1024,
        type_vocab_size=[3, 256, 256, 2, 256, 256, 10],
        task="SQA",
        learning_rate=5e-5 * (128 / 512),
        num_train_steps=200000,
        num_warmup_steps=200,
        use_tpu=False,
        positive_weight=10.0,
        num_aggregation_labels=0,
        num_classification_labels=0,
        aggregation_loss_importance=1.0,
        use_answer_as_supervision=False,
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
        grad_clipping=None,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=False,
        select_one_column=True,
        allow_empty_column_selection=False,
        #disabled_features=[],
        init_cell_selection_weights_to_zero=False,
        #disable_position_embeddings=False,
        reset_position_index_per_cell=False,
        disable_per_token_loss=False,
        span_prediction="none",
        **kwargs):

        super().__init__(max_position_embeddings=max_position_embeddings, type_vocab_size=type_vocab_size, **kwargs)
        
        # Fine-tuning task arguments
        self.task = task
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.positive_weight = positive_weight
        self.num_aggregation_labels = num_aggregation_labels
        self.num_classification_labels = num_classification_labels
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
        self.grad_clipping = grad_clipping
        self.max_num_rows=max_num_rows
        self.max_num_columns = max_num_columns
        self.average_logits_per_cell = average_logits_per_cell
        self.select_one_column = select_one_column
        self.allow_empty_column_selection = allow_empty_column_selection
        #self.disabled_features = disabled_features
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
        #self.disable_position_embeddings = disable_position_embeddings
        self.reset_position_index_per_cell = reset_position_index_per_cell
        self.disable_per_token_loss = disable_per_token_loss
        self.span_prediction = span_prediction

        if task == "SQA":
            # run_task_main.py hparams
            self.num_aggregation_labels = 0
            #self.do_model_aggregation = False
            self.use_answer_as_supervision = None
            
            # hparam_utils.py hparams
            self.init_cell_selection_weights_to_zero = False
            self.learning_rate = 5e-5 * (128 / 512)
            self.num_train_examples = 200000 * 128
            self.select_one_column = True
            self.allow_empty_column_selection = False
            self.train_batch_size = 128
            self.warmup_ratio = 0.01

            self.num_train_steps = int(self.num_train_examples / self.train_batch_size)
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_ratio)

        elif task == "WTQ":
            # run_task_main.py hparams
            self.num_aggregation_labels = 4
            #self.do_model_aggregation = True
            self.use_answer_as_supervision = True 
            
            # hparam_utils.py hparams
            self.grad_clipping = 10.0
            self.num_train_examples = 50000 * 512
            self.train_batch_size = 512
            self.answer_loss_cutoff = 0.664694
            self.cell_select_pref = 0.207951
            self.huber_loss_delta = 0.121194
            self.init_cell_selection_weights_to_zero = True
            self.learning_rate = 0.0000193581
            self.select_one_column = True
            self.allow_empty_column_selection = False
            self.temperature = 0.0352513
            self.warmup_ratio = 0.128960

            self.num_train_steps = int(self.num_train_examples / self.train_batch_size)
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_ratio)
    
        elif task == "WIKISQL":
            # run_task_main.py hparams
            self.num_aggregation_labels = 4
            #self.do_model_aggregation = True
            self.use_answer_as_supervision = True
            
            # hparam_utils.py hparams
            self.grad_clipping = 10.0
            self.num_train_examples = 50000 * 512
            self.train_batch_size = 512
            self.answer_loss_cutoff = 0.185567
            self.cell_select_pref = 0.611754
            self.huber_loss_delta = 1265.74
            self.init_cell_selection_weights_to_zero = False
            self.learning_rate = 0.0000617164
            self.select_one_column = False
            self.allow_empty_column_selection = False
            self.temperature = 0.107515
            self.warmup_ratio = 0.142400

            self.num_train_steps = int(self.num_train_examples / self.train_batch_size)
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_ratio)

        elif task == "WIKISQL_SUPERVISED":
            # run_task_main.py hparams
            self.num_aggregation_labels = 4
            #self.do_model_aggregation = True
            self.use_answer_as_supervision = False
            
            # hparam_utils.py hparams
            self.grad_clipping = 10.0
            self.num_train_examples = 50000 * 512
            self.train_batch_size = 512
            self.answer_loss_cutoff = 36.4519
            self.cell_select_pref = 0.903421
            self.huber_loss_delta = 222.088
            self.init_cell_selection_weights_to_zero = True
            self.learning_rate = 0.0000412331
            self.select_one_column = True
            self.allow_empty_column_selection = True
            self.temperature = 0.763141
            self.warmup_ratio = 0.168479

            self.num_train_steps = int(self.num_train_examples / self.train_batch_size)
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_ratio)
        
        else:
            raise ValueError(f'Unknown task: {task}')
