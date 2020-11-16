# coding=utf-8
# Copyright 2018, Hao Tan, Mohit Bansal
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
""" LXMERT model configuration """


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "unc-nlp/lxmert-base-uncased": "",
}


class LxmertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LxmertModel` or a
    :class:`~transformers.TFLxmertModel`. It is used to instantiate a LXMERT model according to the specified
    arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the LXMERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.LxmertModel` or
            :class:`~transformers.TFLxmertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        r_layers (:obj:`int`, `optional`, defaults to 5):
            Number of hidden layers in the Transformer visual encoder.
        l_layers (:obj:`int`, `optional`, defaults to 9):
            Number of hidden layers in the Transformer language encoder.
        x_layers (:obj:`int`, `optional`, defaults to 5):
            Number of hidden layers in the Transformer cross modality encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 5):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        visual_feat_dim (:obj:`int`, `optional`, defaults to 2048):
            This represents the last dimension of the pooled-object features used as input for the model, representing
            the size of each object feature itself.
        visual_pos_dim (:obj:`int`, `optional`, defaults to 4):
            This represents the number of spacial features that are mixed into the visual features. The default is set
            to 4 because most commonly this will represent the location of a bounding box. i.e., (x, y, width, height)
        visual_loss_normalizer (:obj:`float`, `optional`, defaults to 1/15):
            This represents the scaling factor in which each visual loss is multiplied by if during pretraining, one
            decided to train with multiple vision-based loss objectives.
        num_qa_labels (:obj:`int`, `optional`, defaults to 9500):
            This represents the total number of different question answering (QA) labels there are. If using more than
            one dataset with QA, the user will need to account for the total number of labels that all of the datasets
            have in total.
        num_object_labels (:obj:`int`, `optional`, defaults to 1600):
            This represents the total number of semantically unique objects that lxmert will be able to classify a
            pooled-object feature as belonging too.
        num_attr_labels (:obj:`int`, `optional`, defaults to 400):
            This represents the total number of semantically unique attributes that lxmert will be able to classify a
            pooled-object feature as possessing.
        task_matched (:obj:`bool`, `optional`, defaults to :obj:`True`):
            This task is used for sentence-image matching. If the sentence correctly describes the image the label will
            be 1. If the sentence does not correctly describe the image, the label will be 0.
        task_mask_lm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to add masked language modeling (as used in pretraining models such as BERT) to the loss
            objective.
        task_obj_predict (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to add object prediction, attribute ppredictionand feature regression to the loss objective.
        task_qa (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to add the question-asansweringoss to the objective
        visual_obj_loss (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to calculate the object-prediction loss objective
        visual_attr_loss (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to calculate the attribute-prediction loss objective
        visual_feat_loss (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to calculate the feature-regression loss objective
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return the attentions from the vision, language, and cross-modality layers
            should be returned.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return the hidden states from the vision, language, and cross-modality
            layers should be returned.
    """

    model_type = "lxmert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        num_labels=2,
        num_qa_labels=9500,
        num_object_labels=1600,
        num_attr_labels=400,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        l_layers=9,
        x_layers=5,
        r_layers=5,
        visual_feat_dim=2048,
        visual_pos_dim=4,
        visual_loss_normalizer=6.67,
        task_matched=True,
        task_mask_lm=True,
        task_obj_predict=True,
        task_qa=True,
        visual_obj_loss=True,
        visual_attr_loss=True,
        visual_feat_loss=True,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_qa_labels = num_qa_labels
        self.num_object_labels = num_object_labels
        self.num_attr_labels = num_attr_labels
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.visual_feat_dim = visual_feat_dim
        self.visual_pos_dim = visual_pos_dim
        self.visual_loss_normalizer = visual_loss_normalizer
        self.task_matched = task_matched
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_qa = task_qa
        self.visual_obj_loss = visual_obj_loss
        self.visual_attr_loss = visual_attr_loss
        self.visual_feat_loss = visual_feat_loss
        self.output_hidden_states = output_hidden_states
        self.output_attentions = self.output_attentions
        self.num_hidden_layers = {"vision": r_layers, "cross_encoder": x_layers, "language": l_layers}
