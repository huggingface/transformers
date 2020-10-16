# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Model class. """


import warnings
from collections import OrderedDict

from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    BertGenerationConfig,
    BlenderbotConfig,
    CamembertConfig,
    CTRLConfig,
    DebertaConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    FSMTConfig,
    FunnelConfig,
    GPT2Config,
    LayoutLMConfig,
    LongformerConfig,
    LxmertConfig,
    MBartConfig,
    MobileBertConfig,
    OpenAIGPTConfig,
    PegasusConfig,
    ProphetNetConfig,
    ReformerConfig,
    RetriBertConfig,
    RobertaConfig,
    SqueezeBertConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMProphetNetConfig,
    XLMRobertaConfig,
    XLNetConfig,
    replace_list_option_in_docstrings,
)
from .configuration_marian import MarianConfig
from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings
from .modeling_albert import (
    AlbertForMaskedLM,
    AlbertForMultipleChoice,
    AlbertForPreTraining,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertModel,
)
from .modeling_bart import (
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartForSequenceClassification,
    BartModel,
)
from .modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLMHeadModel,
    BertModel,
)
from .modeling_bert_generation import BertGenerationDecoder, BertGenerationEncoder
from .modeling_blenderbot import BlenderbotForConditionalGeneration
from .modeling_camembert import (
    CamembertForCausalLM,
    CamembertForMaskedLM,
    CamembertForMultipleChoice,
    CamembertForQuestionAnswering,
    CamembertForSequenceClassification,
    CamembertForTokenClassification,
    CamembertModel,
)
from .modeling_ctrl import CTRLLMHeadModel, CTRLModel
from .modeling_deberta import DebertaForSequenceClassification, DebertaModel
from .modeling_distilbert import (
    DistilBertForMaskedLM,
    DistilBertForMultipleChoice,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
)
from .modeling_dpr import DPRQuestionEncoder
from .modeling_electra import (
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraModel,
)
from .modeling_encoder_decoder import EncoderDecoderModel
from .modeling_flaubert import (
    FlaubertForMultipleChoice,
    FlaubertForQuestionAnsweringSimple,
    FlaubertForSequenceClassification,
    FlaubertForTokenClassification,
    FlaubertModel,
    FlaubertWithLMHeadModel,
)
from .modeling_fsmt import FSMTForConditionalGeneration, FSMTModel
from .modeling_funnel import (
    FunnelForMaskedLM,
    FunnelForMultipleChoice,
    FunnelForQuestionAnswering,
    FunnelForSequenceClassification,
    FunnelForTokenClassification,
    FunnelModel,
)
from .modeling_gpt2 import GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model
from .modeling_layoutlm import LayoutLMForMaskedLM, LayoutLMForTokenClassification, LayoutLMModel
from .modeling_longformer import (
    LongformerForMaskedLM,
    LongformerForMultipleChoice,
    LongformerForQuestionAnswering,
    LongformerForSequenceClassification,
    LongformerForTokenClassification,
    LongformerModel,
)
from .modeling_lxmert import LxmertForPreTraining, LxmertModel
from .modeling_marian import MarianMTModel
from .modeling_mbart import MBartForConditionalGeneration
from .modeling_mobilebert import (
    MobileBertForMaskedLM,
    MobileBertForMultipleChoice,
    MobileBertForPreTraining,
    MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification,
    MobileBertForTokenClassification,
    MobileBertModel,
)
from .modeling_openai import OpenAIGPTForSequenceClassification, OpenAIGPTLMHeadModel, OpenAIGPTModel
from .modeling_pegasus import PegasusForConditionalGeneration
from .modeling_prophetnet import ProphetNetForCausalLM, ProphetNetForConditionalGeneration, ProphetNetModel
from .modeling_rag import (  # noqa: F401 - need to import all RagModels to be in globals() function
    RagModel,
    RagSequenceForGeneration,
    RagTokenForGeneration,
)
from .modeling_reformer import (
    ReformerForMaskedLM,
    ReformerForQuestionAnswering,
    ReformerModel,
    ReformerModelWithLMHead,
)
from .modeling_retribert import RetriBertModel
from .modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .modeling_squeezebert import (
    SqueezeBertForMaskedLM,
    SqueezeBertForMultipleChoice,
    SqueezeBertForQuestionAnswering,
    SqueezeBertForSequenceClassification,
    SqueezeBertForTokenClassification,
    SqueezeBertModel,
)
from .modeling_t5 import T5ForConditionalGeneration, T5Model
from .modeling_transfo_xl import TransfoXLLMHeadModel, TransfoXLModel
from .modeling_xlm import (
    XLMForMultipleChoice,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMWithLMHeadModel,
)
from .modeling_xlm_prophetnet import (
    XLMProphetNetForCausalLM,
    XLMProphetNetForConditionalGeneration,
    XLMProphetNetModel,
)
from .modeling_xlm_roberta import (
    XLMRobertaForCausalLM,
    XLMRobertaForMaskedLM,
    XLMRobertaForMultipleChoice,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
)
from .modeling_xlnet import (
    XLNetForMultipleChoice,
    XLNetForQuestionAnsweringSimple,
    XLNetForSequenceClassification,
    XLNetForTokenClassification,
    XLNetLMHeadModel,
    XLNetModel,
)
from .utils import logging


logger = logging.get_logger(__name__)


MODEL_MAPPING = OrderedDict(
    [
        (RetriBertConfig, RetriBertModel),
        (T5Config, T5Model),
        (DistilBertConfig, DistilBertModel),
        (AlbertConfig, AlbertModel),
        (CamembertConfig, CamembertModel),
        (XLMRobertaConfig, XLMRobertaModel),
        (BartConfig, BartModel),
        (LongformerConfig, LongformerModel),
        (RobertaConfig, RobertaModel),
        (LayoutLMConfig, LayoutLMModel),
        (SqueezeBertConfig, SqueezeBertModel),
        (BertConfig, BertModel),
        (OpenAIGPTConfig, OpenAIGPTModel),
        (GPT2Config, GPT2Model),
        (MobileBertConfig, MobileBertModel),
        (TransfoXLConfig, TransfoXLModel),
        (XLNetConfig, XLNetModel),
        (FlaubertConfig, FlaubertModel),
        (FSMTConfig, FSMTModel),
        (XLMConfig, XLMModel),
        (CTRLConfig, CTRLModel),
        (ElectraConfig, ElectraModel),
        (ReformerConfig, ReformerModel),
        (FunnelConfig, FunnelModel),
        (LxmertConfig, LxmertModel),
        (BertGenerationConfig, BertGenerationEncoder),
        (DebertaConfig, DebertaModel),
        (DPRConfig, DPRQuestionEncoder),
        (XLMProphetNetConfig, XLMProphetNetModel),
        (ProphetNetConfig, ProphetNetModel),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (RetriBertConfig, RetriBertModel),
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForPreTraining),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (FSMTConfig, FSMTForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForPreTraining),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (MobileBertConfig, MobileBertForPreTraining),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForPreTraining),
        (LxmertConfig, LxmertForPreTraining),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (MarianConfig, MarianMTModel),
        (FSMTConfig, FSMTForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (MobileBertConfig, MobileBertForMaskedLM),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (FunnelConfig, FunnelForMaskedLM),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        (CamembertConfig, CamembertForCausalLM),
        (XLMRobertaConfig, XLMRobertaForCausalLM),
        (RobertaConfig, RobertaForCausalLM),
        (BertConfig, BertLMHeadModel),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (
            XLMConfig,
            XLMWithLMHeadModel,
        ),  # XLM can be MLM and CLM => model should be split similar to BERT; leave here for now
        (CTRLConfig, CTRLLMHeadModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (BertGenerationConfig, BertGenerationDecoder),
        (ProphetNetConfig, XLMProphetNetForCausalLM),
        (ProphetNetConfig, ProphetNetForCausalLM),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (MobileBertConfig, MobileBertForMaskedLM),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (ReformerConfig, ReformerForMaskedLM),
        (FunnelConfig, FunnelForMaskedLM),
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        (T5Config, T5ForConditionalGeneration),
        (PegasusConfig, PegasusForConditionalGeneration),
        (MarianConfig, MarianMTModel),
        (MBartConfig, MBartForConditionalGeneration),
        (BlenderbotConfig, BlenderbotForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (FSMTConfig, FSMTForConditionalGeneration),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (XLMProphetNetConfig, XLMProphetNetForConditionalGeneration),
        (ProphetNetConfig, ProphetNetForConditionalGeneration),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForSequenceClassification),
        (AlbertConfig, AlbertForSequenceClassification),
        (CamembertConfig, CamembertForSequenceClassification),
        (XLMRobertaConfig, XLMRobertaForSequenceClassification),
        (BartConfig, BartForSequenceClassification),
        (LongformerConfig, LongformerForSequenceClassification),
        (RobertaConfig, RobertaForSequenceClassification),
        (SqueezeBertConfig, SqueezeBertForSequenceClassification),
        (BertConfig, BertForSequenceClassification),
        (XLNetConfig, XLNetForSequenceClassification),
        (MobileBertConfig, MobileBertForSequenceClassification),
        (FlaubertConfig, FlaubertForSequenceClassification),
        (XLMConfig, XLMForSequenceClassification),
        (ElectraConfig, ElectraForSequenceClassification),
        (FunnelConfig, FunnelForSequenceClassification),
        (DebertaConfig, DebertaForSequenceClassification),
        (GPT2Config, GPT2ForSequenceClassification),
        (OpenAIGPTConfig, OpenAIGPTForSequenceClassification),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForQuestionAnswering),
        (AlbertConfig, AlbertForQuestionAnswering),
        (CamembertConfig, CamembertForQuestionAnswering),
        (BartConfig, BartForQuestionAnswering),
        (LongformerConfig, LongformerForQuestionAnswering),
        (XLMRobertaConfig, XLMRobertaForQuestionAnswering),
        (RobertaConfig, RobertaForQuestionAnswering),
        (SqueezeBertConfig, SqueezeBertForQuestionAnswering),
        (BertConfig, BertForQuestionAnswering),
        (XLNetConfig, XLNetForQuestionAnsweringSimple),
        (FlaubertConfig, FlaubertForQuestionAnsweringSimple),
        (MobileBertConfig, MobileBertForQuestionAnswering),
        (XLMConfig, XLMForQuestionAnsweringSimple),
        (ElectraConfig, ElectraForQuestionAnswering),
        (ReformerConfig, ReformerForQuestionAnswering),
        (FunnelConfig, FunnelForQuestionAnswering),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (LayoutLMConfig, LayoutLMForTokenClassification),
        (DistilBertConfig, DistilBertForTokenClassification),
        (CamembertConfig, CamembertForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (XLMConfig, XLMForTokenClassification),
        (XLMRobertaConfig, XLMRobertaForTokenClassification),
        (LongformerConfig, LongformerForTokenClassification),
        (RobertaConfig, RobertaForTokenClassification),
        (SqueezeBertConfig, SqueezeBertForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (MobileBertConfig, MobileBertForTokenClassification),
        (XLNetConfig, XLNetForTokenClassification),
        (AlbertConfig, AlbertForTokenClassification),
        (ElectraConfig, ElectraForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (FunnelConfig, FunnelForTokenClassification),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        (CamembertConfig, CamembertForMultipleChoice),
        (ElectraConfig, ElectraForMultipleChoice),
        (XLMRobertaConfig, XLMRobertaForMultipleChoice),
        (LongformerConfig, LongformerForMultipleChoice),
        (RobertaConfig, RobertaForMultipleChoice),
        (SqueezeBertConfig, SqueezeBertForMultipleChoice),
        (BertConfig, BertForMultipleChoice),
        (DistilBertConfig, DistilBertForMultipleChoice),
        (MobileBertConfig, MobileBertForMultipleChoice),
        (XLNetConfig, XLNetForMultipleChoice),
        (AlbertConfig, AlbertForMultipleChoice),
        (XLMConfig, XLMForMultipleChoice),
        (FlaubertConfig, FlaubertForMultipleChoice),
        (FunnelConfig, FunnelForMultipleChoice),
    ]
)

AUTO_MODEL_PRETRAINED_DOCSTRING = r"""

        The model class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        The model is set in evaluation mode by default using ``model.eval()`` (so for instance, dropout modules are
        deactivated). To train the model, you should first set it back in training mode with ``model.train()``

        Args:
            pretrained_model_name_or_path:
                Can be either:

                    - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                      ``bert-base-uncased``.
                    - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                      ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args (additional positional arguments, `optional`):
                Will be passed along to the underlying model ``__init__()`` method.
            config (:class:`~transformers.PretrainedConfig`, `optional`):
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `shortcut name` string of a
                      pretrained model).
                    - The model was saved using :meth:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by suppling the save directory.
                    - The model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, `optional`):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~transformers.PreTrainedModel.save_pretrained` and
                :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each
                request.
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error
                messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try doanloading the model).
            use_cdn(:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to use Cloudfront (a Content Delivery Network, or CDN) when searching for the model on
                our S3 (faster). Should be set to :obj:`False` for checkpoints larger than 20GB.
            kwargs (additional keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:

                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                      ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's ``__init__`` function.
"""


class AutoModel:
    r"""
    This is a generic model class that will be instantiated as one of the base model classes of the library
    when created with the :meth:`~transformers.AutoModel.from_pretrained` class method or the
    :meth:`~transformers.AutoModel.from_config` class methods.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModel.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the base model classes of the library from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :meth:`~transformers.AutoModel.from_pretrained` to load
            the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModel
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModel.from_config(config)
        """
        if type(config) in MODEL_MAPPING.keys():
            return MODEL_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the base model classes of the library from a pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""

        Examples::

            >>> from transformers import AutoConfig, AutoModel

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModel.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_MAPPING.keys():
            return MODEL_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_MAPPING.keys())
            )
        )


class AutoModelForPreTraining:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with the
    architecture used for pretraining this model---when created with the when created with the
    :meth:`~transformers.AutoModelForPreTraining.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForPreTraining.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForPreTraining is designed to be instantiated "
            "using the `AutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForPreTraining.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_PRETRAINING_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with the architecture used for pretraining this
        model---from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use
            :meth:`~transformers.AutoModelForPreTraining.from_pretrained` to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForPreTraining
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForPreTraining.from_config(config)
        """
        if type(config) in MODEL_FOR_PRETRAINING_MAPPING.keys():
            return MODEL_FOR_PRETRAINING_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_PRETRAINING_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with the architecture used for pretraining this ",
        "model---from a pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForPreTraining

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForPreTraining.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForPreTraining.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForPreTraining.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_PRETRAINING_MAPPING.keys():
            return MODEL_FOR_PRETRAINING_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )


class AutoModelWithLMHead:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    language modeling head---when created with the when created with the
    :meth:`~transformers.AutoModelWithLMHead.from_pretrained` class method or the
    :meth:`~transformers.AutoModelWithLMHead.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).

    .. warning::

        This class is deprecated and will be removed in a future version. Please use
        :class:`~transformers.AutoModelForCausalLM` for causal language models,
        :class:`~transformers.AutoModelForMaskedLM` for masked language models and
        :class:`~transformers.AutoModelForSeq2SeqLM` for encoder-decoder models.
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelWithLMHead is designed to be instantiated "
            "using the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelWithLMHead.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_WITH_LM_HEAD_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a language modeling head---from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :meth:`~transformers.AutoModelWithLMHead.from_pretrained`
            to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelWithLMHead
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelWithLMHead.from_config(config)
        """
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        if type(config) in MODEL_WITH_LM_HEAD_MAPPING.keys():
            return MODEL_WITH_LM_HEAD_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_WITH_LM_HEAD_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a language modeling head---from a pretrained ",
        "model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelWithLMHead

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelWithLMHead.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelWithLMHead.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_WITH_LM_HEAD_MAPPING.keys():
            return MODEL_WITH_LM_HEAD_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )


class AutoModelForCausalLM:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    causal language modeling head---when created with the when created with the
    :meth:`~transformers.AutoModelForCausalLM.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForCausalLM.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForCausalLM is designed to be instantiated "
            "using the `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForCausalLM.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_CAUSAL_LM_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a causal language modeling head---from a
        configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :meth:`~transformers.AutoModelForCausalLM.from_pretrained`
            to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForCausalLM
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('gpt2')
            >>> model = AutoModelForCausalLM.from_config(config)
        """
        if type(config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
            return MODEL_FOR_CAUSAL_LM_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_CAUSAL_LM_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_CAUSAL_LM_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a causal language modeling head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForCausalLM

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForCausalLM.from_pretrained('gpt2')

            >>> # Update configuration during loading
            >>> model = AutoModelForCausalLM.from_pretrained('gpt2', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/gpt2_tf_model_config.json')
            >>> model = AutoModelForCausalLM.from_pretrained('./tf_model/gpt2_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
            return MODEL_FOR_CAUSAL_LM_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_CAUSAL_LM_MAPPING.keys())
            )
        )


class AutoModelForMaskedLM:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    masked language modeling head---when created with the when created with the
    :meth:`~transformers.AutoModelForMaskedLM.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForMasedLM.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForMaskedLM is designed to be instantiated "
            "using the `AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForMaskedLM.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_MASKED_LM_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a masked language modeling head---from a
        configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :meth:`~transformers.AutoModelForMaskedLM.from_pretrained`
            to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForMaskedLM
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForMaskedLM.from_config(config)
        """
        if type(config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
            return MODEL_FOR_MASKED_LM_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_MASKED_LM_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_MASKED_LM_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a masked language modeling head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForMaskedLM

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForMaskedLM.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
            return MODEL_FOR_MASKED_LM_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_MASKED_LM_MAPPING.keys())
            )
        )


class AutoModelForSeq2SeqLM:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    sequence-to-sequence language modeling head---when created with the when created with the
    :meth:`~transformers.AutoModelForSeq2SeqLM.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForSeq2SeqLM.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForSeq2SeqLM is designed to be instantiated "
            "using the `AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForSeq2SeqLM.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a sequence-to-sequence language modeling
        head---from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :meth:`~transformers.AutoModelForSeq2SeqLM.from_pretrained`
            to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForSeq2SeqLM
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('t5')
            >>> model = AutoModelForSeq2SeqLM.from_config(config)
        """
        if type(config) in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys():
            return MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()),
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a sequence-to-sequence language modeling "
        "head---from a pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForSeq2SeqLM

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

            >>> # Update configuration during loading
            >>> model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/t5_tf_model_config.json')
            >>> model = AutoModelForSeq2SeqLM.from_pretrained('./tf_model/t5_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys():
            return MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()),
            )
        )


class AutoModelForSequenceClassification:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    sequence classification head---when created with the when created with the
    :meth:`~transformers.AutoModelForSequenceClassification.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForSequenceClassification.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForSequenceClassification is designed to be instantiated "
            "using the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a sequence classification head---from a
        configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use
            :meth:`~transformers.AutoModelForSequenceClassification.from_pretrained` to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForSequenceClassification
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForSequenceClassification.from_config(config)
        """
        if type(config) in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a sequence classification head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForSequenceClassification

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )


class AutoModelForQuestionAnswering:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    question answering head---when created with the when created with the
    :meth:`~transformers.AutoModeForQuestionAnswering.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForQuestionAnswering.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForQuestionAnswering is designed to be instantiated "
            "using the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForQuestionAnswering.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_QUESTION_ANSWERING_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a question answering head---from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use
            :meth:`~transformers.AutoModelForQuestionAnswering.from_pretrained` to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForQuestionAnswering
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForQuestionAnswering.from_config(config)
        """
        if type(config) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys():
            return MODEL_FOR_QUESTION_ANSWERING_MAPPING[type(config)](config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_QUESTION_ANSWERING_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a question answering head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForQuestionAnswering

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForQuestionAnswering.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys():
            return MODEL_FOR_QUESTION_ANSWERING_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )


class AutoModelForTokenClassification:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    token classification head---when created with the when created with the
    :meth:`~transformers.AutoModelForTokenClassification.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForTokenClassification.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a token classification head---from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use
            :meth:`~transformers.AutoModelForTokenClassification.from_pretrained` to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForTokenClassification
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForTokenClassification.from_config(config)
        """
        if type(config) in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING[type(config)](config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a token classification head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForTokenClassification

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForTokenClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )


class AutoModelForMultipleChoice:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    multiple choice classifcation head---when created with the when created with the
    :meth:`~transformers.AutoModelForMultipleChoice.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForMultipleChoice.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForMultipleChoice is designed to be instantiated "
            "using the `AutoModelForMultipleChoice.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForMultipleChoice.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_MULTIPLE_CHOICE_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a multiple choice classification head---from a
        configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use
            :meth:`~transformers.AutoModelForMultipleChoice.from_pretrained` to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForMultipleChoice
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForMultipleChoice.from_config(config)
        """
        if type(config) in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys():
            return MODEL_FOR_MULTIPLE_CHOICE_MAPPING[type(config)](config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys()),
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_MULTIPLE_CHOICE_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a multiple choice classification head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForMultipleChoice

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForMultipleChoice.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys():
            return MODEL_FOR_MULTIPLE_CHOICE_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys()),
            )
        )
