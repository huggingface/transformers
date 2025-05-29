import math

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import (
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Adapter,
    Wav2Vec2Encoder,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2SamePadLayer,
)
from .configuration_data2vec_audio import Data2VecAudioConfig


_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "Data2VecAudioConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95


class Data2VecAudioConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Data2VecAudioPadLayer(Wav2Vec2SamePadLayer):
    pass


class Data2VecAudioPositionalConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_pos_kernel_size,
            padding=config.conv_pos_kernel_size // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.padding = Data2VecAudioPadLayer(config.conv_pos_kernel_size)
        self.activation = ACT2FN[config.feat_extract_activation]
        # no learnable parameters
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Data2VecAudioPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Data2VecAudioPositionalConvLayer(config) for _ in range(config.num_conv_pos_embeddings)]
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Data2VecAudioFeatureEncoder(Wav2Vec2FeatureEncoder, nn.Module):
    def __init__(self, config):
        nn.Module.__init__()
        self.conv_layers = nn.ModuleList(
            [Data2VecAudioConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        )
        self.gradient_checkpointing = False
        self._requires_grad = True


class Data2VecAudioFeatureProjection(Wav2Vec2FeatureProjection):
    pass


class Data2VecAudioEncoder(Wav2Vec2Encoder):
    pass


class Data2VecAudioAdapter(Wav2Vec2Adapter):
    pass


class Data2VecAudioPreTrainedModel(PreTrainedModel, Wav2Vec2PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Data2VecAudioConfig
    base_model_prefix = "data2vec_audio"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Data2VecAudioFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, Data2VecAudioPositionalConvLayer):
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_adapters(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def init_adapter_layers(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def load_adapter(self):
        raise AttributeError("Not needed for Data2VecAudio")


DATA2VEC_AUDIO_START_DOCSTRING = r"""
    Data2VecAudio was proposed in [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and
    Language](https://arxiv.org/pdf/2202.03555) by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu and
    Michael Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Data2VecAudioConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DATA2VEC_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should be passed if the corresponding processor has `config.return_attention_mask ==
            True`, which is the case for all pre-trained Data2Vec Audio models. Be aware that that even with
            `attention_mask`, zero-padded inputs will have slightly different outputs compared to non-padded inputs
            because there are more than one convolutional layer in the positional encodings. For a more detailed
            explanation, see [here](https://github.com/huggingface/transformers/issues/25621#issuecomment-1713759349).

            </Tip>

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

Data2VecAudioBaseModelOutput = Wav2Vec2BaseModelOutput


@add_start_docstrings(
    "The bare Data2VecAudio Model transformer outputting raw hidden-states without any specific head on top.",
    DATA2VEC_AUDIO_START_DOCSTRING,
)
class Data2VecAudioModel(Data2VecAudioPreTrainedModel, Wav2Vec2Model):
    def __init__(self, config: Data2VecAudioConfig):
        Data2VecAudioPreTrainedModel.__init__(config)
        self.config = config
        self.feature_extractor = Data2VecAudioFeatureEncoder(config)
        self.feature_projection = Data2VecAudioFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        self.encoder = Data2VecAudioEncoder(config)

        self.adapter = Data2VecAudioAdapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Data2VecAudioBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


@add_start_docstrings(
    """Data2VecAudio Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    DATA2VEC_AUDIO_START_DOCSTRING,
)
class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel, Wav2Vec2ForCTC):
    def __init__(self, config):
        Data2VecAudioPreTrainedModel.__init__(config)

        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def tie_weights(self):
        raise AttributeError("Not needed for Data2VecAudio")

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


@add_start_docstrings(
    """
    Data2VecAudio Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
    like SUPERB Keyword Spotting.
    """,
    DATA2VEC_AUDIO_START_DOCSTRING,
)
class Data2VecAudioForSequenceClassification(Wav2Vec2ForSequenceClassification):
    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


@add_start_docstrings(
    """
    Data2VecAudio Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    DATA2VEC_AUDIO_START_DOCSTRING,
)
class Data2VecAudioForAudioFrameClassification(Wav2Vec2ForAudioFrameClassification):
    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


@add_start_docstrings(
    """
    Data2VecAudio Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    DATA2VEC_AUDIO_START_DOCSTRING,
)
class Data2VecAudioForXVector(Wav2Vec2ForXVector):
    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


__all__ = [
    "Data2VecAudioForAudioFrameClassification",
    "Data2VecAudioForCTC",
    "Data2VecAudioForSequenceClassification",
    "Data2VecAudioForXVector",
    "Data2VecAudioModel",
    "Data2VecAudioPreTrainedModel",
]
