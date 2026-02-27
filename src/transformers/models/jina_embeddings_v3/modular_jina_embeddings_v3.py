from collections.abc import Callable

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...integrations import use_kernelized_func
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedConfig
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from ..xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaIntermediate,
    XLMRobertaLMHead,
    XLMRobertaModel,
    XLMRobertaOutput,
    XLMRobertaPooler,
    XLMRobertaPreTrainedModel,
    XLMRobertaSelfOutput,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


class JinaEmbeddingsV3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JinaEmbeddingsV3Model`]. It
    is used to instantiate a Jina-Embeddings-V3 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Jina-Embeddings-V3
    [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of the Jina-Embeddings-V3 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`JinaEmbeddingsV3Model'`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 8194):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g.,  2048 or 4096 or 8194).
        type_vocab_size (`int`, *optional*, defaults to 1):
            The vocabulary size of the `token_type_ids` passed when calling [`JinaEmbeddingsV3Model`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        position_embedding_type (`str`, *optional*, defaults to `"rotary"`):
            The type of position embedding to use in the model.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        lora_adaptations (`list[str]`, *optional*):
            List of task-specific LoRA adaptation names. Defaults to
            `["retrieval.query", "retrieval.passage", "separation", "classification", "text-matching"]`.
        task_instructions (`dict[str, str]`, *optional*):
            Dictionary mapping task names to their instruction prompts.
        lora_rank (`int`, *optional*, defaults to 4):
            The rank of the LoRA adaptation matrices.
        lora_dropout_p (`float`, *optional*, defaults to 0.0):
            Dropout probability for LoRA layers.
        lora_alpha (`int`, *optional*, defaults to 1):
            Scaling factor for LoRA adaptations.
        load_trained_adapters (`bool`, *optional*, defaults to `True`):
            Whether to load trained adapter weights.
        matryoshka_dimensions (`list[int]`, *optional*):
            List of supported dimensions for matryoshka representation learning.
        truncate_dim (`int`, *optional*):
            Dimension to truncate embeddings to.

    Examples:

    ```python
    >>> from transformers import JinaEmbeddingsV3Config, JinaEmbeddingsV3Model

    >>> # Initializing a Jina-Embeddings-V3 jinaai/jina-embeddings-v3 style configuration
    >>> configuration = JinaEmbeddingsV3Config()

    >>> # Initializing a model (with random weights) from the jinaai/jina-embeddings-v3 style configuration
    >>> model = JinaEmbeddingsV3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "jina_embeddings_v3"

    def __init__(
        self,
        vocab_size: int | None = 250002,
        hidden_size: int | None = 1024,
        num_hidden_layers: int | None = 24,
        num_attention_heads: int | None = 16,
        intermediate_size: int | None = 4096,
        hidden_act: str | None = "gelu",
        hidden_dropout_prob: float | None = 0.1,
        attention_probs_dropout_prob: float | None = 0.1,
        max_position_embeddings: int | None = 8194,
        type_vocab_size: int | None = 1,
        initializer_range: float | None = 0.02,
        layer_norm_eps: float | None = 1e-5,
        pad_token_id: int | None = 1,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 2,
        position_embedding_type: str | None = "rotary",
        rope_parameters: RopeParameters | dict | None = None,
        classifier_dropout: float | None = None,
        lora_adaptations: list[str] | None = None,
        task_instructions: dict[str, str] | None = None,
        lora_rank: int | None = 4,
        lora_dropout_p: float | None = 0.0,
        lora_alpha: int | None = 1,
        load_trained_adapters: bool | None = True,
        emb_pooler: str | None = None,
        matryoshka_dimensions: list[int] | None = None,
        truncate_dim: int | None = None,
        **kwargs,
    ):
        if rope_parameters is None:
            rope_parameters = {"rope_theta": 20000.0}

        if lora_adaptations is None:
            lora_adaptations = [
                "retrieval.query",
                "retrieval.passage",
                "separation",
                "classification",
                "text-matching",
            ]

        if task_instructions is None:
            task_instructions = {
                "retrieval.query": "Represent the query for retrieving evidence documents: ",
                "retrieval.passage": "Represent the document for retrieval: ",
                "separation": "",
                "classification": "",
                "text-matching": "",
            }

        if matryoshka_dimensions is None:
            matryoshka_dimensions = [32, 64, 128, 256, 512, 768, 1024]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.position_embedding_type = position_embedding_type
        self.rope_parameters = rope_parameters
        self.classifier_dropout = classifier_dropout
        self.lora_adaptations = lora_adaptations
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_p = lora_dropout_p
        self.load_trained_adapters = load_trained_adapters
        self.matryoshka_dimensions = matryoshka_dimensions
        self.task_instructions = task_instructions
        self.emb_pooler = emb_pooler
        self.truncate_dim = truncate_dim

        super().__init__(**kwargs)


class JinaEmbeddingsV3Embeddings(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.shape
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(input_shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class JinaEmbeddingsV3RotaryEmbedding(LlamaRotaryEmbedding):
    pass


@use_kernelized_func(apply_rotary_pos_emb)
class JinaEmbeddingsV3SelfAttention(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.attention_head_size * config.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seq_len, 3, self.num_attention_heads, self.attention_head_size)

        qkv = self.Wqkv(hidden_states).view(hidden_shape)
        query_states, key_states, value_states = qkv.unbind(dim=-3)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        return attn_output, attn_weights


class JinaEmbeddingsV3SelfOutput(XLMRobertaSelfOutput):
    pass


class JinaEmbeddingsV3Attention(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.attention_class = JinaEmbeddingsV3SelfAttention(config)
        self.output = JinaEmbeddingsV3SelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        attention_output, attn_weights = self.attention_class(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights


class JinaEmbeddingsV3Intermediate(XLMRobertaIntermediate):
    pass


class JinaEmbeddingsV3Output(XLMRobertaOutput):
    pass


class JinaEmbeddingsV3Layer(GradientCheckpointingLayer):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.attention = JinaEmbeddingsV3Attention(config)
        self.intermediate = JinaEmbeddingsV3Intermediate(config)
        self.output = JinaEmbeddingsV3Output(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            position_embeddings,
            **kwargs,
        )

        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class JinaEmbeddingsV3Encoder(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([JinaEmbeddingsV3Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
        )


class JinaEmbeddingsV3Pooler(XLMRobertaPooler):
    pass


class JinaEmbeddingsV3PreTrainedModel(XLMRobertaPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": JinaEmbeddingsV3Layer,
        "attentions": JinaEmbeddingsV3Attention,
    }


@auto_docstring
class JinaEmbeddingsV3Model(XLMRobertaModel):
    def __init__(self, config: JinaEmbeddingsV3Config, add_pooling_layer=True):
        super().__init__(config)
        del self.gradient_checkpointing
        self.rotary_emb = JinaEmbeddingsV3RotaryEmbedding(config)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling | tuple:
        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape[0], input_shape[1]

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = (input_ids != self.config.pad_token_id).long()
            else:
                # Cannot infer padding from embeddings alone, defaulting to all ones
                attention_mask = torch.ones(input_shape, device=device, dtype=torch.long)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

    def _create_attention_masks(self):
        raise AttributeError("Not needed for JinaEmbeddingsV3")


class JinaEmbeddingsV3LMHead(XLMRobertaLMHead):
    pass


class JinaEmbeddingsV3ForMaskedLM(JinaEmbeddingsV3PreTrainedModel):
    _tied_weights_keys = {
        "lm_head.decoder.weight": "roberta.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config):
        super().__init__(config)

        self.roberta = JinaEmbeddingsV3Model(config, add_pooling_layer=False)
        self.lm_head = JinaEmbeddingsV3LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        return self.roberta.embeddings.word_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        r"""
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.

            [What are token type IDs?](../glossary#token-type-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        sequence_output = outputs[0]

        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class JinaEmbeddingsV3ForSequenceClassification(XLMRobertaForSequenceClassification):
    pass


class JinaEmbeddingsV3ForTokenClassification(XLMRobertaForTokenClassification):
    pass


class JinaEmbeddingsV3ForQuestionAnswering(XLMRobertaForQuestionAnswering):
    pass


__all__ = [
    "JinaEmbeddingsV3Config",
    "JinaEmbeddingsV3PreTrainedModel",
    "JinaEmbeddingsV3Model",
    "JinaEmbeddingsV3ForMaskedLM",
    "JinaEmbeddingsV3ForSequenceClassification",
    "JinaEmbeddingsV3ForTokenClassification",
    "JinaEmbeddingsV3ForQuestionAnswering",
    "JinaEmbeddingsV3Layer",
]
