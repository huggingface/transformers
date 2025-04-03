import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_peft_available,
    logging,
    replace_return_docstrings,
)

from .configuration_granite_speech import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GraniteSpeechConfig"


@dataclass
class GraniteSpeechCausalLMOutputWithPast(ModelOutput):
    """
    Base class for LlavaNext causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GraniteSpeechEncoderProjectorQFormer(nn.Module):
    def __init__(self, config: GraniteSpeechConfig):
        super().__init__()
        self.hidden_size = config.projector_config.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.window_size // config.downsample_rate

        self.query = nn.Parameter(torch.zeros(1, self.num_queries, config.projector_config.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        # Generally, this will be create the model for blip_2_qformer,
        # but we write it flexibly to allowed other projectors here as needed.
        self.qformer = AutoModel.from_config(config.projector_config)
        self.linear = nn.Linear(config.projector_config.hidden_size, config.text_config.hidden_size)

    def forward(self, x, atts):
        batch_size, seq_len, dim = x.size()
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        x = nn.functional.pad(x, (0, 0, 0, pad), "constant", 0)
        x = x.view(batch_size * nblocks, self.window_size, dim)

        query_output = self.qformer(
            query_embeds=self.query.data,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        query_proj = self.linear(
            query_output.last_hidden_state.view(batch_size, nblocks * self.window_size // self.downsample_rate, -1)
        )
        return query_proj


### Encoder
class GraniteSpeechCTCModel(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super(GraniteSpeechCTCModel, self).__init__()
        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.layers = nn.ModuleList([GraniteSpeechConformerBlock(config) for _ in range(config.num_layers)])

        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim, bias=True)
        self.context_size = config.context_size
        self.input_dim = config.input_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

    def forward(self, x: torch.Tensor):
        x = self.input_linear(x)
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, self.context_size)
            if idx == self.num_layers // 2:
                x_mid = x.clone()
                x_mid = self.out(x_mid)
                x += self.out_mid(nn.Softmax(dim=-1)(x_mid))
        return x


# NOTE: Conformer adapated from: https://github.com/lucidrains/conformer.git
class GraniteSpeechConformerDepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in, bias=False)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class GraniteSpeechConformerAttention(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        self.num_heads = config.num_heads
        inner_dim = config.dim_head * self.num_heads
        self.dim_head = config.dim_head
        self.scale = self.dim_head**-0.5
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.to_q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(config.hidden_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, config.hidden_dim)

        self.max_pos_emb = 512
        self.rel_pos_emb = nn.Embedding(2 * self.max_pos_emb + 1, self.dim_head)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, context_size):
        x = self.pre_norm(x)
        device, h, max_pos_emb = x.device, self.num_heads, self.max_pos_emb
        bs, n, d = x.shape
        assert context_size > 0 and context_size <= max_pos_emb

        nb = math.ceil(n / context_size)
        nr = n % context_size
        if nr > 0:
            # right padding to reach block size
            x = torch.nn.functional.pad(x, (0, 0, 0, context_size - nr))

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = [t.reshape(bs, nb, context_size, h, -1).transpose(2, 3) for t in (q, k, v)]

        # shaw's relative positional embedding
        seq = torch.arange(context_size, device=device)
        dist = seq.view(-1, 1) - seq.view(1, -1)
        dist = torch.clamp(dist, -context_size, context_size) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        rel_pos_emb_expanded = rel_pos_emb.view([1, 1, 1] + list(rel_pos_emb.shape))
        pos_attn = torch.sum(q.unsqueeze(-2) * rel_pos_emb_expanded, dim=-1) * self.scale

        if nr > 0:
            # masked attention in the extended block
            mask = torch.ones(context_size, context_size, dtype=bool, device=device)
            mask[:nr, :nr] = 0
            mask_value = -torch.finfo(pos_attn.dtype).max
            pos_attn[:, -1, :].masked_fill_(mask, mask_value)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=pos_attn, scale=self.scale)
        out = out.transpose(2, 3).reshape(bs, x.shape[1], -1)
        out = self.to_out(out[:, :n, :])
        return self.dropout(out)


class GraniteSpeechConformerFeedForward(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.up_proj = nn.Linear(config.hidden_dim, config.hidden_dim * config.feedforward_mult)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        self.down_proj = nn.Linear(config.hidden_dim * config.feedforward_mult, config.hidden_dim)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.up_proj(x)
        x = self.dropout(self.act_fn(x))
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class GraniteSpeechConformerConvModule(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor
        padding = self.calc_same_padding(config.conv_kernel_size)

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(config.hidden_dim, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depth_conv = GraniteSpeechConformerDepthWiseConv1d(
            inner_dim, inner_dim, kernel_size=config.conv_kernel_size, padding=padding
        )
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, config.hidden_dim, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.up_conv(x.permute(0, 2, 1))
        x = self.glu(x)
        x = self.depth_conv(x)
        x = self.silu(self.batch_norm(x))
        x = self.down_conv(x).permute(0, 2, 1)
        x = self.dropout(x)
        return x

    @staticmethod
    def calc_same_padding(kernel_size: int):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)


class GraniteSpeechConformerBlock(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super().__init__()
        self.ff1 = GraniteSpeechConformerFeedForward(config)
        self.attn = GraniteSpeechConformerAttention(config)
        self.conv = GraniteSpeechConformerConvModule(config)
        self.ff2 = GraniteSpeechConformerFeedForward(config)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, hidden_states, context_size):
        hidden_states = 0.5 * self.ff1(hidden_states) + hidden_states
        hidden_states = self.attn(hidden_states, context_size) + hidden_states
        hidden_states = self.conv(hidden_states) + hidden_states
        hidden_states = 0.5 * self.ff2(hidden_states) + hidden_states
        hidden_states = self.post_norm(hidden_states)
        return hidden_states


GRANITE_SPEECH_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (`GraniteSpeechConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Granite Speech Model outputting raw hidden-states without any specific head on top.",
    GRANITE_SPEECH_START_DOCSTRING,
)
class GraniteSpeechPreTrainedModel(PreTrainedModel):
    config_class = GraniteSpeechConfig
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


GRANITE_SPEECH_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, audio seq len, mel feat dim)):
            The tensors corresponding to the input audios. input features can be obtained using
            [`AutoFeatureExtractor`]. See [`GraniteSpeechFeatureExtractor.__call__`] for details.
            [`GraniteSpeechProcessor`] uses [`GraniteSpeechFeatureExtractor`] for processing audio.
        input_mask (`torch.Tensor`, *optional*)
            Mask for extracted audio features that should should be ignored when creating the merged
            multimodal representation (i.e., due to padding).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The Granite Speech model, which consists of an audio encoder, projector, and language model.""",
    GRANITE_SPEECH_START_DOCSTRING,
)
class GraniteSpeechForConditionalGeneration(GraniteSpeechPreTrainedModel, GenerationMixin):
    def __init__(self, config: GraniteSpeechConfig):
        super().__init__(config)
        # NOTE: It doesn't matter when we initialize from config, but we should be careful
        # to make sure this does not pick up the adapter_config if in the future we use
        # from_pretrained or something similar, since that should be set by the composite
        # model; don't need to consider it twice
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.encoder = GraniteSpeechCTCModel(config.encoder_config)
        self.projector = GraniteSpeechEncoderProjectorQFormer(config)

        if config.has_lora_adapter and not is_peft_available():
            logger.warning(
                "Config indicates that a lora adapter should be present, but "
                "peft is not installed; this will cause the model to perform "
                "incorrectly when audio inputs are provided. Please install "
                "peft and reload the model!"
            )

        self.post_init()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_audio_features(self, input_features):
        encoder_embeds = self.encoder(input_features)
        projected_embeds = self.projector(encoder_embeds, None)
        return projected_embeds

    @add_start_docstrings_to_model_forward(GRANITE_SPEECH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GraniteSpeechCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        input_features_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple[torch.Tensor], GraniteSpeechCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        TODO - add example for usage.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_features is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_features and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            # Get the base embeddings; set all audio tokens to 0 index
            # to avoid out of vocabulary issues with the LLM embedding.
            # Audio features will be masked into is_audio_idx indices later.
            is_audio_idx = input_ids == self.config.audio_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[is_audio_idx] = 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if input_features is not None:
            if input_features.dtype != self.dtype:
                logger.warning(f"input features are casted to {self.dtype}")
                input_features = input_features.to(self.dtype)
            # Get the audio features from the encoder / projector
            audio_features = self.get_audio_features(input_features)

            # Merge the audio features into the LLM embeddings
            inputs_embeds = self.get_merged_audio_embeddings(
                input_ids=input_ids,
                audio_features=audio_features,
                input_features_mask=input_features_mask,
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GraniteSpeechCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        input_features=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward audio inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, input_features should be None because
        # input ids do not contain special audio token anymore Otherwise we need
        # input feature values to be passed to the model
        if cache_position[0] == 0:
            model_inputs["input_features"] = input_features
        return model_inputs

    def get_merged_audio_embeddings(self, input_ids, audio_features, input_features_mask):
        """
        Adds the audio token to the model's LLM vocabulary so that we can pass it
        through the tokenizer; it's assumed that the embeddings corresponding to the
        <|audio|> token will be clobbered with speech features.
        """
        is_audio_index = input_ids == self.config.audio_token_index
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.language_model.get_input_embeddings()(llm_input_ids)  # [bsz, # features, hidden size]

        # Mask the audio features into the text embeddings
        special_audio_mask = is_audio_index.unsqueeze(-1)
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        if input_features_mask is not None:
            assert torch.all(
                is_audio_index.int().sum(dim=1) == input_features_mask.int().sum(dim=1)
            ).item(), "number of features should align"
            audio_features = audio_features[input_features_mask]

        inputs_embeds = inputs_embeds.masked_scatter(
            special_audio_mask,
            audio_features,
        )
        return inputs_embeds

    def generate(self, *args, **kwargs):
        """This model is expected to have a lora adapater, which is only
        enabled when considering audio inputs. As such, we override generate
        to conditionally enable / disable the lora adapter based on whether
        or not any input features were provided.
        """
        input_features = kwargs.pop("input_features", None)
        if is_peft_available and self._hf_peft_config_loaded:
            if input_features is not None:
                self.enable_adapters()
            else:
                self.disable_adapters()
        return super().generate(*args, input_features=input_features, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        # overwrite save_pretrained to first save the adapter if we have one
        # NOTE - this will use the base model path we are exporting in the lora
        # adapter, which may not necessarily be the best behavior, but for now
        # we keep this for portability, since using the local dir causes problems
        # if the model is loaded from outside of the current working dir.
        if is_peft_available and self._hf_peft_config_loaded:
            super().save_pretrained(*args, **kwargs)
        # Then save the base model afterwards
        self._hf_peft_config_loaded = False
        super().save_pretrained(*args, **kwargs)


__all__ = [
    "GraniteSpeechForConditionalGeneration",
    "GraniteSpeechPreTrainedModel",
]
