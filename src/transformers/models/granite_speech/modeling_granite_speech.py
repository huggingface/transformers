import math
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import torch
import torch.utils.checkpoint
from torch import nn, einsum
import torch.nn.functional as F

from ...modeling_outputs import ModelOutput

from transformers import Blip2QFormerModel
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from ..auto import AutoModelForCausalLM
from .configuration_granite_speech import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
)


@dataclass
class GraniteSpeechCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None


### Projector
class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config: GraniteSpeechConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ds_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = self.window_size // self.ds_rate
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, config.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        # NOTE: It would be better to create this from config, similar to the LLM.
        # To do this, we need to register the QFormer model into an automodel, which
        # will require pulling it out into its own dir so that it's accessible under
        # transformers.models.X
        self.qformer = Blip2QFormerModel(config)
        self.linear = nn.Linear(config.hidden_size, config.llm_dim)

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
            query_output.last_hidden_state.view(
                batch_size, nblocks * self.window_size // self.ds_rate, -1
            )
        )
        return query_proj

### Encoder
class CTCModel(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super(CTCModel, self).__init__()

        self.rnn_trL = [nn.Linear(config.input_dim, config.hidden_dim, bias=True)]
        for l in range(config.num_layers):
            self.rnn_trL.append(
                ConformerBlock(
                    dim=config.hidden_dim,
                    dim_head=config.dim_head,
                    heads=config.num_heads,
                    ff_mult=config.feedforward_mult,
                    conv_expansion_factor=config.conv_expansion_factor,
                    conv_kernel_size=config.conv_kernel_size,
                    context_size=config.context_size,  # attention context size
                    attn_dropout=config.dropout,
                    ff_dropout=config.dropout,
                    conv_dropout=config.dropout,
                )
            )
            self.rnn_tr = nn.Sequential(*self.rnn_trL)

        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim, bias=True)
        self.context_size = config.context_size
        self.input_dim = config.input_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

    def forward(self, x: torch.Tensor):
        x = self.rnn_trL[0](x)
        for l in range(1, self.num_layers + 1):
            x = self.rnn_trL[l](x, self.context_size)
            if l == self.num_layers // 2:
                x_mid = x.clone()
                x_mid = self.out(x_mid)
                x += self.out_mid(nn.Softmax(dim=-1)(x_mid))
        return x


# NOTE: Conformer adapated from: https://github.com/lucidrains/conformer.git
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        x = x.permute(self.dims)
        return x


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in, bias=False)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context_size, **kwargs):
        x = self.norm(x)
        return self.fn(x, context_size, **kwargs)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        context_size=200,
        max_pos_emb=512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context_size):
        device, h, max_pos_emb = x.device, self.heads, self.max_pos_emb
        bs, n, d = x.shape
        assert(context_size > 0 and context_size <= max_pos_emb)

        nb = n // context_size
        nr = n % context_size
        if nr > 0:
            y = torch.zeros(x.shape[0], context_size-nr, x.shape[2], device=device)
            x = torch.cat((x,y), dim=1)
            nb += 1

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q, k, v = map(
            lambda t: t.reshape(bs, nb, context_size, h, -1).transpose(2, 3),
            (q, k, v),
        )
        dots = einsum('b m h i d, b m h j d -> b m h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(context_size, device = device)
        dist = seq.view(-1, 1) - seq.view(1, -1)
        dist = torch.clamp(dist,-context_size, context_size) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b m h c d, c r d -> b m h c r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if nr > 0:
            mask = torch.ones(context_size, context_size, device=device)
            mask[:nr,:nr] = 0
            mask_value = -torch.finfo(dots.dtype).max
            dots[:,-1,:].masked_fill_(mask.bool(), mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b m h i j, b m h j d -> b m h i d', attn, v)
        out = out.transpose(2, 3).reshape(bs, x.shape[1], -1)
        out = self.to_out(out[:,:n,:])
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = self.calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Permute(dims=(0, 2, 1)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            DepthWiseConv1d(inner_dim,
                            inner_dim,
                            kernel_size=kernel_size,
                            padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Permute(dims=(0, 2, 1)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def calc_same_padding(kernel_size: int):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=2,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        context_size=-1,
        attn_dropout=0.,
        ff_dropout=0.,
        conv_dropout=0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            context_size=context_size,
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNormAttn(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context_size):
        x = self.ff1(x) + x
        x = self.attn(x, context_size) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


class GraniteSpeechPretrainedModel(PreTrainedModel):
    config_class = GraniteSpeechConfig
    _supports_cache_class = True

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


class GraniteSpeechForConditionalGeneration(GraniteSpeechPretrainedModel, GenerationMixin):
    def __init__(self, config: GraniteSpeechConfig, is_legacy=False, skip_lora=True):
        if is_legacy:
            self._legacy_load(config, skip_lora)
        else:
            self._transformers_load(config)

    def _transformers_load(self, config: GraniteSpeechConfig):
        super().__init__(config)
        # NOTE: It doesn't matter when we initialize from config, but we should be careful
        # to make sure this does not pick up the adapter_config if in the future we use
        # from_pretrained or something similar, since that should be set by the composite
        # model; don't need to consider it twice
        self.language_model = AutoModelForCausalLM.from_config(config.llm_config)

        if self.language_model._tied_weights_keys is not None:
            # TODO - fix uninitialized lm head issues
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.encoder = CTCModel(config.encoder_config)
        self.projector = EncoderProjectorQFormer(config.projector_config)
        self.post_init()

    def _legacy_load(self, config: GraniteSpeechConfig, skip_lora=False):
        """NOTE: This should only be used for testing the model and converting;
        we should use the other loading logic, which does NOT explicitly create
        an encapsulated peft model, and instead handles it through the peft mixin
        if we have an adapter config present.
        """
        super().__init__(config)
        from peft import get_peft_model, LoraConfig, TaskType
        from transformers import GraniteForCausalLM

        self.language_model = GraniteForCausalLM.from_pretrained("ibm-granite/granite-3.1-8b-instruct")

        if not skip_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=64,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
            )
            self.language_model = get_peft_model(self.language_model, peft_config)

            lora_state_dict = torch.load(
                "data/lora_adapter.pt", map_location="cpu", weights_only=True
            )
            self.language_model.load_state_dict(lora_state_dict, strict=False)
        else:
            print("Did not load lora adapters!")
        self.encoder = CTCModel(config.encoder_config)
        self.projector = EncoderProjectorQFormer(config.projector_config)
        encoder_state_dict = torch.load(
            "data/encoder.pt", map_location="cpu", weights_only=True
        )
        self.encoder.load_state_dict(encoder_state_dict, strict=False)

        projector_state_dict = torch.load(
            "data/projector.pt", map_location="cpu", weights_only=True
        )
        self.projector.load_state_dict(projector_state_dict, strict=True)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_audio_features(self, input_features):
        encoder_embeds = self.encoder(input_features)
        projected_embeds = self.projector(encoder_embeds, None)
        return projected_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
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
    ):
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
            # Get the audio features from the encoder / projector 
            audio_features = self.get_audio_features(input_features)

            # Merge the audio features into the LLM embeddings
            inputs_embeds = self.get_merged_audio_embeddings(
                input_ids=input_ids,
                audio_features=audio_features,
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
            labels=labels,
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

    def get_merged_audio_embeddings(self, input_ids, audio_features):
        """
        Adds the audio token to the model's LLM vocabulary so that we can pass it
        through the tokenizer; it's assumed that the embeddings corresponding to the
        <|audio|> token will be clobbered with speech features.

        TODO - This needs to be adapted to handle batches of variable length sequences
        and potentially labels.
        """
        is_audio_index = input_ids == self.config.audio_token_index
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.language_model.get_input_embeddings()(
            llm_input_ids
        )  # [bsz, # features, hidden size]

        # Mask the audio features into the text embeddings
        special_audio_mask = is_audio_index.unsqueeze(-1)
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(
            special_audio_mask,
            audio_features,
        )
        return inputs_embeds

    def generate(self, input_features=None, **kwargs):
        """This model is expected to have a lora adapater, which is only
        enabled when considering audio inputs. As such, we override generate
        to conditionally enable / disable the lora adapter based on whether
        or not any input features were provided.
        """
        if input_features is not None:
            self.enable_adapters()
        else:
            self.disable_adapters()
        return super().generate(input_features=input_features, **kwargs)

__all__ = [
    "GraniteSpeechForConditionalGeneration",
    "GraniteSpeechPretrainedModel",
]
