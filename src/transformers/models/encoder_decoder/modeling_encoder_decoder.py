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
"""Classes to support Encoder-Decoder architectures"""

import inspect
import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig


logger = logging.get_logger(__name__)


DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@auto_docstring
class EncoderDecoderModel(PreTrainedModel, GenerationMixin):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    config: EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(
        self,
        config: PreTrainedConfig | None = None,
        encoder: PreTrainedModel | None = None,
        decoder: PreTrainedModel | None = None,
    ):
        r"""
        encoder (`PreTrainedModel`, *optional*):
            The encoder model to use.
        decoder (`PreTrainedModel`, *optional*):
            The decoder model to use.
        """
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if getattr(config.decoder, "cross_attention_hidden_size", None) is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            from ..auto.modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from ..auto.modeling_auto import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        # update `_attn_implementation` because the attn is set in a deepcopied config within PreTrainedModel
        self.config.encoder._attn_implementation = self.encoder.config._attn_implementation
        self.config.decoder._attn_implementation = self.decoder.config._attn_implementation
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and getattr(self.decoder.config, "cross_attention_hidden_size", None) is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError(
                "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
                "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
            )

        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        if module in self.encoder.modules():
            self.encoder._init_weights(module)
        elif module in self.decoder.modules():
            self.decoder._init_weights(module)

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str | None = None,
        decoder_pretrained_model_name_or_path: str | None = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import EncoderDecoderModel

        >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./bert2bert")
        >>> # load fine-tuned model
        >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder:
            del kwargs["encoder_" + key]
        for key in kwargs_decoder:
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if getattr(encoder_config, "is_decoder", False) or getattr(
                    encoder_config, "add_cross_attention", False
                ):
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and causal mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )
            else:
                decoder_config = kwargs_decoder["config"]

            if (
                getattr(decoder_config, "is_decoder", None) is False
                or getattr(decoder_config, "add_cross_attention", None) is False
            ):
                logger.info(
                    f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                    f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                    f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                )
                decoder_config.is_decoder = True
                decoder_config.add_cross_attention = True

            kwargs_decoder["config"] = decoder_config
            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        encoder_outputs: tuple[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Examples:

        ```python
        >>> from transformers import EncoderDecoderModel, BertTokenizer
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google-bert/bert-base-uncased", "google-bert/bert-base-uncased"
        ... )  # initialize Bert2Bert from pre-trained checkpoints

        >>> # training
        >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
        >>> model.config.pad_token_id = tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
        >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2bert")
        >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

        >>> # generation
        >>> generated = model.generate(input_ids)
        ```"""
        # `record outputs` can rely on the absence of the kwarg to retrieve whether the config should be used or not
        # Hence, we use this workaround to allow for defaults to work as expected
        kwargs_shared = {key: kwargs[key] for key in ["output_attentions", "output_hidden_states"] if key in kwargs}

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_encoder = kwargs_encoder | kwargs_shared

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)
        kwargs_decoder = kwargs_decoder | kwargs_shared

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and getattr(self.decoder.config, "cross_attention_hidden_size", None) is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            if decoder_attention_mask is None:
                decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id).to(decoder_input_ids.dtype)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )


__all__ = ["EncoderDecoderModel"]
