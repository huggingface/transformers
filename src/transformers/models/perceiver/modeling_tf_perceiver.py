import abc
import math
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
from transformers.modeling_tf_utils import shape_list
from transformers.activations_tf import ACT2FN
from transformers.modeling_tf_outputs import TFBaseModelOutputWithCrossAttentions
from ...modeling_tf_utils import TFPreTrainedModel
from .configuration_perceiver import PerceiverConfig
from typing import Dict, Mapping, Callable, Any, Optional, Tuple
from ...file_utils import ModelOutput
from ...modeling_tf_utils import get_initializer, TFMaskedLanguageModelingLoss, TFSequenceClassificationLoss


ModalitySizeType = Mapping[str, int]
PreprocessorOutputType = Tuple[tf.Tensor, Optional[tf.Tensor], tf.Tensor]
PreprocessorType = Callable[..., PreprocessorOutputType]
PostprocessorType = Callable[..., Any]


@dataclass
class TFPerceiverModelOutput(ModelOutput):
    logits: tf.Tensor = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFPerceiverMaskedLMOutput(ModelOutput):
    """
    Base class for Perceiver's masked language model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_latents,
            num_latents)`. Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class PerceiverDecoderOutput(ModelOutput):
    """
    Base class for Perceiver decoder outputs, with potential cross-attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Output of the basic decoder.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    logits: tf.Tensor = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFPerceiverClassifierOutput(ModelOutput):
    """
    Base class for Perceiver's outputs of sequence/image classification models, optical flow and multimodal
    autoencoding.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None


class TFPerceiverSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self, config, is_cross_attention=False, qk_channels=None, v_channels=None, num_heads=1, q_dim=None, kv_dim=None
    ):
        super(TFPerceiverSelfAttention, self).__init__()
        self.num_heads = num_heads

        if qk_channels is None:
            qk_channels = q_dim

        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

        self.query = tf.keras.layers.Dense(qk_channels)
        self.key = tf.keras.layers.Dense(qk_channels)
        self.value = tf.keras.layers.Dense(v_channels)

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size, channels_per_head):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, channels_per_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
        training=False,
    ):
        batch_size = shape_list(hidden_states)[0]

        hidden_states = self.layernorm1(hidden_states)

        is_cross_attention = False
        if tf.is_tensor(inputs):
            inputs = self.layernorm2(inputs)
            is_cross_attention = True

        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        queries = self.transpose_for_scores(queries, batch_size, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, batch_size, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, batch_size, self.v_channels_per_head)

        attention_scores = tf.matmul(queries, keys, transpose_b=True)
        dk = tf.cast(self.qk_channels_per_head, dtype=attention_scores.dtype)

        _, _, _, v_head_dim = shape_list(values)
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        attention_probs = self.dropout(attention_probs, training=training)

        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        context_layer = tf.matmul(attention_probs, values)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        new_context_layer_shape = shape_list(context_layer)[:-2] + [hiddens]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TFPerceiverSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, output_channels):
        super(TFPerceiverSelfOutput, self).__init__()
        self.dense = tf.keras.layers.Dense(output_channels)

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states


class TFPerceiverAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
    ):
        super(TFPerceiverAttention, self).__init__()

        if is_cross_attention and qk_channels is None:
            if config.cross_attention_shape_for_attention == "q":
                qk_channels = q_dim
            elif config.cross_attention_shape_for_attention == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {config.cross_attention_shape_for_attention} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels

        self.self = TFPerceiverSelfAttention(
            config=config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )

        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        else:
            if output_channels is None:
                output_channels = v_channels

        self.layer_output = TFPerceiverSelfOutput(config, output_channels)
        self.use_query_residual = use_query_residual

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        attention_output = self.layer_output(self_outputs[0])

        if self.use_query_residual:
            attention_output = tf.add(attention_output, hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TFPerceiverMLP(tf.keras.layers.Layer):
    def __init__(self, config, input_size, widening_factor):
        super(TFPerceiverMLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(widening_factor * input_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = tf.keras.layers.Dense(input_size)

    def call(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class TFPerceiverLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super(TFPerceiverLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TFPerceiverAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.mlp = TFPerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]

        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)

        layer_output = tf.add(layer_output, attention_output)

        outputs = (layer_output,) + outputs

        return outputs


class TFPerceiverEncoder(tf.keras.layers.Layer):
    def __init__(self, config, kv_dim=None):
        super(TFPerceiverEncoder, self).__init__()
        self.config = config

        if config.d_latents % config.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_self_attend_heads ({config.num_self_attention_heads})."
            )
        if config.d_latents % config.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({config.num_cross_attention_heads})."
            )

        self.cross_attention = TFPerceiverLayer(
            config,
            is_cross_attention=True,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            num_heads=config.num_cross_attention_heads,
            q_dim=config.d_latents,
            kv_dim=kv_dim,
            widening_factor=config.cross_attention_widening_factor,
            use_query_residual=config.use_query_residual,
        )

        self.self_attention_layers = []
        for _ in range(config.num_self_attends_per_block):
            layer = TFPerceiverLayer(
                config,
                is_cross_attention=False,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_self_attention_heads,
                q_dim=config.d_latents,
                kv_dim=config.d_latents,
                widening_factor=config.self_attention_widening_factor,
            )
            self.self_attention_layers.append(layer)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        layer_outputs = self.cross_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        for _ in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attention_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return TFBaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class TFPerceiverEmbeddings(tf.keras.layers.Layer):
    """Construct the latent embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def build(self, input_shape):
        self.latents = self.add_weight(
            shape=(self.config.num_latents, self.config.d_latents),
            initializer=get_initializer(self.config.initializer_range),
        )
        super().build(input_shape)

    def call(self, batch_size):
        x = tf.reshape(self.latents, (1, self.config.num_latents, self.config.d_latents))
        x = tf.tile(x, [batch_size, 1, 1])
        return x


class TFPerceiverPreTrainedModel(TFPreTrainedModel):
    config_class = PerceiverConfig
    base_model_prefix = "perceiver"


class TFPerceiverModel(TFPerceiverPreTrainedModel):
    def __init__(
        self,
        config,
        decoder=None,
        input_preprocessor: PreprocessorType = None,
        output_postprocessor: PostprocessorType = None,
    ):
        super().__init__(config)
        self.config = config

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = TFPerceiverEmbeddings(config)
        self.encoder = TFPerceiverEncoder(
            config, kv_dim=input_preprocessor.num_channels if input_preprocessor is not None else config.d_model
        )
        self.decoder = decoder

    def get_input_embeddings(self):
        return self.embeddings.latents

    def set_input_embeddings(self, value):
        self.embeddings.latents = value

    def call(
        self,
        inputs,
        attention_mask=None,
        subsampled_output_points=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.input_preprocessor is not None:
            inputs, modality_sizes, inputs_without_pos = self.input_preprocessor(inputs)
        else:
            modality_sizes = None
            inputs_without_pos = None
            d = shape_list(inputs)[-1]
            if d != self.config.d_model:
                raise ValueError(
                    f"Last dimension of the inputs: {d} doesn't correspond to config.d_model: {self.config.d_model}. "
                    "Make sure to set config.d_model appropriately."
                )

        batch_size, seq_length, _ = shape_list(inputs)

        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length), value=1.0)
        attention_mask_shape = shape_list(attention_mask)
        extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=tf.float32)

        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * (self.config.num_blocks * self.config.num_self_attends_per_block)

        embedding_output = self.embeddings(batch_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        if self.decoder:
            if subsampled_output_points is not None:
                output_modality_sizes = {
                    "audio": subsampled_output_points["audio"].shape[0],
                    "image": subsampled_output_points["image"].shape[0],
                    "label": 1,
                }
            else:
                output_modality_sizes = None
            decoder_query = self.decoder.decoder_query(
                inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_output_points
            )
            decoder_outputs = self.decoder(
                decoder_query,
                z=sequence_output,
                query_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )
            logits = decoder_outputs.logits

            if output_attentions and decoder_outputs.cross_attentions is not None:
                if return_dict:
                    encoder_outputs.cross_attentions = (
                        encoder_outputs.cross_attentions,
                        decoder_outputs.cross_attentions,
                    )
                else:
                    encoder_outputs = encoder_outputs + decoder_outputs.cross_attentions

            if self.output_postprocessor:
                logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)

        if not return_dict:
            if logits is not None:
                return (logits, sequence_output) + encoder_outputs[1:]
            else:
                return (sequence_output,) + encoder_outputs[1:]

        return TFPerceiverModelOutput(
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class TFPerceiverForMaskedLM(TFPerceiverPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config):
        super(TFPerceiverForMaskedLM, self).__init__(config)

        text_preprocessor = TFPerceiverTextPreprocessor(config)

        trainable_position_encoding_kwargs_decoder = dict(
            num_channels=text_preprocessor.num_channels, index_dims=config.max_position_embeddings
        )

        self.perceiver = TFPerceiverModel(
            config,
            input_preprocessor=text_preprocessor,
            decoder=TFPerceiverBasicDecoder(
                config,
                output_num_channels=config.d_latents,
                output_index_dims=config.max_position_embeddings,  # we need to define the seq_len of the inputs beforehand
                num_channels=text_preprocessor.num_channels,
                qk_channels=8 * 32,
                v_channels=text_preprocessor.num_channels,
                num_heads=8,
                use_query_residual=False,
                final_project=False,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
            ),
        )
        self.embedding_decoder = TFPerceiverEmbeddingDecoder(config)

    def call(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        input_ids=None,
    ):
        if inputs is not None and input_ids is not None:
            raise ValueError("You cannot use both `inputs` and `input_ids`")
        elif inputs is None and input_ids is not None:
            inputs = input_ids

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.embedding_decoder(
            hidden_states=outputs.logits if return_dict else outputs[0],
            embedding_layer=self.perceiver.input_preprocessor.embeddings,
        )

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.compute_loss(labels=labels, logits=logits)
            # loss_fct = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # -100 index = padding token
            # masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return TFPerceiverMaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class TFPerceiverForSequenceClassification(TFPerceiverPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config):
        super(TFPerceiverForSequenceClassification, self).__init__(config)

        trainable_position_encoding_kwargs_decoder = dict(num_channels=config.d_latents, index_dims=1)

        self.num_labels = config.num_labels
        self.perceiver = TFPerceiverModel(
            config,
            input_preprocessor=TFPerceiverTextPreprocessor(config),
            decoder=TFPerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

    def call(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        input_ids=None,
    ):
        if inputs is not None and input_ids is not None:
            raise ValueError("You cannot use both `inputs` and `input_ids`")
        elif inputs is None and input_ids is not None:
            inputs = input_ids

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            loss = self.compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFPerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class TFPerceiverForImageClassificationLearned(TFPerceiverPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config):
        super(TFPerceiverForImageClassificationLearned, self).__init__(config)

        trainable_position_encoding_kwargs_preprocessor = dict(num_channels=256, index_dims=config.image_size ** 2)
        trainable_position_encoding_kwargs_decoder = dict(num_channels=config.d_latents, index_dims=1)

        self.num_labels = config.num_labels
        self.perceiver = TFPerceiverModel(
            config,
            input_preprocessor=TFPerceiverImagePreprocessor(
                config,
                prep_type="conv1x1",
                spatial_downsample=1,
                out_channels=256,
                position_encoding_type="trainable",
                concat_or_add_pos="concat",
                project_pos_dim=256,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_preprocessor,
            ),
            decoder=TFPerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

    def call(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        pixel_values=None,
    ):
        if inputs is not None and pixel_values is not None:
            raise ValueError("You cannot use both `inputs` and `pixel_values`")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            loss = self.compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFPerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class TFPerceiverForImageClassificationFourier(TFPerceiverPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config):
        super(TFPerceiverForImageClassificationFourier, self).__init__(config)

        fourier_position_encoding_kwargs_preprocessor = dict(
            concat_pos=True, max_resolution=(224, 224), num_bands=64, sine_only=False
        )
        trainable_position_encoding_kwargs_decoder = dict(num_channels=config.d_latents, index_dims=1)

        self.num_labels = config.num_labels
        self.perceiver = TFPerceiverModel(
            config,
            input_preprocessor=TFPerceiverImagePreprocessor(
                config,
                prep_type="pixels",
                spatial_downsample=1,
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=TFPerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

    def call(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        pixel_values=None,
    ):
        if inputs is not None and pixel_values is not None:
            raise ValueError("You cannot use both `inputs` and `pixel_values`")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            loss = self.compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFPerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class TFPerceiverForImageClassificationConvProcessing(TFPerceiverPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config):
        super(TFPerceiverForImageClassificationConvProcessing, self).__init__(config)

        fourier_position_encoding_kwargs_preprocessor = dict(
            concat_pos=True, max_resolution=(56, 56), num_bands=64, sine_only=False
        )
        trainable_position_encoding_kwargs_decoder = dict(num_channels=config.d_latents, index_dims=1)

        self.num_labels = config.num_labels
        self.perceiver = TFPerceiverModel(
            config,
            input_preprocessor=TFPerceiverImagePreprocessor(
                config,
                prep_type="conv",
                spatial_downsample=1,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
            ),
            decoder=TFPerceiverClassificationDecoder(
                config,
                num_channels=config.d_latents,
                trainable_position_encoding_kwargs=trainable_position_encoding_kwargs_decoder,
                use_query_residual=True,
            ),
        )

    def call(
        self,
        inputs=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        pixel_values=None,
    ):
        if inputs is not None and pixel_values is not None:
            raise ValueError("You cannot use both `inputs` and `pixel_values`")
        elif inputs is None and pixel_values is not None:
            inputs = pixel_values
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            loss = self.compute_loss(labels=labels, logits=logits)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFPerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

class TFPerceiverForOpticalFlow:
    pass


class TFPerceiverForMultimodalAutoencoding:
    pass


def build_position_encoding(
    position_encoding_type,
    out_channels=None,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
):

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = TFPerceiverTrainablePositionEncoding(**trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # We don't use the index_dims argument, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = TFPerceiverFourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")

    # Optionally, project the position encoding to a target dimension:
    positions_projection = tf.keras.layers.Dense(project_pos_dim) if project_pos_dim > 0 else tf.identity

    return output_pos_enc, positions_projection


class TFPerceiverAbstractDecoder(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_query_channels(self):
        raise NotImplementedError

    @abc.abstractmethod
    def call(self, query, z, query_mask=None):
        raise NotImplementedError


class TFPerceiverProjectionDecoder:
    pass


class TFPerceiverBasicDecoder(TFPerceiverAbstractDecoder):
    def __init__(
        self,
        config,
        output_num_channels,
        position_encoding_type="trainable",
        # The following 2 arguments are ignored if position_encoding_type == 'none':
        output_index_dims=None,
        num_channels=128,
        subsampled_index_dims=None,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        widening_factor=1,
        use_query_residual=False,
        concat_preprocessed_input=False,
        final_project=True,
        position_encoding_only=False,
        **position_encoding_kwargs,
    ):
        super(TFPerceiverBasicDecoder, self).__init__()
        self.output_num_channels = output_num_channels
        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs
        if position_encoding_type != "none":
            self.output_position_encodings, self.positions_projection = build_position_encoding(
                position_encoding_type=position_encoding_type, **position_encoding_kwargs
            )

        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only

        # for multimodal autoencoding, we don't need the decoder cross-attention and final layer
        # so then we will set position_encoding_only to True
        if not self.position_encoding_only:
            self.decoding_cross_attention = TFPerceiverLayer(
                config,
                is_cross_attention=True,
                qk_channels=qk_channels,
                v_channels=v_channels,
                num_heads=num_heads,
                q_dim=num_channels,
                kv_dim=config.d_latents,
                widening_factor=widening_factor,
                use_query_residual=use_query_residual,
            )
            self.final_layer = tf.keras.layers.Dense(output_num_channels) if final_project else tf.identity

    @property
    def num_query_channels(self) -> int:
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError(
                "You cannot calculate number of decoder query channels when position_encoding_type is set to none"
            )
        if self.position_encoding_only:
            if "project_pos_dim" in self.position_encoding_kwargs:
                return self.position_encoding_kwargs["project_pos_dim"]
            return self.output_position_encodings.output_size()
        if self.final_project:
            return self.output_num_channels
        return self.num_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
        if subsampled_points is not None:
            # subsampled_points are the indices if the inputs would be flattened
            # however, the inputs aren't flattened, that's why we use unravel_index
            # to get the indices for the unflattened array
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            indices = [x for x in tf.unravel_index(indices=subsampled_points, dims=self.output_index_dims)]
            pos = tf.stack(indices, axis=1)
            batch_size = inputs.shape[0]
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / tf.convert_to_tensor(self.output_index_dims)[None, :]
            pos = tf.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, pos=pos
                )

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = tf.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]

            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(index_dims, batch_size, device=inputs.device)

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)

        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
            pos_emb = tf.concat([inputs_without_pos, pos_emb], axis=-1)

        return pos_emb

    def call(self, query, z, query_mask=None, output_attentions=False):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        cross_attentions = () if output_attentions else None

        layer_outputs = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
            output_attentions=output_attentions,
        )
        output = layer_outputs[0]

        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)

        logits = self.final_layer(output)

        return PerceiverDecoderOutput(logits=logits, cross_attentions=cross_attentions)


class TFPerceiverClassificationDecoder(TFPerceiverAbstractDecoder):
    def __init__(self, config, **decoder_kwargs):
        super(TFPerceiverClassificationDecoder, self).__init__()

        self.num_labels = config.num_labels
        self.decoder = TFPerceiverBasicDecoder(
            config,
            output_num_channels=self.num_labels,
            output_index_dims=1,  # Predict a single logit array.
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points
        )

    def call(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        # B x 1 x num_classes -> B x num_classes
        logits = decoder_outputs.logits[:, 0, :]

        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)


class TFPerceiverOpticalFlowDecoder:
    pass


class TFPerceiverBasicVideoAutoencodingDecoder:
    pass


class TFPerceiverMultimodalDecoder:
    pass


class TFPerceiverAbstractPositionEncoding(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
    """Perceiver abstract position encoding."""

    @property
    @abc.abstractmethod
    def num_dimensions(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def output_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def call(self, batch_size, pos):
        raise NotImplementedError


class TFPerceiverTrainablePositionEncoding(TFPerceiverAbstractPositionEncoding):
    """Trainable position encoding."""

    def __init__(self, index_dims, num_channels=128):
        super(TFPerceiverTrainablePositionEncoding, self).__init__()
        self._num_channels = num_channels
        self._index_dims = index_dims
        self._emb_dim = tf.reduce_prod(self._index_dims)

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            shape=(self._emb_dim, self._num_channels),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
        )
        super().build(input_shape)

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        return self._num_channels

    def call(self, batch_size):
        position_embeddings = self.position_embeddings
        if batch_size is not None:
            position_embeddings = tf.reshape(position_embeddings, (1, self._emb_dim, self._num_channels))
            position_embeddings = tf.tile(position_embeddings, [batch_size, 1, 1])
        return position_embeddings


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = tf.stack([tf.linspace(start=min_freq, stop=res / 2, num=num_bands) for res in max_resolution], axis=0)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = tf.reshape(per_pos_features, [-1, tf.reduce_prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = tf.math.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = tf.concat(
            [tf.math.sin(np.pi * per_pos_features), tf.math.cos(np.pi * per_pos_features)], axis=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        npos, dpos = shape_list(per_pos_features)
        per_pos_features = tf.reshape(per_pos_features, (1, npos, dpos))
        per_pos_features = tf.tile(per_pos_features, [batch_size, 1, 1])
        per_pos_features = tf.concat([pos, per_pos_features], axis=-1)
    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    def _linspace(n_xels_per_dim):
        return tf.linspace(start=output_range[0], stop=output_range[1], num=n_xels_per_dim)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = tf.meshgrid(*dim_ranges)

    return tf.stack(array_index_grid, axis=-1)


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = tf.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = tf.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        if pos.shape[-1] != len(index_dims):
            raise ValueError("Spatial features have the wrong number of dimensions.")
    return pos


class TFPerceiverFourierPositionEncoding(TFPerceiverAbstractPositionEncoding):
    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super(TFPerceiverFourierPositionEncoding, self).__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def call(self, index_dims, batch_size, pos=None):
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        )
        return fourier_pos_enc


class TFAbstractPreprocessor(tf.keras.layers.Layer):
    @property
    def num_channels(self) -> int:
        """Returns size of preprocessor output."""
        raise NotImplementedError()


class TFPerceiverTextPreprocessor(TFAbstractPreprocessor):
    def __init__(self, config) -> None:
        super(TFPerceiverTextPreprocessor, self).__init__()
        self.config = config
        self.embeddings = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.d_model)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=config.max_position_embeddings, output_dim=config.d_model
        )

    @property
    def num_channels(self) -> int:
        return self.config.d_model

    def call(self, inputs):
        embeddings = self.embeddings(inputs)

        seq_length = inputs.shape[1]
        position_ids = tf.range(start=0, limit=seq_length, delta=1, dtype=tf.float32)
        embeddings = tf.add(embeddings, self.position_embeddings(position_ids))
        return embeddings, None, None


class TFPerceiverEmbeddingDecoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(TFPerceiverEmbeddingDecoder, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size), initializer="zeros")
        super().build(input_shape)

    def call(self, hidden_states, embedding_layer):
        batch_size, seq_len, d_model = shape_list(hidden_states)
        output = tf.matmul(
            tf.reshape(hidden_states, [-1, d_model]), embedding_layer.get_weights()[0], transpose_b=True
        )  # Flatten batch dim
        output = tf.add(output, self.bias)

        return tf.reshape(output, [batch_size, seq_len, self.vocab_size])


class TFPerceiverMultimodalPostprocessor:
    pass


class TFPerceiverClassificationPostprocessor:
    pass


class TFPerceiverAudioPostprocessor:
    pass


class TFPerceiverProjectionPostprocessor:
    pass


def space_to_depth(frames: tf.Tensor, temporal_block_size: int = 1, spatial_block_size: int = 1) -> tf.Tensor:
    """
    Space to depth transform. Rearranges blocks of spatial data, into depth.

    This function assumes the channels to be first, but will place the channels last after transformation.

    Based on https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15.
    """
    frames_shape = shape_list(frames)
    frames_dim = len(frames_shape)

    if frames_dim == 4:
        batch_size, num_channels, height, width = frames_shape
        # split up dimensions (height by spatial_block_size, width by spatial_block_size)
        frames = tf.transpose(
            frames,
            perm=[
                batch_size,
                num_channels,
                height // spatial_block_size,
                spatial_block_size,
                width // spatial_block_size,
                spatial_block_size,
            ],
        )
        # move blocks to last dimension: (batch_size, H//bs, W//bs, bs, bs, C)
        frames = tf.transpose(frames, perm=[0, 2, 4, 3, 5, 1])
        # concatenate blocks along channel dimension: (batch_size, H//bs, W//bs, bs*bs*C)
        frames = tf.transpose(
            frames,
            perm=[
                batch_size,
                height // spatial_block_size,
                width // spatial_block_size,
                (spatial_block_size ** 2) * num_channels,
            ],
        )
        return frames
    elif frames_dim == 5:
        batch_size, time, num_channels, height, width = frames_shape
        # split up dimensions (time by temporal_block_size, height by spatial_block_size, width by spatial_block_size)
        frames = tf.transpose(
            frames,
            perm=[
                batch_size,
                time // temporal_block_size,
                temporal_block_size,
                num_channels,
                height // spatial_block_size,
                spatial_block_size,
                width // spatial_block_size,
                spatial_block_size,
            ],
        )
        # move blocks to last dimension: (batch_size, T//ts, H//bs, W//bs, ts, bs, bs, C)
        frames = tf.transpose(frames, perm=[0, 1, 4, 6, 2, 5, 7, 3])
        # concatenate blocks along channel dimension: (batch_size, T//ts, H//bs, W//bs, ts*bs*bs*C)
        frames = tf.transpose(
            frames,
            perm=[
                batch_size,
                time // temporal_block_size,
                height // spatial_block_size,
                width // spatial_block_size,
                temporal_block_size * (spatial_block_size ** 2) * num_channels,
            ],
        )
        return frames
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, channels, height, width)"
            " or rank 5 (batch, time, channels, height, width)"
        )


class TFPerceiverImagePreprocessor(TFAbstractPreprocessor):
    def __init__(
        self,
        config,
        prep_type="conv",
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        position_encoding_type: str = "fourier",
        in_channels: int = 3,
        out_channels: int = 64,
        conv_after_patching: bool = False,
        conv_after_patching_in_channels: int = 54,  # only relevant when conv_after_patching = True
        conv2d_use_batchnorm: bool = True,
        concat_or_add_pos: str = "concat",
        project_pos_dim: int = -1,
        **position_encoding_kwargs,
    ):
        super(TFPerceiverImagePreprocessor, self).__init__()
        self.config = config

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError(f"Prep_type {prep_type} is invalid")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.in_channels = in_channels
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        if self.prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial and 1 expected for temporal downsampling with conv."
                )
            self.convnet = TFConv2DDownsample(
                in_channels=in_channels,
                num_layers=int(convnet_num_layers),
                out_channels=out_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )
        elif self.prep_type == "conv1x1":
            if temporal_downsample != 1:
                raise ValueError("Conv1x1 does not downsample in time.")
            self.convnet_1x1 = tf.keras.Sequential(
                [
                    tf.keras.layers.Permute([2, 3, 1]),
                    tf.keras.layers.Conv2D(
                        filters=out_channels,
                        kernel_size=1,
                        # spatial_downsample is unconstrained for 1x1 convolutions.
                        strides=(spatial_downsample, spatial_downsample),
                    ),
                    tf.keras.layers.Permute([3, 1, 2]),
                ]
            )

        # Position embeddings
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

        # Optional convolutional layer after patches.
        self.conv_after_patches = tf.keras.layers.Dense(self.out_channels) if conv_after_patching else tf.identity

    @property
    def num_channels(self) -> int:
        # Let's assume that the number of resolutions (in the context of image preprocessing)
        # of the input data is 2 or 3 depending on whether we are processing image or video respectively.
        # In this case, for convenience, we will declare is_temporal variable,
        # which will show whether the data has a temporal dimension or not.
        is_temporal = self.position_embeddings.num_dimensions > 2

        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim

        # inputs
        if self.conv_after_patching or self.prep_type in ("conv1x1", "conv"):
            inp_dim = self.out_channels
        elif self.prep_type == "pixels":
            inp_dim = self.in_channels
            if not is_temporal:
                inp_dim = math.ceil(inp_dim / self.spatial_downsample)
        elif self.prep_type == "patches":
            if self.conv_after_patching:
                inp_dim = self.out_channels
            else:
                inp_dim = self.in_channels * self.spatial_downsample ** 2
                if is_temporal:
                    inp_dim *= self.temporal_downsample

        return inp_dim + pos_dim

    def _build_network_inputs(self, inputs: tf.Tensor, pos: tf.Tensor, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.

        """
        inputs_shape = shape_list(inputs)
        batch_size = inputs_shape[0]
        index_dims = inputs_shape[1:-1]
        indices = tf.reduce_prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs_shape) > 3 and network_input_is_1d:
            inputs = tf.reshape(inputs, [batch_size, indices, -1])

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            pos_enc = tf.reshape(pos_enc, inputs_shape[:-1] + [-1])
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = tf.concat([inputs, pos_enc], axis=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = tf.add(inputs, pos_enc)

        return inputs_with_pos, inputs

    def call(self, inputs: tf.Tensor, pos: Optional[tf.Tensor] = None, network_input_is_1d: bool = True):
        inputs_shape = shape_list(inputs)
        inputs_dim = len(inputs_shape)

        if self.prep_type == "conv":
            # Convnet image featurization.
            # Downsamples spatially by a factor of 4
            inputs = self.convnet(inputs)

        elif self.prep_type == "conv1x1":
            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)

        elif self.prep_type == "pixels":
            # if requested, downsamples in the crudest way
            if inputs_dim == 4:
                inputs = inputs[:: self.spatial_downsample, :: self.spatial_downsample]
            elif inputs_dim == 5:
                inputs = inputs[
                    :, :: self.temporal_downsample, :, :: self.spatial_downsample, :: self.spatial_downsample
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        elif self.prep_type == "patches":
            # Space2depth featurization.
            # Video: B x T x C x H x W
            inputs = space_to_depth(
                inputs, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample
            )

            if inputs_dim == 5 and inputs_shape[1] == 1:
                # for flow
                inputs = inputs.squeeze(dim=1)

            # Optionally apply conv layer.
            inputs = self.conv_after_patches(inputs)

        if self.prep_type != "patches":
            # move channels to last dimension, as the _build_network_inputs method below expects this
            if inputs_dim == 4:
                inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
            elif inputs.ndim == 5:
                inputs = tf.transpose(inputs, perm=[0, 1, 4, 3, 2])
            else:
                raise ValueError("Unsupported data format for conv1x1.")

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        modality_sizes = None  # Size for each modality, only needed for multimodal

        return inputs, modality_sizes, inputs_without_pos


class TFPerceiverOneHotPreprocessor:
    pass


class TFPerceiverAudioPreprocessor:
    pass


class TFPerceiverMultimodalPreprocessor:
    pass


class TFConv2DDownsample(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int = 1,
        in_channels: int = 3,
        out_channels: int = 64,
        use_batchnorm: bool = True,
    ):
        super(TFConv2DDownsample, self).__init__()

        # use channel_first to be same as pytroch !!!!!???
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=7, strides=2, use_bias=False, padding="same"
        )
        self.batchnorm = tf.keras.layers.BatchNormalization() if use_batchnorm else tf.identity
        self.relu = tf.keras.layers.ReLU()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)

    def call(self, inputs):
        out = tf.transpose(inputs, perm=[0, 2, 3, 1])  # convert to NHWC, because NCHW not working on cpu :|
        out = self.conv(inputs)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = tf.transpose(inputs, perm=[0, 3, 1, 2])  # back to NCHW
        return out
