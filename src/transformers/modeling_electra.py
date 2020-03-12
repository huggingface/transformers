import torch

from transformers import PreTrainedModel, BertConfig
from transformers.activations import get_activation
from .modeling_bert import BertModel, BertEmbeddings, BertLayerNorm, BertEncoder
import torch.nn as nn

import logging
import math
import os

logger = logging.getLogger(__name__)


def load_tf_weights_in_electra(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        original_name = name

        name = name.replace("electra/embeddings/", "embeddings/")
        name = name.replace("electra", "discriminator")
        name = name.replace("dense_1", "dense_prediction")
        name = name.replace("discriminator/embeddings_project", "discriminator_embeddings_project")
        name = name.replace("generator/embeddings_project", "generator_embeddings_project")
        name = name.replace("generator_predictions/output_bias", "bias")

        name = name.split("/")
        print(original_name, name)
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
                "temperature",
            ]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            print("transposing", original_name)
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape, original_name
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class ElectraEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        assert config.hidden_size % 2 == 0
        self.word_embeddings = nn.Embedding(config.vocab_size, int(config.hidden_size / 2), padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, int(config.hidden_size / 2))
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, int(config.hidden_size / 2))

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(int(config.hidden_size / 2), eps=config.layer_norm_eps)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds

class ElectraDiscriminatorPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        return self.dense_prediction(hidden_states)


class ElectraGeneratorPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = BertLayerNorm(int(config.hidden_size / 2))
        self.dense = nn.Linear(config.hidden_size, int(config.hidden_size / 2))

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class ElectraMainLayer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = BertEncoder(config)

    def forward(self, embedding_output, attention_mask=None, head_mask=None):
        encoder_outputs, all_selves, attention_scores = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        return encoder_outputs


class ElectraModel(PreTrainedModel):

    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ElectraEmbeddings(config)

        self.discriminator = ElectraMainLayer(config)
        self.discriminator_embeddings_project = nn.Linear(int(config.hidden_size / 2), config.hidden_size)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

        self.generator = ElectraMainLayer(config)
        self.generator_embeddings_project = nn.Linear(int(config.hidden_size / 2), config.hidden_size)
        self.generator_predictions = ElectraGeneratorPredictions(config)
        self.generator_lm_head = nn.Linear(int(config.hidden_size / 2), config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # self.generator_lm_head.bias = self.bias

        self.config = config

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.generator_lm_head

    @staticmethod
    def _gather_positions(sequence, positions):
        batch_size, sequence_length, dimension = sequence.shape
        position_shift = (sequence_length * torch.arange(batch_size)).unsqueeze(-1)
        flat_positions = torch.reshape(positions + position_shift, [-1]).long()
        flat_sequence = torch.reshape(sequence, [batch_size * sequence_length, dimension])
        gathered = flat_sequence.index_select(0, flat_positions)
        return torch.reshape(gathered, [batch_size, -1, dimension])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        masked_lm_positions=None,
        masked_lm_ids=None,
        masked_lm_weights=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output, input_embeds = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        discriminator_embedding_output = self.discriminator_embeddings_project(embedding_output)
        generator_embedding_output = self.generator_embeddings_project(embedding_output)

        discriminator_hidden_states = self.discriminator(
            discriminator_embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask
        )
        generator_hidden_states = self.generator(
            generator_embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask
        )

        return_dict = {
            "input_embeds": input_embeds,
            "embeddings_output": embedding_output,
            "discriminator_embedding_output": discriminator_embedding_output,
            "generator_embedding_output": generator_embedding_output,
            "discriminator_sequence_output": discriminator_hidden_states[0],
            "generator_sequence_output": generator_hidden_states[0],
            "generator_pooled_output": generator_hidden_states[0][:, 0]
        }

        if self.config.output_hidden_states:
            return_dict["discriminator_hidden_states"] = discriminator_hidden_states[-1]
            return_dict["generator_hidden_states"] = generator_hidden_states[-1]

        # Masked language modeling softmax layer
        if masked_lm_weights is not None:
            relevant_hidden = self._gather_positions(generator_hidden_states[0], masked_lm_positions)
            hidden_states = self.generator_predictions(relevant_hidden)
            # hidden_states = self.generator_lm_head(hidden_states)
            hidden_states = torch.matmul(hidden_states, self.embeddings.word_embeddings.weight.T)
            return_dict['x'] = hidden_states
            hidden_states = hidden_states + self.bias
            return_dict["logits"] = hidden_states

            probs = torch.softmax(hidden_states, dim=-1)
            log_probs = torch.log_softmax(hidden_states, -1)
            preds = torch.argmax(log_probs, dim=-1)

            return_dict["probs"] = probs
            return_dict["preds"] = preds

        return return_dict
