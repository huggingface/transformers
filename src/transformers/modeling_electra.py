import torch
from .modeling_bert import BertModel, BertEmbeddings, BertLayerNorm
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

        name = name.split("/")
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
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
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
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, generator_hidden_states):
        hidden_states = self.LayerNorm(generator_hidden_states)
        hidden_states = self.dense(hidden_states)

        return hidden_states


class ElectraModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = ElectraEmbeddings(config)

        self.discriminator = BertModel(config)
        self.discriminator_embeddings_project = nn.Linear(int(config.hidden_size / 2), config.hidden_size)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

        self.generator = BertModel(config)
        self.generator_embeddings_project = nn.Linear(int(config.hidden_size / 2), config.hidden_size)
        self.generator_predictions = ElectraGeneratorPredictions(config)

        self.config = config

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

        embedding_output, input_embeds = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        discriminator_embedding_output = self.discriminator_embeddings_project(embedding_output)
        generator_embedding_output = self.generator_embeddings_project(embedding_output)

        discriminator_hidden_states = self.discriminator(
            inputs_embeds=discriminator_embedding_output, attention_mask=attention_mask
        )
        generator_hidden_states = self.generator(
            inputs_embeds=generator_embedding_output, attention_mask=attention_mask
        )

        return_dict = {
            "input_embeds": input_embeds,
            "embeddings_output": embedding_output,
            "discriminator_embedding_output": discriminator_embedding_output,
            "generator_embedding_output": generator_embedding_output,
            "discriminator_output": discriminator_hidden_states[0],
            "generator_output": generator_hidden_states[0],
            "generator_selves": generator_hidden_states[-2]
        }

        if self.config.output_hidden_states:
            return_dict["discriminator_hidden_states"] = discriminator_hidden_states[-1]
            return_dict["generator_hidden_states"] = generator_hidden_states[-1]

        return return_dict
