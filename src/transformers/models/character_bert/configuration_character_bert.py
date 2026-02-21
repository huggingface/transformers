# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""CharacterBERT model configuration."""

from ..bert.configuration_bert import BertConfig


class CharacterBertConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`CharacterBertModel`]. It is used to instantiate
    a CharacterBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    [helboukkouri/character-bert-general](https://huggingface.co/helboukkouri/character-bert-general).

    Configuration objects inherit from [`~PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`~PreTrainedConfig`] for more information.

    Args:
        character_embedding_dim (`int`, *optional*, defaults to 16):
            Character embedding dimension used before the convolution stack.
        character_vocab_size (`int`, *optional*, defaults to 262):
            Number of supported character IDs before applying the +1 offset for masking/padding.
        max_characters_per_token (`int`, *optional*, defaults to 50):
            Maximum number of characters represented for each token.
        character_cnn_filters (`tuple[tuple[int, int], ...]`, *optional*):
            Defaults to `((1, 32), (2, 32), (3, 64), (4, 128), (5, 256), (6, 512), (7, 1024))`.
            Convolution widths and output channels used in the character CNN.
        num_highway_layers (`int`, *optional*, defaults to 2):
            Number of highway layers applied after the convolution outputs.
        mlm_vocab_size (`int`, *optional*):
            Legacy alias for `vocab_size` in original CharacterBERT checkpoints.

    Example:

    ```python
    >>> from transformers import CharacterBertConfig, CharacterBertModel

    >>> configuration = CharacterBertConfig()
    >>> model = CharacterBertModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "character_bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
        is_decoder=False,
        add_cross_attention=False,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        character_embedding_dim=16,
        character_vocab_size=262,
        max_characters_per_token=50,
        character_cnn_filters=((1, 32), (2, 32), (3, 64), (4, 128), (5, 256), (6, 512), (7, 1024)),
        num_highway_layers=2,
        **kwargs,
    ):
        legacy_character_embedding_dim = kwargs.pop("character_embeddings_dim", None)
        legacy_character_cnn_filters = kwargs.pop("cnn_filters", None)
        legacy_max_characters_per_token = kwargs.pop("max_word_length", None)
        legacy_mlm_vocab_size = kwargs.pop("mlm_vocab_size", None)

        if legacy_character_embedding_dim is not None and character_embedding_dim == 16:
            character_embedding_dim = legacy_character_embedding_dim
        if legacy_character_cnn_filters is not None and character_cnn_filters == (
            (1, 32),
            (2, 32),
            (3, 64),
            (4, 128),
            (5, 256),
            (6, 512),
            (7, 1024),
        ):
            character_cnn_filters = legacy_character_cnn_filters
        if legacy_max_characters_per_token is not None and max_characters_per_token == 50:
            max_characters_per_token = legacy_max_characters_per_token

        if legacy_mlm_vocab_size is not None and vocab_size == 30522:
            vocab_size = legacy_mlm_vocab_size

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            is_decoder=is_decoder,
            add_cross_attention=add_cross_attention,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.character_embedding_dim = character_embedding_dim
        self.character_vocab_size = character_vocab_size
        self.max_characters_per_token = max_characters_per_token
        self.character_cnn_filters = tuple((int(width), int(channels)) for width, channels in character_cnn_filters)
        self.num_highway_layers = num_highway_layers

    @property
    def mlm_vocab_size(self) -> int:
        # Legacy alias kept for backward compatibility with older CharacterBERT checkpoints.
        return self.vocab_size

    @mlm_vocab_size.setter
    def mlm_vocab_size(self, value: int) -> None:
        self.vocab_size = value


__all__ = ["CharacterBertConfig"]
