# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""XLNet configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="xlnet/xlnet-large-cased")
@strict
class XLNetConfig(PreTrainedConfig):
    r"""
    ff_activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the If string, `"gelu"`, `"relu"`, `"silu"` and
        `"gelu_new"` are supported.
    attn_type (`str`, *optional*, defaults to `"bi"`):
        The attention type used by the model. Set `"bi"` for XLNet, `"uni"` for Transformer-XL.
    mem_len (`int` or `None`, *optional*):
        The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous
        forward pass won't be re-computed. See the
        [quickstart](https://huggingface.co/transformers/quickstart.html#using-the-past) for more information.
    reuse_len (`int`, *optional*):
        The number of tokens in the current batch to be cached and reused in the future.
    use_mems_eval (`bool`, *optional*, defaults to `True`):
        Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.
    use_mems_train (`bool`, *optional*, defaults to `False`):
        Whether or not the model should make use of the recurrent memory mechanism in train mode.
        <Tip>
        For pretraining, it is recommended to set `use_mems_train` to `True`. For fine-tuning, it is recommended to
        set `use_mems_train` to `False` as discussed
        [here](https://github.com/zihangdai/xlnet/issues/41#issuecomment-505102587). If `use_mems_train` is set to
        `True`, one has to make sure that the train batches are correctly pre-processed, *e.g.* `batch_1 = [[This
        line is], [This is the]]` and `batch_2 = [[ the first line], [ second line]]` and that all batches are of
        equal size.
        </Tip>
    bi_data (`bool`, *optional*, defaults to `False`):
        Whether or not to use bidirectional input pipeline. Usually set to `True` during pretraining and `False`
        during finetuning.
    clamp_len (`int`, *optional*, defaults to -1):
        Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no clamping.
    same_length (`bool`, *optional*, defaults to `False`):
        Whether or not to use the same attention length for each token.
    summary_type (`str`, *optional*, defaults to "last"):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Has to be one of the following options:
            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_last_dropout (`float`, *optional*, defaults to 0.1):
        Used in the sequence classification and multiple choice models.
        The dropout ratio to be used after the projection and activation.
    start_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    end_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.

    Examples:

    ```python
    >>> from transformers import XLNetConfig, XLNetModel

    >>> # Initializing a XLNet configuration
    >>> configuration = XLNetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XLNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlnet"
    keys_to_ignore_at_inference = ["mems"]
    attribute_map = {
        "n_token": "vocab_size",  # Backward compatibility
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 32000
    d_model: int = 1024
    n_layer: int = 24
    n_head: int = 16
    d_inner: int = 4096
    d_head: int | None = None
    ff_activation: str = "gelu"
    attn_type: str = "bi"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    dropout: float | int = 0.1
    mem_len: int | None = 512
    reuse_len: int | None = None
    use_mems_eval: bool = True
    use_mems_train: bool = False
    bi_data: bool = False
    clamp_len: int = -1
    same_length: bool = False
    summary_type: str = "last"
    summary_use_proj: bool = True
    summary_activation: str = "tanh"
    summary_last_dropout: float | int = 0.1
    start_n_top: int = 5
    end_n_top: int = 5
    pad_token_id: int | None = 5
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.d_head = self.d_head or self.d_model // self.n_head
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.d_model % self.n_head != 0:
            raise ValueError(f"'d_model % n_head' ({self.d_model % self.n_head}) should be equal to 0")
        if self.d_head != self.d_model // self.n_head:
            raise ValueError(
                f"`d_head` ({self.d_head}) should be equal to `d_model // n_head` ({self.d_model // self.n_head})"
            )

    @property
    def max_position_embeddings(self):
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        return -1

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        # Message copied from Transformer-XL documentation
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )


__all__ = ["XLNetConfig"]
