# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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

from typing import Union

from ...configuration_utils import PretrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)
from ...utils import (
    auto_docstring,
    is_torch_available,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..idefics.modeling_idefics import IdeficsBaseModelOutputWithPast, IdeficsCausalLMOutputWithPast
from ..janus.image_processing_janus import JanusImageProcessor
from ..janus.image_processing_janus_fast import JanusImageProcessorFast
from ..janus.modeling_janus import JanusForConditionalGeneration, JanusModel, JanusPreTrainedModel


if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


class DeepseekVLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekVLModel`]. It is used to instantiate a
    DeepseekVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DeepseekVL
    [deepseek-community/deepseek-vl-1.3b-chat](https://huggingface.co/deepseek-community/deepseek-vl-1.3b-chat) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SiglipVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 100015):
            The index representing image tokens in the model's token vocabulary.

    Example:

    ```python
    >>> from transformers import DeepseekVLConfig, DeepseekVLModel

    >>> # Initializing a DeepseekVL deepseek-community/deepseek-vl-1.3b-chat style configuration
    >>> configuration = DeepseekVLConfig()

    >>> # Initializing a model (with random weights) from the deepseek-community/deepseek-vl-1.3b-chat style configuration
    >>> model = DeepseekVLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        text_config: AutoConfig = None,
        vision_config: AutoConfig = None,
        image_token_id: int = 100015,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `LlamaConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `SiglipVisionConfig` with default values.")

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id


class DeepseekVLBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast):
    pass


class DeepseekVLCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast):
    pass


class DeepseekVLAligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_features = config.vision_config.hidden_size
        out_features = config.text_config.hidden_size

        self.linear1 = nn.Linear(in_features, out_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, vision_encodings: torch.Tensor) -> torch.Tensor:
        x = self.linear1(vision_encodings)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class DeepseekVLPreTrainedModel(JanusPreTrainedModel):
    _no_split_modules = ["LlamaDecoderLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        # Required only for Linear layer in DeepseekVLAligner
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


@auto_docstring
class DeepseekVLModel(JanusModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vision_model = AutoModel.from_config(config.vision_config)
        self.aligner = DeepseekVLAligner(config)

        self.language_model = AutoModel.from_config(config=config.text_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing.
        self.post_init()

        del self.vqmodel
        del self.generation_embeddings
        del self.generation_aligner
        del self.generation_head


class DeepseekVLForConditionalGeneration(JanusForConditionalGeneration):
    def prepare_embeddings_for_image_generation(self):
        raise AttributeError("Not needed for DeepseekVL")

    def decode_image_tokens(self):
        raise AttributeError("Not needed for DeepseekVL")

    def generate(self):
        raise AttributeError("Not needed for DeepseekVL")


class DeepseekVLImageProcessor(JanusImageProcessor):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def postprocess(self):
        raise AttributeError("Not needed for DeepseekVL")

    def unnormalize(self):
        raise AttributeError("Not needed for DeepseekVL")


class DeepseekVLImageProcessorFast(JanusImageProcessorFast):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def postprocess(self):
        raise AttributeError("Not needed for DeepseekVL")


class DeepseekVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "common_kwargs": {"return_tensors": "pt"},
    }


class DeepseekVLProcessor(ProcessorMixin):
    r"""
    Constructs a DeepseekVL processor which wraps a DeepseekVL Image Processor and a Llama tokenizer into a single processor.

    [`DeepseekVLProcessor`] offers all the functionalities of [`DeepseekVLImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~DeepseekVLProcessor.__call__`] and [`~DeepseekVLProcessor.decode`] for more information.

    Args:
        image_processor ([`DeepseekVLImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        num_image_tokens (`int`, *optional*, defaults to 576):
            The number of special image tokens used as placeholders for visual content in text sequences.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "num_image_tokens"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        num_image_tokens=576,
    ):
        self.image_token = tokenizer.image_token
        self.num_image_tokens = num_image_tokens

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        images: ImageInput = None,
        **kwargs: Unpack[DeepseekVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        DeepseekVLImageProcessor's [`~DeepseekVLImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            DeepseekVLProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = []
        one_img_tokens = self.image_token * self.num_image_tokens
        for prompt in text:
            prompt = prompt.replace(self.image_token, one_img_tokens)
            prompt_strings.append(prompt)

        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # process images if pixel_values are provided
        if images is not None:
            data["pixel_values"] = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = [
    "DeepseekVLConfig",
    "DeepseekVLPreTrainedModel",
    "DeepseekVLModel",
    "DeepseekVLForConditionalGeneration",
    "DeepseekVLImageProcessor",
    "DeepseekVLImageProcessorFast",
    "DeepseekVLProcessor",
]
