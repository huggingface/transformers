from transformers.models.gemma.modeling_gemma import GemmaForSequenceClassification
from transformers.models.llama.configuration_llama import LlamaConfig


# Example where we only want to only modify the docstring
class MyNewModel2Config(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`GemmaModel`]. It is used to instantiate an Gemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma-7B.
    e.g. [google/gemma-7b](https://huggingface.co/google/gemma-7b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GemmaModel`]
    ```python
    >>> from transformers import GemmaModel, GemmaConfig
    >>> # Initializing a Gemma gemma-7b style configuration
    >>> configuration = GemmaConfig()
    >>> # Initializing a model from the gemma-7b style configuration
    >>> model = GemmaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""


# Example where alllllll the dependencies are fetched to just copy the entire class
class MyNewModel2ForSequenceClassification(GemmaForSequenceClassification):
    pass
