from ..llava.configuration_llava import LlavaConfig
from ..llava.modeling_llava import LlavaModel, LlavaForConditionalGeneration
import torch
from typing import Optional, Union
from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING

class FastVlmConfig(LlavaConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastVLMForConditionalGeneration`]. It is used to instantiate a
    FastVLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the FastVLM-7B.

    e.g. [KamilaMila/FastVLM-7B](https://huggingface.co/KamilaMila/FastVLM-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:  TODO !!!!!!!!!!!
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to -2):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.

    Example:

    ```python
    >>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava llava-1.5-7b style configuration
    >>> configuration = LlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-1.5-7b style configuration
    >>> model = LlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "fast_vlm"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=151646,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        image_seq_length=256,
        multimodal_projector_bias=True,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length

        if vision_feature_select_strategy != "full":
            raise ValueError(
                "Only vision_feature_select_strategy='full' supported in FastVLM!"
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "timm_wrapper")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["timm_wrapper"](
                architecture="fastvit_mci3",
                do_pooling=True,
                global_pool="avg",
                hidden_size=3072,
                initializer_range=0.02,
                model_args={"inference_mode": True}
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        PretrainedConfig.__init__(**kwargs)

class FastVlmModel(LlavaModel):
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, list[int]]`, *optional*):
                The index/indices of the layer to select the vision feature.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone. 
                Only "full" supported.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # only this value makes sense in FastVLM
        if vision_feature_select_strategy != "full":
            raise ValueError(f"Unexpected select feature strategy: {vision_feature_select_strategy}, Only 'full' is supported.")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_tower(pixel_values, **kwargs)  # add more choice here! 

        # since the vision tower is hybrid in FastVLM, its output needs to be handled differently from Llava
        selected_image_feature = image_outputs.last_hidden_state 
        selected_image_feature = selected_image_feature.flatten(2).permute(0, 2, 1)

        image_features = self.multi_modal_projector(selected_image_feature)

        if "image_sizes" in kwargs:
            split_sizes = [
                (height // self.vision_tower.patch_size) * (width // self.vision_tower.patch_size)
                for height, width in kwargs["image_sizes"]
            ]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        return image_features
    
class FastVlmForConditionalGeneration(LlavaForConditionalGeneration):
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FastVlmForConditionalGeneration

        >>> model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/fast_vlm-1.5-7b-hf") #TODO change!!!
        >>> processor = AutoProcessor.from_pretrained("fast_vlm-hf/fast_vlm-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""
        super().forward(**super_kwargs)
        

__all__ = ["FastVlmForConditionalGeneration", "FastVlmModel", "FastVlmPreTrainedModel", "FastVlmConfig"]