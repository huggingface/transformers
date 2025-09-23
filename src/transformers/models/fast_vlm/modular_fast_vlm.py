from ..llava.configuration_llava import LlavaConfig
from ..llava.modeling_llava import LlavaModel, LlavaForConditionalGeneration
import torch
from typing import Optional, Union

class FastVlmConfig(LlavaConfig):
    model_type = "fast_vlm"

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
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
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

        if vision_feature_select_strategy != "full":
            raise ValueError(f"Unexpected select feature strategy: {vision_feature_select_strategy}")
        
        if vision_feature_layer != -1:
            raise ValueError(f"Unexpected vision feature layer: {vision_feature_layer}")


        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

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
    pass

__all__ = ["FastVlmForConditionalGeneration", "FastVlmModel", "FastVlmPreTrainedModel", "FastVlmConfig"]